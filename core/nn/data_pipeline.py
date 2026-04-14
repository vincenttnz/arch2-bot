"""
core/nn/data_pipeline.py
========================
Structured data collection for training neural models.

Records every decision point with full context:
  - raw screenshot
  - detected screen state
  - detected skills / buttons
  - chosen action + click coords
  - reward delta
  - HP estimate
  - enemy / projectile counts

Output format
─────────────
data/
├── raw_frames/          YYYYMMDD_HHMMSS_ffffff.png
├── labeled_frames/      same name, annotated PNG
├── skill_cards/         {skill_name}/NNN.png   (icon crops, auto-organised)
├── episodes/            episode_NNN.jsonl       (one line per step)
└── replay/              replay_NNN.pkl          (numpy arrays for RL)

screen_labels.jsonl  — one JSON per labeled frame
skill_cards.jsonl    — one JSON per captured card crop
actions.jsonl        — one JSON per bot decision
"""
from __future__ import annotations

import json
import time
import threading
import pickle
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Frame Logger  (Phase 0 data collection)
# ─────────────────────────────────────────────────────────────────────────────

class FrameLogger:
    """
    Saves every skill-screen frame with its full annotation to disk.
    Runs in a background thread so it never blocks the bot loop.
    Implements Phase 0 logging requirements from the roadmap.
    """

    def __init__(self, data_root: Path | str):
        self.data_root     = Path(data_root)
        self._raw_dir      = self.data_root / "raw_frames"
        self._labeled_dir  = self.data_root / "labeled_frames"
        self._cards_dir    = self.data_root / "skill_cards"
        self._episodes_dir = self.data_root / "episodes"
        self._replay_dir   = self.data_root / "replay"
        for d in (self._raw_dir, self._labeled_dir, self._cards_dir,
                  self._episodes_dir, self._replay_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._screen_labels_path = self.data_root / "screen_labels.jsonl"
        self._skill_cards_path   = self.data_root / "skill_cards.jsonl"
        self._actions_path       = self.data_root / "actions.jsonl"

        self._queue: deque = deque(maxlen=512)
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ── Public API ─────────────────────────────────────────────────────────────

    def log_frame(
        self,
        frame: np.ndarray,
        screen: str,
        skills: list[dict] | None = None,
        action: dict | None = None,
        reward: float = 0.0,
        hp_ratio: float = 1.0,
        enemy_count: int = 0,
        projectile_count: int = 0,
        debug_frame: np.ndarray | None = None,
    ):
        """Queue a frame + metadata for async disk write."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        record = dict(
            stamp=stamp,
            frame=frame.copy() if frame is not None else None,
            debug_frame=debug_frame.copy() if debug_frame is not None else None,
            screen=screen,
            skills=skills or [],
            action=action or {},
            reward=reward,
            hp_ratio=hp_ratio,
            enemy_count=enemy_count,
            projectile_count=projectile_count,
            ts=time.time(),
        )
        with self._lock:
            self._queue.append(record)

    def log_skill_card(self, skill_name: str, icon_bgr: np.ndarray, confidence: float):
        """Save an icon crop to skill_cards/{skill_name}/."""
        if icon_bgr is None or icon_bgr.size == 0 or not skill_name:
            return
        skill_key = skill_name.lower().replace(" ", "_").replace("'", "")
        dest_dir  = self._cards_dir / skill_key
        dest_dir.mkdir(parents=True, exist_ok=True)
        existing  = len(list(dest_dir.glob("*.png")))
        dest_path = dest_dir / f"{existing + 1:04d}.png"
        icon_save = cv2.resize(icon_bgr, (112, 112), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dest_path), icon_save)
        entry = dict(
            skill=skill_name, path=str(dest_path),
            confidence=confidence, ts=time.time(),
        )
        self._append_jsonl(self._skill_cards_path, entry)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    # ── Background worker ──────────────────────────────────────────────────────

    def _worker(self):
        while not self._stop.is_set() or self._queue:
            try:
                with self._lock:
                    rec = self._queue.popleft() if self._queue else None
                if rec is None:
                    time.sleep(0.05)
                    continue
                self._flush(rec)
            except Exception as e:
                print(f"[FrameLogger] worker error: {e}")

    def _flush(self, rec: dict):
        stamp = rec["stamp"]
        frame = rec.get("frame")
        dbg   = rec.get("debug_frame")

        # Raw frame
        if frame is not None:
            cv2.imwrite(str(self._raw_dir / f"{stamp}.png"), frame)
        if dbg is not None:
            cv2.imwrite(str(self._labeled_dir / f"{stamp}.png"), dbg)

        # screen_labels.jsonl
        entry = {
            k: v for k, v in rec.items()
            if k not in ("frame", "debug_frame")
        }
        entry["raw_path"]     = str(self._raw_dir     / f"{stamp}.png")
        entry["labeled_path"] = str(self._labeled_dir / f"{stamp}.png")
        self._append_jsonl(self._screen_labels_path, entry)

        # actions.jsonl
        if rec.get("action"):
            self._append_jsonl(self._actions_path, {
                "stamp": stamp, "screen": rec["screen"],
                "ts": rec["ts"], **rec["action"],
                "reward": rec["reward"],
            })

    @staticmethod
    def _append_jsonl(path: Path, record: dict):
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Episode Recorder  (Phase 1 RL data pipeline)
# ─────────────────────────────────────────────────────────────────────────────

class EpisodeRecorder:
    """
    Records one game episode (start-to-death) as a sequence of transitions
    (state, action, reward, next_state, done).

    On episode end, flushes both a JSONL log and a numpy .npz replay file
    for use in offline RL training.
    """

    MAX_STEPS = 50_000   # safety cap

    def __init__(self, data_root: Path | str, episode_id: int | None = None):
        self.data_root  = Path(data_root)
        self.ep_id      = episode_id if episode_id is not None else int(time.time())
        self._steps: list[dict] = []
        self._frame_buf: list[np.ndarray] = []
        self._start_ts  = time.time()

    def step(
        self,
        screen: str,
        action_name: str,
        action_idx: int,
        click_xy: tuple[int, int],
        reward: float,
        done: bool,
        hp_ratio: float,
        enemy_count: int,
        skills_detected: list[str],
        skill_chosen: str,
        frame_small: np.ndarray | None = None,
    ):
        if len(self._steps) >= self.MAX_STEPS:
            return
        rec = dict(
            ep=self.ep_id,
            t=len(self._steps),
            ts=time.time() - self._start_ts,
            screen=screen,
            action=action_name,
            action_idx=action_idx,
            click_x=click_xy[0],
            click_y=click_xy[1],
            reward=reward,
            done=done,
            hp_ratio=hp_ratio,
            enemy_count=enemy_count,
            skills_detected=skills_detected,
            skill_chosen=skill_chosen,
        )
        self._steps.append(rec)
        if frame_small is not None:
            self._frame_buf.append(
                cv2.resize(frame_small, (84, 84), interpolation=cv2.INTER_AREA)
            )

    def flush(self) -> Path:
        """Write episode to disk and return the episode file path."""
        ep_dir = self.data_root / "episodes"
        ep_dir.mkdir(parents=True, exist_ok=True)
        ep_path = ep_dir / f"episode_{self.ep_id:06d}.jsonl"
        with ep_path.open("w", encoding="utf-8") as f:
            for s in self._steps:
                f.write(json.dumps(s, default=str) + "\n")

        # Numpy replay for RL
        if self._frame_buf:
            rp_dir  = self.data_root / "replay"
            rp_dir.mkdir(parents=True, exist_ok=True)
            rp_path = rp_dir / f"replay_{self.ep_id:06d}.npz"
            frames  = np.stack(self._frame_buf, axis=0)       # (T, 84, 84, 3)
            rewards = np.array([s["reward"]     for s in self._steps], dtype=np.float32)
            actions = np.array([s["action_idx"] for s in self._steps], dtype=np.int32)
            dones   = np.array([s["done"]       for s in self._steps], dtype=bool)
            np.savez_compressed(str(rp_path),
                                frames=frames, rewards=rewards,
                                actions=actions, dones=dones)
        return ep_path

    @property
    def total_reward(self) -> float:
        return sum(s["reward"] for s in self._steps)

    @property
    def num_steps(self) -> int:
        return len(self._steps)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder  (Phase 2 — builds training sets from logged frames)
# ─────────────────────────────────────────────────────────────────────────────

class DatasetBuilder:
    """
    Reads screen_labels.jsonl and skill_cards.jsonl and produces
    ready-to-train PyTorch datasets (or numpy arrays if torch unavailable).

    Usage
    ─────
        builder = DatasetBuilder(data_root)
        screen_ds = builder.build_screen_dataset()   # (X, y) arrays
        card_ds   = builder.build_card_dataset()
    """

    def __init__(self, data_root: Path | str):
        self.data_root = Path(data_root)

    def build_screen_dataset(
        self, target_size: tuple = (224, 224)
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Returns (X, y, class_names) where:
          X: float32 (N, H, W, 3) normalised 0-1
          y: int32   (N,)
        """
        from core.nn.models import SCREEN_CLASSES
        label_path = self.data_root / "screen_labels.jsonl"
        if not label_path.exists():
            return np.zeros((0, *target_size, 3), dtype=np.float32), np.zeros(0, dtype=np.int32), SCREEN_CLASSES

        xs, ys = [], []
        with label_path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                raw_path = rec.get("raw_path", "")
                screen   = rec.get("screen", "").lower()
                if screen not in SCREEN_CLASSES:
                    continue
                img = cv2.imread(raw_path)
                if img is None:
                    continue
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                xs.append(img)
                ys.append(SCREEN_CLASSES.index(screen))

        if not xs:
            return np.zeros((0, *target_size, 3), dtype=np.float32), np.zeros(0, dtype=np.int32), SCREEN_CLASSES
        return np.stack(xs), np.array(ys, dtype=np.int32), SCREEN_CLASSES

    def build_card_dataset(
        self, target_size: tuple = (112, 112)
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Returns (X, y, skill_names) from skill_cards/ directory tree.
        """
        cards_dir = self.data_root / "skill_cards"
        if not cards_dir.exists():
            return np.zeros((0, *target_size, 3), dtype=np.float32), np.zeros(0, dtype=np.int32), []

        skill_names: list[str] = sorted(
            d.name for d in cards_dir.iterdir() if d.is_dir()
        )
        xs, ys = [], []
        for idx, skill in enumerate(skill_names):
            skill_dir = cards_dir / skill
            for p in sorted(skill_dir.glob("*.png")):
                img = cv2.imread(str(p))
                if img is None:
                    continue
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                xs.append(img)
                ys.append(idx)

        if not xs:
            return np.zeros((0, *target_size, 3), dtype=np.float32), np.zeros(0, dtype=np.int32), skill_names
        return np.stack(xs), np.array(ys, dtype=np.int32), skill_names

    def summary(self) -> dict:
        label_path = self.data_root / "screen_labels.jsonl"
        card_path  = self.data_root / "skill_cards.jsonl"
        n_labels = sum(1 for _ in open(label_path)) if label_path.exists() else 0
        n_cards  = sum(1 for _ in open(card_path))  if card_path.exists()  else 0
        cards_dir = self.data_root / "skill_cards"
        skill_dirs = list(cards_dir.iterdir()) if cards_dir.exists() else []
        return {
            "labeled_frames": n_labels,
            "skill_card_entries": n_cards,
            "skills_with_data": len(skill_dirs),
            "total_card_images": sum(
                len(list(d.glob("*.png"))) for d in skill_dirs if d.is_dir()
            ),
        }
