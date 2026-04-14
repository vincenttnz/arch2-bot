"""
core/skill_recogniser.py
=========================
Dual-stage skill-card recogniser.

Stage 1 – Perceptual Hash DB (pHash)
  • Built once by scanning core/templates/skills/<skill>/*.jpg|png
  • Saved to models/skill_hash_db.pkl  (~50 KB on disk)
  • Runtime: <1 ms per card, zero GPU, always available
  • Confidence: 1 − (Hamming distance / 256 bits)

Stage 2 – YOLOv8 image classifier  (optional, GPU)
  • Trained on data/skill_cards/ (78 k images, ~15 min on RTX 3060)
  • Saved to models/skill_classifier.pt
  • Used when pHash confidence < PHASH_THRESHOLD or pHash DB missing

Priority table
  Sourced from allclash.com/best-skills-in-archero-2/ and
  progameguides.com/archero-2/archero-2-tier-list/
  Scores: 10 = always pick, 0 = never pick (skip)

Public API
----------
    rec = SkillRecogniser(project_root)

    # Identify one card crop
    name, conf = rec.identify(card_bgr)

    # Pick best card from a full skill-select screen
    slot, name, conf = rec.pick_best_from_screen(screen_bgr, n_slots=3)
    # → slot is 0-based; send adb.click to that column

    # Rebuild the hash DB (call from GUI "Build Hash DB" button)
    rec.build_hash_db()
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO as _YOLO
    _YOLO_OK = True
except ImportError:
    _YOLO_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# SKILL PRIORITY TABLE
# ─────────────────────────────────────────────────────────────────────────────
# allclash.com/best-skills-in-archero-2/  +  progameguides.com tier list
# 10 = S-tier, always pick │ 9 = S-tier │ 8-7 = A-tier │ 5-4 = B-tier
# 2 = C-tier │ 0 = never pick (junk / error screens)

SKILL_PRIORITY: dict[str, int] = {
    # ── S-tier 10 ──────────────────────────────────────────────────────────
    "ricochet":             10,   # best DPS skill in game
    "multishot":            10,
    "front_arrow":          10,
    "diagonal_arrow":       10,
    "piercing_arrow":       10,
    # ── S-tier 9 ───────────────────────────────────────────────────────────
    "charged_arrow":         9,
    "lightwing_arrow":       9,
    "instant_strike":        9,
    "blitz_strike":          9,
    "bolt":                  9,
    "fairy_of_the_wing":     9,
    "venom":                 9,
    "super_venom":           9,
    "toxic_meteror":         9,   # game typo preserved
    # ── A-tier 8 ───────────────────────────────────────────────────────────
    "beam_strike":           8,
    "beam_circle":           8,
    # ── A-tier 7 ───────────────────────────────────────────────────────────
    "fire_circle":           7,
    "energy_ring":           7,
    "lightning_sprite":      7,
    "laser_sprite":          7,
    "ice_spike_sprite":      7,
    "bomb_sprite":           7,
    "vine_pursuit":          7,
    "warriors_breath":       7,
    "wind_blessing":         7,
    "cloudfooted":           7,
    "soul_of_swiftness":     7,
    "frenzy_potion":         7,
    "sprite_frenzy":         7,
    # ── B-tier 5 ───────────────────────────────────────────────────────────
    "plant_guardian":        5,
    "insect_lure":           5,
    "breath_of_wind":        5,
    "slow_field":            5,
    "vampiric_circle":       5,
    "poison_circle":         5,
    "stand_strong":          5,
    "demon_slayer":          5,
    "life_conversion":       5,
    # ── B-tier 4 ───────────────────────────────────────────────────────────
    "restore_hp":            4,
    "short_range_strike":    4,
    # ── C-tier 2 ───────────────────────────────────────────────────────────
    "atk_increase":          2,
    "combat_data":           2,
    "jump":                  2,
    "perilous_recovery":     2,
    "bolt_meteor":           2,
    "super_freeze":          2,
    "cat_s_purr":            2,
    # ── SKIP 0 – never pick ────────────────────────────────────────────────
    "error":                 0,
    "junk":                  0,
    "none":                  0,
    "start":                 0,
    "exact_skill_name":      0,
    "01":                    0,
    "_anchors":              0,
}

# Recognition thresholds
PHASH_THRESHOLD = 0.72   # below this → try YOLO
YOLO_THRESHOLD  = 0.50   # below this → fall back to pHash result

# Card slot geometry (fractions of screen dimensions)
# Archero 2 centres cards in a column; Y positions differ by layout.
_SLOT_Y_3  = [0.32, 0.52, 0.72]    # 3-card layout
_SLOT_Y_2  = [0.42, 0.62]           # 2-card layout
_CARD_W    = 0.55                    # card width as fraction of screen W
_CARD_H    = 0.17                    # card height as fraction of screen H


# ─────────────────────────────────────────────────────────────────────────────
# Hashing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dhash(img: np.ndarray, size: int = 16) -> np.ndarray:
    """Difference hash (dHash). Returns flat uint8 bit array of length size*size."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    resized = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff    = (resized[:, 1:] > resized[:, :-1]).astype(np.uint8)
    return diff.flatten()


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def _hash_conf(dist: int, bits: int = 256) -> float:
    return max(0.0, 1.0 - dist / bits)


# ─────────────────────────────────────────────────────────────────────────────
# SkillRecogniser
# ─────────────────────────────────────────────────────────────────────────────

class SkillRecogniser:
    """
    Dual-stage skill recogniser.

    Usage
    -----
    Instantiate once at bot startup:
        rec = SkillRecogniser(project_root="F:/arch2")

    The __init__ automatically:
      1. Tries to load models/skill_hash_db.pkl (pHash DB)
      2. Builds it from core/templates/skills/ if missing
      3. Loads models/skill_classifier.pt  (YOLO cls model, if present)
    """

    def __init__(self, project_root):
        self.root     = Path(project_root)
        self.db_path  = self.root / "models" / "skill_hash_db.pkl"
        self.clf_path = self.root / "models" / "skill_classifier.pt"
        self.tmpl_dir = self.root / "core" / "templates" / "skills"

        # {skill_name: [hash_array, …]}
        self._db: dict[str, list[np.ndarray]] = {}
        self._clf = None
        self._clf_names: list[str] = []

        self._load_or_build_db()
        self._load_classifier()

    # ── Hash DB ───────────────────────────────────────────────────────────────

    def _load_or_build_db(self):
        if self.db_path.exists():
            try:
                with open(self.db_path, "rb") as f:
                    self._db = pickle.load(f)
                total = sum(len(v) for v in self._db.values())
                print(f"[SkillRec] Hash DB loaded: {len(self._db)} skills "
                      f"({total} hashes)")
                return
            except Exception as e:
                print(f"[SkillRec] Hash DB corrupt ({e}), rebuilding …")
        self.build_hash_db()

    def build_hash_db(self) -> int:
        """
        Scan core/templates/skills/<name>/*.jpg|png, hash every image,
        save to models/skill_hash_db.pkl.  Returns number of skills indexed.

        Call this from the GUI "Build Hash DB" button or whenever new
        template images are added.
        """
        if not self.tmpl_dir.exists():
            print(f"[SkillRec] Template dir missing: {self.tmpl_dir}")
            return 0

        db: dict[str, list[np.ndarray]] = {}
        _skip = {"_anchors"}

        for skill_dir in sorted(self.tmpl_dir.iterdir()):
            if not skill_dir.is_dir() or skill_dir.name in _skip:
                continue
            hashes: list[np.ndarray] = []
            for p in skill_dir.iterdir():
                if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                img = cv2.imread(str(p))
                if img is not None:
                    hashes.append(_dhash(img))
            if hashes:
                db[skill_dir.name] = hashes

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump(db, f)

        self._db = db
        total = sum(len(v) for v in db.values())
        print(f"[SkillRec] Hash DB built: {len(db)} skills, {total} hashes "
              f"→ {self.db_path}")
        return len(db)

    # ── Classifier ────────────────────────────────────────────────────────────

    def _load_classifier(self):
        if not _YOLO_OK or not self.clf_path.exists():
            return
        try:
            self._clf = _YOLO(str(self.clf_path))
            self._clf_names = list(self._clf.names.values()) \
                              if hasattr(self._clf, "names") else []
            print(f"[SkillRec] Classifier loaded: {len(self._clf_names)} classes")
        except Exception as e:
            print(f"[SkillRec] Classifier load failed: {e}")
            self._clf = None

    # ── Internal recognition ──────────────────────────────────────────────────

    def _phash_identify(self, card: np.ndarray) -> tuple[str, float]:
        if not self._db:
            return "unknown", 0.0
        q = _dhash(card)
        best_name, best_dist = "unknown", 9999
        for name, hashes in self._db.items():
            for h in hashes:
                d = _hamming(q, h)
                if d < best_dist:
                    best_dist, best_name = d, name
        return best_name, _hash_conf(best_dist)

    def _clf_identify(self, card: np.ndarray) -> tuple[str, float]:
        if self._clf is None:
            return "unknown", 0.0
        try:
            res   = self._clf.predict(card, verbose=False, task="classify")
            probs = res[0].probs
            idx   = int(probs.top1)
            conf  = float(probs.top1conf)
            name  = self._clf.names.get(idx, "unknown")
            return name, conf
        except Exception:
            return "unknown", 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def identify(self, card: np.ndarray) -> tuple[str, float]:
        """
        Identify a single skill card crop.

        Returns (skill_name, confidence 0-1).
        Runs pHash first; escalates to YOLO classifier if confidence is low.
        """
        name1, conf1 = self._phash_identify(card)
        if conf1 >= PHASH_THRESHOLD:
            return name1, conf1

        name2, conf2 = self._clf_identify(card)
        if conf2 >= YOLO_THRESHOLD:
            return name2, conf2

        # Return whichever was more confident
        return (name2, conf2) if conf2 > conf1 else (name1, conf1)

    def pick_best_from_screen(
        self,
        screen: np.ndarray,
        n_slots: int = 3,
    ) -> tuple[int, str, float]:
        """
        Crop each card slot from a full-screen ADB frame, identify all,
        return (slot_index, skill_name, confidence) for the highest-priority card.

        slot_index is 0-based; use it to compute the x-coordinate to click.
        """
        y_fracs = _SLOT_Y_3 if n_slots >= 3 else _SLOT_Y_2
        best    = (0, "unknown", 0.0, -1)   # (slot, name, conf, prio)

        for i, yf in enumerate(y_fracs):
            crop        = _crop_card(screen, yf)
            name, conf  = self.identify(crop)
            prio        = SKILL_PRIORITY.get(name, 1)
            if prio > best[3] or (prio == best[3] and conf > best[2]):
                best = (i, name, conf, prio)

        return best[0], best[1], best[2]

    # ── Convenience ───────────────────────────────────────────────────────────

    def priority(self, skill_name: str) -> int:
        return SKILL_PRIORITY.get(skill_name, 1)

    def status(self) -> dict:
        return {
            "db_skills":  len(self._db),
            "db_hashes":  sum(len(v) for v in self._db.values()),
            "classifier": self.clf_path.name if self._clf else "none",
        }

    @property
    def has_model(self) -> bool:
        return self._clf is not None

    @property
    def has_db(self) -> bool:
        return bool(self._db)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helper
# ─────────────────────────────────────────────────────────────────────────────

def _crop_card(screen: np.ndarray, y_frac: float) -> np.ndarray:
    """Crop a single card bounding box from a full-screen BGR frame."""
    h, w = screen.shape[:2]
    cx   = w // 2
    cw   = int(w * _CARD_W)
    ch   = int(h * _CARD_H)
    cy   = int(h * y_frac)
    x1   = max(0, cx - cw // 2)
    x2   = min(w, cx + cw // 2)
    y1   = max(0, cy - ch // 2)
    y2   = min(h, cy + ch // 2)
    return screen[y1:y2, x1:x2]
