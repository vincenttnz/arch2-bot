"""
core/nn/inference.py
====================
Neural inference engine — loads trained models and exposes clean APIs
for the bot loop.

Falls back gracefully to heuristic methods when models are not available
or torch is not installed.  This enables a smooth Phase 0 → Phase 9 hybrid
transition where each neural component replaces its heuristic counterpart
as soon as a trained model file appears on disk.

Public API
──────────
    engine = NeuralEngine(models_root, data_root)

    # Screen classification
    screen = engine.classify_screen(frame)          # → "combat" | "skill" | …

    # Skill card recognition  (replaces OCR primary)
    name, conf = engine.classify_card(icon_bgr)     # → ("front arrow", 0.97)

    # Combat features (player pos, enemies, projectiles)
    feats = engine.detect_combat(frame)

    # Dodge action
    action_name, idx = engine.dodge_action(frame, combat_feats)

    # Skill selection
    pick_idx = engine.select_skill(matches, hp_ratio, wave)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False

from core.nn.models import (
    SCREEN_CLASSES, NUM_SCREEN_CLASSES,
    DODGE_ACTIONS, NUM_DODGE_ACTIONS,
    STATE_DIM, INPUT_H, INPUT_W, CARD_H, CARD_W,
)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bgr_to_tensor(img: np.ndarray, h: int, w: int) -> "torch.Tensor":
    """Resize BGR → (1,3,h,w) float32 tensor normalised 0-1."""
    resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)


def _frame_to_state(frame: np.ndarray, h: int = 84, w: int = 84) -> "torch.Tensor":
    """Resize BGR → (1,3,h,w) for state encoder."""
    return _bgr_to_tensor(frame, h, w)


# ─────────────────────────────────────────────────────────────────────────────
# Neural Engine
# ─────────────────────────────────────────────────────────────────────────────

class NeuralEngine:
    """
    Unified inference interface.  Each component lazy-loads its model file
    on first use; if the file doesn't exist it logs once and uses heuristics.
    """

    MODEL_FILES = {
        "screen":    "screen_model.pt",
        "skill":     "skill_card_model.pt",
        "skill_cls": "skill_card_classes.json",
        "combat":    "combat_detector.pt",
        "dodge":     "dodge_policy.pt",
        "selection": "selection_policy.pt",
    }

    # Confidence threshold below which we fall back to heuristic result
    FALLBACK_THRESHOLD = 0.60

    def __init__(self, models_root: Path | str, data_root: Path | str | None = None):
        self.models_root = Path(models_root)
        self.data_root   = Path(data_root) if data_root else None
        self._models: dict[str, Any] = {}
        self._warned: set[str] = set()
        self._skill_classes: list[str] = []
        self.device = "cpu"

        # Perf counters
        self.screen_uses_neural  = False
        self.skill_uses_neural   = False
        self.dodge_uses_neural   = False

    # ── Internal loaders ──────────────────────────────────────────────────────

    def _warn_once(self, key: str, msg: str):
        if key not in self._warned:
            self._warned.add(key)
            print(f"[NeuralEngine] {msg}")

    def _load_screen_model(self):
        if "screen" in self._models:
            return self._models["screen"]
        if not _TORCH:
            self._warn_once("screen", "torch not installed — using heuristic screen detection")
            return None
        path = self.models_root / self.MODEL_FILES["screen"]
        if not path.exists():
            self._warn_once("screen", f"screen_model.pt not found at {path} — using heuristic")
            return None
        from core.nn.models import ScreenClassifier
        m = ScreenClassifier(NUM_SCREEN_CLASSES)
        ck = torch.load(str(path), map_location=self.device)
        m.load_state_dict(ck["model_state"])
        m.eval()
        self._models["screen"] = m
        print(f"[NeuralEngine] Loaded screen classifier (val_acc={ck.get('val_acc',0):.3f})")
        self.screen_uses_neural = True
        return m

    def _load_skill_model(self):
        if "skill" in self._models:
            return self._models["skill"], self._skill_classes
        if not _TORCH:
            self._warn_once("skill", "torch not installed — using OCR/template for skills")
            return None, []
        pt_path  = self.models_root / self.MODEL_FILES["skill"]
        cls_path = self.models_root / self.MODEL_FILES["skill_cls"]
        if not pt_path.exists():
            self._warn_once("skill", f"skill_card_model.pt not found — using OCR/template")
            return None, []
        if not cls_path.exists():
            self._warn_once("skill", f"skill_card_classes.json not found — cannot load skill model")
            return None, []
        skill_classes = json.loads(cls_path.read_text())
        from core.nn.models import SkillCardClassifier
        m = SkillCardClassifier(num_skills=len(skill_classes))
        ck = torch.load(str(pt_path), map_location=self.device)
        m.load_state_dict(ck["model_state"])
        m.eval()
        self._models["skill"] = m
        self._skill_classes = skill_classes
        print(f"[NeuralEngine] Loaded skill classifier ({len(skill_classes)} skills, "
              f"val_acc={ck.get('val_acc',0):.3f})")
        self.skill_uses_neural = True
        return m, skill_classes

    def _load_dodge_policy(self):
        if "dodge" in self._models:
            return self._models["dodge"]
        if not _TORCH:
            self._warn_once("dodge", "torch not installed — using geometric dodge")
            return None
        path = self.models_root / self.MODEL_FILES["dodge"]
        if not path.exists():
            self._warn_once("dodge", f"dodge_policy.pt not found — using geometric dodge")
            return None
        from core.nn.models import StateEncoder, DodgePolicy
        encoder = StateEncoder(num_frames=1)
        policy  = DodgePolicy(state_dim=STATE_DIM)
        ck = torch.load(str(path), map_location=self.device)
        encoder.load_state_dict(ck["encoder"])
        policy.load_state_dict(ck["policy"])
        encoder.eval(); policy.eval()
        self._models["dodge"] = (encoder, policy)
        print("[NeuralEngine] Loaded dodge policy")
        self.dodge_uses_neural = True
        return encoder, policy

    # ── Public methods ─────────────────────────────────────────────────────────

    def classify_screen(self, frame: np.ndarray, heuristic_result: str = "UNKNOWN") -> str:
        """
        Classify game screen.  Returns uppercase string matching GameStateDetector.
        Falls back to heuristic_result if model unavailable or confidence too low.
        """
        model = self._load_screen_model()
        if model is None:
            return heuristic_result

        with torch.no_grad():
            x   = _bgr_to_tensor(frame, INPUT_H, INPUT_W)
            idx, probs = model.predict(x)
        conf  = float(probs[0, idx])
        label = SCREEN_CLASSES[int(idx)]

        if conf < self.FALLBACK_THRESHOLD:
            return heuristic_result
        return label.upper()

    def classify_card(
        self, icon_bgr: np.ndarray
    ) -> tuple[str, float]:
        """
        Classify a skill card icon crop.
        Returns (skill_name_normalised, confidence).
        Returns ("unknown", 0.0) on failure.
        """
        model, classes = self._load_skill_model()
        if model is None or not classes:
            return "unknown", 0.0

        with torch.no_grad():
            x   = _bgr_to_tensor(icon_bgr, CARD_H, CARD_W)
            idx, probs = model.predict(x)
        conf  = float(probs[0, idx])
        label = classes[int(idx)]
        if conf < self.FALLBACK_THRESHOLD:
            return "unknown", 0.0
        return label.replace("_", " "), conf

    def dodge_action(
        self,
        frame: np.ndarray,
        combat_feats: dict,
        greedy: bool = False,
    ) -> tuple[str, int]:
        """
        Returns (action_name, action_idx) for combat movement.
        Falls back to geometric heuristic if model unavailable.
        """
        result = self._load_dodge_policy()
        if result is None:
            return self._geometric_dodge(combat_feats)
        encoder, policy = result

        with torch.no_grad():
            x     = _frame_to_state(frame)
            state = encoder(x)
            extras = self._build_extras(combat_feats)
            action_idx, _, _ = policy.act(state, extras, greedy=greedy)
        return DODGE_ACTIONS[action_idx], action_idx

    def _build_extras(self, feats: dict) -> "torch.Tensor":
        """Pack combat_feats dict into the 6-float extras vector."""
        hp            = float(feats.get("hp_ratio", 1.0))
        enemy_count   = min(float(feats.get("enemy_count", 0)) / 10.0, 1.0)
        nearest_dx    = float(feats.get("nearest_enemy_dx", 0.0))
        nearest_dy    = float(feats.get("nearest_enemy_dy", 0.0))
        proj_density  = min(float(feats.get("projectile_count", 0)) / 5.0, 1.0)
        pickup_nearby = float(feats.get("pickup_nearby", 0.0))
        arr = np.array([hp, enemy_count, nearest_dx, nearest_dy,
                        proj_density, pickup_nearby], dtype=np.float32)
        return torch.from_numpy(arr).unsqueeze(0)

    @staticmethod
    def _geometric_dodge(feats: dict) -> tuple[str, int]:
        """
        Heuristic fallback: move away from nearest enemy + projectile threat.
        Maps to the nearest of the 9 discrete dodge actions.
        """
        dx = -float(feats.get("nearest_enemy_dx", 0.0))
        dy = -float(feats.get("nearest_enemy_dy", 0.0))
        proj_dx = float(feats.get("proj_threat_dx", 0.0))
        proj_dy = float(feats.get("proj_threat_dy", 0.0))
        dx += proj_dx * 2.0
        dy += proj_dy * 2.0
        thresh = 0.15
        if abs(dx) < thresh and abs(dy) < thresh:
            return "idle", 0
        # Map to 8 directions
        if abs(dx) > abs(dy) * 1.5:
            return ("right", 4) if dx > 0 else ("left", 3)
        if abs(dy) > abs(dx) * 1.5:
            return ("down", 2) if dy > 0 else ("up", 1)
        if dx > 0 and dy > 0:
            return "down_right", 8
        if dx > 0 and dy < 0:
            return "up_right",   7
        if dx < 0 and dy > 0:
            return "down_left",  6
        return "up_left", 5

    def status(self) -> dict:
        return {
            "torch_available":   _TORCH,
            "screen_neural":     self.screen_uses_neural,
            "skill_neural":      self.skill_uses_neural,
            "dodge_neural":      self.dodge_uses_neural,
            "loaded_models":     list(self._models.keys()),
        }
