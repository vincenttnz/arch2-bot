from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


class ValkyrieDetector:
    def __init__(self, project_root: str | Path | None = None) -> None:
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
        self.templates = self._load_templates()
        self._last_box = None

    def _candidate_paths(self) -> Iterable[Path]:
        base = self.project_root
        return [
            base / "core" / "templates" / "skills" / "valkyrie",
            base / "core" / "templates",
            base / "templates" / "skills" / "valkyrie",
            base / "templates",
        ]

    def _load_templates(self):
        images = []
        for root in self._candidate_paths():
            if not root.exists():
                continue
            if root.is_dir() and root.name == "valkyrie":
                pngs = sorted(root.glob("*.png"))
            else:
                pngs = []
                for old in ("valk_icon.png", "valk_icon_48.png", "valk_icon_64.png", "valk_icon_96.png", "valk_icon_128.png"):
                    p = root / old
                    if p.exists():
                        pngs.append(p)
            for path in pngs:
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
        return images

    def detect(self, frame, threshold: float = 0.80):
        if not self.templates:
            self._last_box = None
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        best = 0.0
        best_loc = None
        best_shape = None
        for template in self.templates:
            if template.shape[0] > gray.shape[0] or template.shape[1] > gray.shape[1]:
                continue
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if float(max_val) > best:
                best = float(max_val)
                best_loc = max_loc
                best_shape = template.shape
        if best >= threshold and best_loc is not None and best_shape is not None:
            x, y = best_loc
            h, w = best_shape
            self._last_box = (int(x), int(y), int(w), int(h))
            return self._last_box
        self._last_box = None
        return None

    def get_debug_box(self):
        return self._last_box
