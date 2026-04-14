from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


class TemplateMatcher:
    def __init__(self, template_name: str, project_root: str | Path | None = None) -> None:
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
        self.template = self._load_template(template_name)

    def _candidate_paths(self, template_name: str) -> Iterable[Path]:
        base = self.project_root
        return [
            base / "core" / "templates" / "skills" / template_name,
            base / "core" / "templates" / template_name,
            base / "templates" / "skills" / template_name,
            base / "templates" / template_name,
        ]

    def _load_template(self, template_name: str) -> np.ndarray:
        for path in self._candidate_paths(template_name):
            if path.exists():
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    return img
        raise FileNotFoundError(f"Template not found: {template_name}")

    def find(self, image: np.ndarray, threshold: float = 0.80):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        result = cv2.matchTemplate(gray, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        return bool(max_val >= threshold), float(max_val), (int(max_loc[0]), int(max_loc[1]))
