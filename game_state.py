
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from core.vision_skill_select import SkillSelectorVision

class GameStateDetector:
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parent
        self.skill_selector = SkillSelectorVision(project_root=self.project_root)

    def is_pause_screen(self, frame):
        if frame is None or frame.size == 0:
            return False
        if self.skill_selector.detect_context(frame) is not None:
            return False
        h, w = frame.shape[:2]
        bottom = frame[int(h * 0.78):int(h * 0.98), int(w * 0.06):int(w * 0.94)]
        if bottom.size == 0:
            return False
        hsv = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, np.array([35, 40, 80]), np.array([100, 255, 255]))
        red1 = cv2.inRange(hsv, np.array([0, 70, 80]), np.array([12, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([170, 70, 80]), np.array([179, 255, 255]))
        orange = cv2.inRange(hsv, np.array([10, 70, 80]), np.array([30, 255, 255]))
        warm = cv2.bitwise_or(cv2.bitwise_or(red1, red2), orange)
        green_ratio = float(np.count_nonzero(green)) / max(1, green.size)
        warm_ratio = float(np.count_nonzero(warm)) / max(1, warm.size)
        return green_ratio > 0.06 and warm_ratio > 0.06

    def detect(self, frame, params=None):
        if frame is None or frame.size == 0:
            return "UNKNOWN"
        ctx = self.skill_selector.detect_context(frame)
        if ctx in {"DEVIL", "ANGEL", "VALKYRIE", "SKILL"}:
            return ctx
        if self.is_pause_screen(frame):
            return "PAUSE"
        return "COMBAT"
