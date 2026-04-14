from __future__ import annotations

from .vision_projectiles import ProjectileDetector


class DodgeAssistant:
    def __init__(self, project_root=None, danger_threshold: float = 0.55) -> None:
        self.projectile_detector = ProjectileDetector(project_root) if project_root is not None else None
        self.danger_threshold = float(danger_threshold)

    def should_dodge(self, frame) -> bool:
        if self.projectile_detector is None:
            return False
        count = len(self.projectile_detector.detect(frame))
        return count > 0 and float(min(1.0, count / 3.0)) >= self.danger_threshold

    def get_dodge_direction(self, frame) -> str:
        return "right" if self.should_dodge(frame) else "none"
