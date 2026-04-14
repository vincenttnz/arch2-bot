try:
    from core.vision_skill_select import SkillSelectorVision
except Exception:
    from vision_skill_select import SkillSelectorVision


class SkillDetector:
    """Backward-compatible wrapper around the new skill selector."""

    def __init__(self, project_root=None):
        self.selector = SkillSelectorVision(project_root=project_root)

    def detect(self, frame, **_kwargs):
        return bool(self.selector.is_skill_screen(frame))
