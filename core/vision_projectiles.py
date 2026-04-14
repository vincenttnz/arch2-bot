from pathlib import Path
from ultralytics import YOLO


class ProjectileDetector:
    def __init__(self, project_root):
        self.root = Path(project_root)
        self.model_path = self.root / "models" / "combat" / "weights" / "best.pt"
        self.model = YOLO(self.model_path) if self.model_path.exists() else None

    def detect(self, frame):
        if self.model is None:
            return []
        results = self.model(frame, verbose=False, conf=0.3)[0]
        return [
            ((b.xyxy[0][0] + b.xyxy[0][2]) / 2, (b.xyxy[0][1] + b.xyxy[0][3]) / 2)
            for b in results.boxes
            if int(b.cls) == 2
        ]
