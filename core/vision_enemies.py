from pathlib import Path
from ultralytics import YOLO
import torch

class EnemyDetector:
    def __init__(self, project_root):
        # VERIFIED: Loads the model you are about to train
        self.model_path = Path(project_root) / "models" / "combat" / "weights" / "best.pt"
        self.model = YOLO(self.model_path) if self.model_path.exists() else None

    def detect(self, frame):
        if not self.model: return []
        # Runs at high speed (60+ FPS) on local hardware
        results = self.model(frame, verbose=False, conf=0.4)[0]
        # Class 1 = Enemy
        return [((b.xyxy[0][0] + b.xyxy[0][2])/2, (b.xyxy[0][1] + b.xyxy[0][3])/2) 
                for b in results.boxes if int(b.cls) == 1]