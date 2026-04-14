import cv2
import numpy as np
from pathlib import Path

class SkillSelectorVision:
    def __init__(self, project_root):
        self.project_root = Path(project_root)

    def detect_context(self, frame):
        """Clean detection for Menu, Skills (2/3), or Combat."""
        if frame is None: return "COMBAT"
        h, w = frame.shape[:2]

        # 1. MENU CHECK
        nav = frame[int(h*0.88):int(h*0.96), int(w*0.2):int(w*0.8)]
        if np.mean(nav) > 165: return "MENU"

        # 2. SKILL BANNER CHECK
        banner = frame[int(h*0.18):int(h*0.25), int(w*0.2):int(w*0.8)]
        mask = cv2.inRange(cv2.cvtColor(banner, cv2.COLOR_BGR2HSV), 
                           np.array([18, 150, 150]), np.array([35, 255, 255]))
        
        if cv2.countNonZero(mask) > 1000:
            # 2-Card vs 3-Card Logic: 
            # We sample slightly higher (h*0.4) where the solid white icon box of a middle card lives.
            # This avoids the character standing in the background dead-center.
            mid_sample = frame[int(h*0.38):int(h*0.42), int(w*0.48):int(w*0.52)]
            
            # A solid white card box averages > 150. Backgrounds (even with characters) average lower.
            return "SKILL_2" if np.mean(mid_sample) < 120 else "SKILL_3"
        
        return "COMBAT"