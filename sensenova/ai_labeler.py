import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import os
import shutil
from collections import Counter

# ==========================================================
# WINDOWS TESSERACT CONFIG
# ==========================================================
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class AILabeler:
    def __init__(self, config):
        self.input_dirs = config['input_dirs']
        self.output_root = config['output_root']
        self.output_map = config['output_map']
        self.template_path = config['template_path']
        self.stats = Counter()
        self.player_crop_dir = os.path.join(self.output_root, "player_crops")
        
        for folder_name in self.output_map.values():
            os.makedirs(os.path.join(self.output_root, folder_name), exist_ok=True)
        os.makedirs(self.player_crop_dir, exist_ok=True)

        self.template = cv2.imread(self.template_path, 0) if os.path.exists(self.template_path) else None

    def detect_and_crop_player(self, img, filename):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv, np.array([45, 150, 150]), np.array([75, 255, 255]))
        contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0: continue
            if 3.0 < (w/h) < 8.0 and 40 < w < 200 and 10 < h < 45:
                roi = hsv[y+h:min(y+h+90, img.shape[0]), max(0, x-30):min(x+w+30, img.shape[1])]
                if roi.size == 0: continue
                blonde = cv2.countNonZero(cv2.inRange(roi, np.array([15, 100, 100]), np.array([35, 255, 255])))
                blue = cv2.countNonZero(cv2.inRange(roi, np.array([90, 100, 100]), np.array([130, 255, 255])))
                if blonde > 15 or blue > 25:
                    cx, cy = x+(w//2), y+(h//2)+50
                    y1, y2, x1, x2 = max(0,cy-120), min(img.shape[0],cy+120), max(0,cx-120), min(img.shape[1],cx+120)
                    cv2.imwrite(os.path.join(self.player_crop_dir, f"player_{filename}"), img[y1:y2, x1:x2])
                    return True
        return False

    def classify_frame(self, image_path):
        img = cv2.imread(image_path)
        if img is None: return "unrecognized", 0.0
        
        filename = os.path.basename(image_path).upper()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # ---------------------------------------------------------
        # 1. COMBAT OVERRIDE (Content-based, ignores filename)
        # ---------------------------------------------------------
        # If we see the player, it is Combat.
        if self.detect_and_crop_player(img, os.path.basename(image_path)):
            return "player_detected", 1.0

        # Check for speed multiplier (x2, x3) or pause button in the top 30%
        top_roi = gray[0:int(h*0.3), :]
        try:
            ui_text = pytesseract.image_to_string(top_roi).lower()
            if any(k in ui_text for k in ["x2", "x3", "x1", "lv", "02:", "03:"]):
                # Double check to ensure we aren't seeing rarity text misread as UI
                if not any(sk in ui_text for sk in ["epic", "rare", "fine"]):
                    return "combat", 1.0
        except: pass

        # ---------------------------------------------------------
        # 2. SKILL CARD DETECTION
        # ---------------------------------------------------------
        # A) Template match for full screen
        if self.template is not None and w >= self.template.shape[1] and h >= self.template.shape[0]:
            res = cv2.matchTemplate(gray, self.template, cv2.TM_CCOEFF_NORMED)
            if cv2.minMaxLoc(res)[1] > 0.8: return "skill_card", 0.9

        # B) OCR for Rarity + Keywords (Catches the vertical crops)
        try:
            full_text = pytesseract.image_to_string(gray).lower()
            skill_triggers = ["epic", "fine", "rare", "get ", "choose a", "pick up", "learned"]
            if any(k in full_text for k in skill_triggers):
                # If it's a very narrow crop, we only accept it if it HAS these keywords
                return "skill_card", 0.8
        except: pass

        # ---------------------------------------------------------
        # 3. MENU DETECTION
        # ---------------------------------------------------------
        try:
            if any(k in full_text for k in ["damage count", "has ended", "defeat", "victory"]):
                return "menu", 1.0
        except: pass

        # ---------------------------------------------------------
        # 4. INTEGRITY CHECK (Toss leftovers that are too narrow/small)
        # ---------------------------------------------------------
        if w < 120 or (h/w) > 5.0:
            return "unrecognized", 0.0

        return "unrecognized", 0.0

    def run(self):
        all_files = []
        for d in self.input_dirs:
            if os.path.exists(d):
                all_files.extend([(d, f) for f in os.listdir(d) if f.lower().endswith(('.jpg', '.png'))])

        total = len(all_files)
        print(f"🚀 Re-scanning {total} files with tuned precision...")
        for i, (dir_path, filename) in enumerate(all_files, 1):
            file_path = os.path.join(dir_path, filename)
            label, _ = self.classify_frame(file_path)
            self.stats[label] += 1
            print(f"📦 [{i}/{total}] {label.upper()} | {filename}", end='\r')
            
            dest_folder = os.path.join(self.output_root, self.output_map.get(label, "trash"))
            dest_path = os.path.join(dest_folder, filename)
            
            # Avoid moving to same location
            if os.path.abspath(file_path) != os.path.abspath(dest_path):
                try: shutil.move(file_path, dest_path)
                except: pass
                
        self.print_summary()

    def print_summary(self):
        print("\n\n" + "="*45)
        print(f"{'CATEGORY':<15} | {'COUNT':<5}")
        print("-" * 45)
        for label, count in self.stats.items():
            print(f"{label.upper():<15} | {count:<5}")
        print("="*45)

if __name__ == "__main__":
    BASE = r"C:\Users\Vince\Desktop\arch2"
    CONFIG = {
        'input_dirs': [
            os.path.join(BASE, "data", "raw_frames"), 
            os.path.join(BASE, "data", "skill_cards"),
            os.path.join(BASE, "data", "labeled_data", "skills"),
            os.path.join(BASE, "data", "labeled_data", "menus"),
            os.path.join(BASE, "data", "labeled_data", "combat"),
            os.path.join(BASE, "data", "labeled_data", "trash")
        ],
        'output_root': os.path.join(BASE, "data", "labeled_data"),
        'template_path': os.path.join(BASE, "SenseNova-SI", "refresh_template.png"),
        'output_map': {
            'skill_card': 'skills',
            'combat': 'combat',
            'player_detected': 'player_frames',
            'menu': 'menus',
            'unrecognized': 'trash'
        }
    }
    AILabeler(CONFIG).run()