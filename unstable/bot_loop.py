"""
bot_loop.py  —  Archero 2 Bot with Skill Priority & Perceptual Hash Recognition
================================================================================
- Uses perceptual hash database to instantly recognise skill cards.
- Picks the best skill based on a pre‑defined priority list.
- If recognition fails, clicks the middle card (safe fallback).
- Coordinates are calculated dynamically based on screen size.
"""
import time
import cv2
import numpy as np
import threading
import os
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
from PIL import Image
import imagehash

try:
    from ultralytics import YOLO
    _YOLO_OK = True
except ImportError:
    _YOLO_OK = False

# ----------------------------------------------------------------------
# Skill priority list (higher index = better)
# Based on Archero 2 tier lists (Ricochet, Front Arrow, Attack Speed, etc.)
# Edit this dictionary or load from a JSON file.
SKILL_PRIORITY = {
    "ricochet": 100,
    "front_arrow": 95,
    "multi_shot": 95,
    "attack_speed": 90,
    "critical_hit": 85,
    "giant_growth": 80,
    "sword_strike": 75,
    "circle_of_death": 70,
    "bolt": 65,
    "venom": 60,
    "ice_ring": 55,
    "vampiric_circle": 50,
    "boss_slayer": 45,
    "pursuit_strike": 40,
    "wind_blessing": 35,
    "life_conversion": 30,
    "warriors_breath": 25,
    "cloudfooted": 20,
    "stand_strong": 15,
    "restore_hp": 10,
}
# ----------------------------------------------------------------------

class SkillCardRecognizer:
    """Perceptual hash‑based recogniser for skill cards."""
    def __init__(self, db_path="data/skill_card_hashes.json"):
        self.db_path = Path(db_path)
        self.hash_to_skill = {}
        self.load()

    def build_database(self, skills_root="core/templates/skills"):
        skills_root = Path(skills_root)
        if not skills_root.exists():
            print(f"❌ Skill root not found: {skills_root}")
            return
        hash_map = {}
        for skill_folder in skills_root.iterdir():
            if not skill_folder.is_dir():
                continue
            skill_name = skill_folder.name.lower().replace(" ", "_")
            for img_path in skill_folder.glob("*.[jp][pn]g"):
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize((128, 128))
                    phash = str(imagehash.phash(img))
                    if phash not in hash_map:
                        hash_map[phash] = skill_name
                except Exception:
                    pass
        with open(self.db_path, "w") as f:
            json.dump(hash_map, f)
        print(f"✅ Built skill hash database: {len(hash_map)} entries")

    def load(self):
        if self.db_path.exists():
            with open(self.db_path, "r") as f:
                self.hash_to_skill = json.load(f)
        else:
            self.hash_to_skill = {}

    def recognize(self, card_pil_image):
        """Return skill name or None."""
        try:
            card = card_pil_image.resize((128, 128))
            phash = str(imagehash.phash(card))
            return self.hash_to_skill.get(phash)
        except Exception:
            return None


class ArcheroBot:
    def __init__(self, adb, project_root=None, logger=None,
                 log_frames=False, log_every=5, yolo_model=None):
        self.adb = adb
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.log = logger if logger else print
        self.log_frames = log_frames
        self.log_every = log_every
        self.yolo_model = yolo_model
        self.running = False
        self.thread = None

        # Bot state
        self.step = 0
        self.total_reward = 0.0
        self.episode_reward = 0.0
        self.deaths = 0
        self.dodges = 0
        self.last_health = 1.0
        self.last_skill_pick_step = -100
        self.last_enemy_kill_step = -100
        self.last_damage_step = -100

        # Movement parameters
        self.move_interval = 0.25        # 250ms
        self.center_attraction = 0.3
        self.safe_radius = 300

        # Paths for debugging
        self.debug_dir = self.project_root / "data" / "bot_debug"
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.reward_state_path = self.project_root / "data" / "reward_state.json"

        # Skill recognizer
        self.skill_recognizer = SkillCardRecognizer(db_path=str(self.project_root / "data" / "skill_card_hashes.json"))
        self.skill_recognizer.load()

    def get_status(self) -> Dict[str, Any]:
        return {
            "screen": "running" if self.running else "stopped",
            "reward": self.total_reward,
            "total_reward": self.total_reward,
            "step": self.step,
            "deaths": self.deaths,
            "dodges": self.dodges,
        }

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self._save_reward_state()

    def run(self):
        self.running = True
        self.log("[BOT] Neural Engine Active – Skill Priority Enabled")
        self._save_reward_state()

        if self.yolo_model is None and _YOLO_OK:
            self._load_latest_model()

        last_move_time = time.time()
        last_log_time = time.time()
        last_frame_save = time.time()
        last_health_check = time.time()

        while self.running:
            now = time.time()
            frame = self.adb.screencap()
            if frame is None:
                time.sleep(0.05)
                continue

            detections = self._detect_objects(frame)

            # Check for skill screen (highest priority)
            if self._is_skill_screen(frame):
                self._pick_best_skill(frame)
                time.sleep(1.5)
                continue

            if now - last_health_check >= 0.5:
                self._update_health_and_reward(frame)
                last_health_check = now

            if now - last_move_time >= self.move_interval:
                move_vector = self._compute_move_vector(frame, detections)
                self.adb.move_vector(move_vector[0], move_vector[1], frame.shape)
                last_move_time = now
                self.step += 1
                self.total_reward += 0.1
                self.episode_reward += 0.1

            if now - last_log_time >= 1.0:
                self._log_status(detections)
                last_log_time = now
                self._save_reward_state()

            if self.log_frames and (now - last_frame_save) >= self.log_every:
                debug_img = self._draw_debug(frame, detections)
                cv2.imwrite(str(self.debug_dir / f"step_{self.step:06d}.jpg"), debug_img)
                last_frame_save = now

            time.sleep(0.05)

        self.log("[BOT] Stopped.")
        self._save_reward_state()

    # ----------------------------------------------------------------------
    # Model loading
    # ----------------------------------------------------------------------
    def _load_latest_model(self):
        runs_dir = self.project_root / "runs" / "detect"
        if not runs_dir.exists():
            return
        candidates = sorted(runs_dir.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            try:
                self.yolo_model = YOLO(str(candidates[0]))
                self.log(f"[BOT] Loaded model: {candidates[0].name}")
            except Exception as e:
                self.log(f"[BOT] Failed to load model: {e}")

    def _detect_objects(self, frame):
        if self.yolo_model is None:
            return []
        try:
            results = self.yolo_model(frame, conf=0.35, verbose=False)
            detections = []
            for box in results[0].boxes:
                xc, yc, w, h = box.xywh[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.yolo_model.names[cls]
                detections.append({"label": label, "x": xc, "y": yc, "w": w, "h": h, "conf": conf})
            return detections
        except Exception:
            return []

    # ----------------------------------------------------------------------
    # Movement
    # ----------------------------------------------------------------------
    def _compute_move_vector(self, frame, detections):
        h, w = frame.shape[:2]
        px, py = w // 2, h // 2
        fx, fy = 0.0, 0.0

        # Attraction to centre
        fx += (w/2 - px) * self.center_attraction
        fy += (h/2 - py) * self.center_attraction

        for d in detections:
            label = d["label"]
            if label not in ("projectile", "mob", "boss", "aoe_indicator"):
                continue
            dx = px - d["x"]
            dy = py - d["y"]
            dist = np.hypot(dx, dy) + 1e-5
            if dist < self.safe_radius:
                weight = 3.5 if label == "projectile" else 2.0 if label == "mob" else 1.5
                falloff = np.exp(-dist / (self.safe_radius/2))
                repulsion = weight * falloff
                fx += (dx / dist) * repulsion
                fy += (dy / dist) * repulsion

        max_mag = max(abs(fx), abs(fy), 0.01)
        return fx / max_mag, fy / max_mag

    # ----------------------------------------------------------------------
    # Skill screen detection & selection (fallback to middle card)
    # ----------------------------------------------------------------------
    def _is_skill_screen(self, frame):
        try:
            import pytesseract
            h, w = frame.shape[:2]
            crop = frame[int(h*0.1):int(h*0.3), int(w*0.2):int(w*0.8)]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            text = pytesseract.image_to_string(thresh).lower()
            return "level up" in text or "choose a skill" in text
        except:
            return False

    def _pick_best_skill(self, frame):
        """
        Extract card images, recognise skills, pick the highest priority one.
        If no skill is recognised, click the middle card.
        Card regions are calculated dynamically relative to screen size.
        """
        h, w = frame.shape[:2]
        # Each card occupies roughly 26% of width, with gaps.
        # Coordinates: left card from 8% to 34%, middle from 36% to 62%, right from 64% to 90%
        card_width = int(w * 0.26)
        gap = int(w * 0.02)
        start_x = int(w * 0.08)
        card_y_top = int(h * 0.62)
        card_y_bottom = int(h * 0.85)

        card_regions = []
        for i in range(3):
            x1 = start_x + i * (card_width + gap)
            x2 = x1 + card_width
            card_regions.append((x1, card_y_top, x2, card_y_bottom))

        # Fallback absolute coordinates for 1080x1920 (if needed)
        # But we'll use the dynamic ones above.

        skills = []
        for i, (x1, y1, x2, y2) in enumerate(card_regions):
            # Ensure coordinates are within frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            card_crop = frame[y1:y2, x1:x2]
            if card_crop.size == 0:
                continue
            pil_img = Image.fromarray(cv2.cvtColor(card_crop, cv2.COLOR_BGR2RGB))
            skill_name = self.skill_recognizer.recognize(pil_img)
            if skill_name:
                skills.append((i, skill_name))
            else:
                # Fallback: try OCR on the crop
                try:
                    import pytesseract
                    gray = cv2.cvtColor(card_crop, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                    text = pytesseract.image_to_string(thresh).strip().lower()
                    text = re.sub(r'[^a-z_]', '', text)
                    if text:
                        skills.append((i, text))
                except:
                    pass

        if not skills:
            self.log("[BOT] No skill recognised – picking middle card (fallback)")
            best_idx = 1   # middle card
        else:
            # Choose skill with highest priority
            best_idx = max(skills, key=lambda x: SKILL_PRIORITY.get(x[1], 0))[0]

        # Click the centre of the chosen card
        x1, y1, x2, y2 = card_regions[best_idx]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        self.adb.click(cx, cy)
        skill_name = skills[best_idx][1] if skills else "unknown"
        self.log(f"[BOT] Picked skill: {skill_name} (priority)")
        self.total_reward += 10
        self.episode_reward += 10
        self.last_skill_pick_step = self.step

    # ----------------------------------------------------------------------
    # Health & death detection
    # ----------------------------------------------------------------------
    def _update_health_and_reward(self, frame):
        # Simplified health bar detection (green bar in top‑left)
        try:
            h, w = frame.shape[:2]
            roi = frame[int(h*0.02):int(h*0.07), int(w*0.02):int(w*0.17)]
            if roi.size == 0:
                return
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower_green = np.array([40, 50, 50])
            upper_green = np.array([80, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)
            green_pixels = cv2.countNonZero(mask)
            health = green_pixels / roi.shape[1] if roi.shape[1] > 0 else 1.0
            if health < self.last_health - 0.05:
                penalty = (self.last_health - health) * -100
                self.total_reward += penalty
                self.episode_reward += penalty
                self.log(f"[BOT] Health drop – penalty {penalty:.0f}")
            self.last_health = health
        except:
            pass

        # Death screen check
        try:
            import pytesseract
            h, w = frame.shape[:2]
            crop = frame[int(h*0.7):int(h*0.9), int(w*0.3):int(w*0.7)]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            text = pytesseract.image_to_string(thresh).lower()
            if "defeated" in text or "you died" in text:
                self.total_reward -= 100
                self.episode_reward -= 100
                self.deaths += 1
                self.log("[BOT] Death detected – penalty -100")
        except:
            pass

    # ----------------------------------------------------------------------
    # Logging & debug
    # ----------------------------------------------------------------------
    def _log_status(self, detections):
        threats = [d["label"] for d in detections if d["label"] != "player"]
        threat_str = ", ".join(threats[:5]) if threats else "none"
        self.log(f"[BOT] Step {self.step} | Reward {self.total_reward:.1f} | Threats: {threat_str}")

    def _draw_debug(self, frame, detections):
        debug = frame.copy()
        h, w = frame.shape[:2]
        px, py = w//2, h//2
        cv2.circle(debug, (px, py), 10, (0, 255, 0), -1)
        for d in detections:
            x, y, wd, hd = d["x"], d["y"], d["w"], d["h"]
            x1, y1 = int(x - wd/2), int(y - hd/2)
            x2, y2 = int(x + wd/2), int(y + hd/2)
            color = (0,0,255) if d["label"]=="projectile" else (0,255,255) if d["label"]=="mob" else (255,0,0)
            cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
            cv2.putText(debug, d["label"], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(debug, f"Step: {self.step}  Reward: {self.total_reward:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        return debug

    def _save_reward_state(self):
        data = {
            "total_reward": self.total_reward,
            "episode_reward": self.episode_reward,
            "step": self.step,
            "deaths": self.deaths,
            "dodges": self.dodges,
            "timestamp": time.time()
        }
        try:
            with open(self.reward_state_path, "w") as f:
                json.dump(data, f)
        except Exception:
            pass