import cv2
import numpy as np
import pydirectinput
import keyboard
import time
from pathlib import Path

class EvasionEngine:
    def __init__(self, parent):
        self.parent = parent  # reference to UnifiedCommandCenter
        self.is_evading = False
        self.model = None
        self.safe_radius = 400
        self.min_conf = 0.55
        self.panic_key = 'shift'
        self.panic_exit = 'f10'
        self.flow_scale = 0.5
        self.flow_predict_frames = 3

    def start(self, model, safe_radius, min_conf):
        self.model = model
        self.safe_radius = safe_radius
        self.min_conf = min_conf
        self.is_evading = True
        self._worker()

    def stop(self):
        self.is_evading = False
        for k in ['w','a','s','d']:
            pydirectinput.keyUp(k)

    def _worker(self):
        self.parent.log("⚔ EVASION ENGINE STARTED")
        prev_gray = None
        dodges = 0
        manual = 0
        last_log = time.time()

        # Start OpenCV thread
        cv2.startWindowThread()
        hud_win = "Vision_HUD_Overlay"

        while self.is_evading:
            if keyboard.is_pressed(self.panic_exit):
                break

            rect = self.parent.get_game_rect()
            if not rect:
                time.sleep(0.5)
                continue

            # Capture only the game window
            with self.parent.sct as sct:
                img = np.array(sct.grab(rect))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            h, w = frame.shape[:2]
            px, py = w//2, h//2

            # Optical flow
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = None
            if prev_gray is not None and prev_gray.shape == gray.shape:
                small_gray = cv2.resize(gray, (0,0), fx=self.flow_scale, fy=self.flow_scale)
                small_prev = cv2.resize(prev_gray, (0,0), fx=self.flow_scale, fy=self.flow_scale)
                flow = cv2.calcOpticalFlowFarneback(small_prev, small_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            prev_gray = gray

            # Model inference
            results = self.model.predict(frame, conf=self.min_conf, verbose=False)
            hud = frame.copy()
            override = any(keyboard.is_pressed(k) for k in ['w','a','s','d', self.panic_key])
            fx = fy = 0.0

            cv2.circle(hud, (px, py), self.safe_radius, (255,255,255), 1)

            threat_counts = {}
            for box in results[0].boxes:
                c = box.xywh[0]
                hx, hy = float(c[0]), float(c[1])
                hw, hh = float(c[2])/2, float(c[3])/2
                label = self.model.names[int(box.cls[0])]
                threat_counts[label] = threat_counts.get(label, 0) + 1

                # Predict trajectory for projectiles
                if label == 'projectile' and flow is not None:
                    cx_flow = int(hx * self.flow_scale)
                    cy_flow = int(hy * self.flow_scale)
                    if 0 <= cx_flow < flow.shape[1] and 0 <= cy_flow < flow.shape[0]:
                        vx, vy = flow[cy_flow, cx_flow]
                        vx = vx / self.flow_scale * self.flow_predict_frames
                        vy = vy / self.flow_scale * self.flow_predict_frames
                        hx += vx
                        hy += vy

                col = (0,255,0) if label == 'mob' else (0,255,255) if label == 'projectile' else (0,0,255)
                cv2.rectangle(hud, (int(hx-hw), int(hy-hh)), (int(hx+hw), int(hy+hh)), col, 2)

                dx, dy = px - hx, py - hy
                dist = np.hypot(dx, dy) + 1e-5
                if dist < self.safe_radius:
                    sigma = self.safe_radius / 3
                    falloff = np.exp(-(dist**2)/(2*sigma**2))
                    base_wt = 3.5 if label == 'projectile' else 2.0 if label == 'aoe_indicator' else 1.0
                    heading = 1.0
                    if label == 'projectile' and flow is not None:
                        vel = np.array([vx, vy]) if 'vx' in locals() else np.array([0,0])
                        speed = np.linalg.norm(vel)
                        if speed > 0:
                            dir_to_player = np.array([dx, dy]) / dist
                            heading = 1.0 + max(0, np.dot(vel/speed, dir_to_player))
                    weight = base_wt * falloff * heading
                    fx += (dx/dist) * weight
                    fy += (dy/dist) * weight

            thr = 0.25
            if time.time() - last_log > 1.5:
                t_str = ", ".join(f"{k}:{v}" for k,v in threat_counts.items()) if threat_counts else "None"
                if override:
                    self.parent.log(f"[HUD] Manual Override | Threats: {t_str}")
                elif abs(fx) > thr or abs(fy) > thr:
                    self.parent.log(f"[HUD] EVADING ({fx:.1f},{fy:.1f}) | Threats: {t_str}")
                else:
                    self.parent.log(f"[HUD] Safe | Threats: {t_str}")
                last_log = time.time()

            # Execute movement
            if override:
                manual += 1
                for k in ['w','a','s','d']: pydirectinput.keyUp(k)
                cv2.putText(hud, "MANUAL OVERRIDE", (20,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,165,255), 2)
            elif abs(fx) > thr or abs(fy) > thr:
                dodges += 1
                if fx > thr:
                    pydirectinput.keyDown('d'); pydirectinput.keyUp('a')
                elif fx < -thr:
                    pydirectinput.keyDown('a'); pydirectinput.keyUp('d')
                if fy > thr:
                    pydirectinput.keyDown('s'); pydirectinput.keyUp('w')
                elif fy < -thr:
                    pydirectinput.keyDown('w'); pydirectinput.keyUp('s')
                cv2.putText(hud, f"EVADING", (20,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,255), 2)
            else:
                for k in ['w','a','s','d']: pydirectinput.keyUp(k)
                cv2.putText(hud, "IDLE", (20,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,255,0), 2)

            # Show HUD
            cv2.imshow(hud_win, cv2.resize(hud, (1280,720)))
            cv2.waitKey(1)

            # Adaptive sleep
            time.sleep(0.033 if len(results[0].boxes) > 0 else 0.1)

        cv2.destroyWindow(hud_win)
        for k in ['w','a','s','d']: pydirectinput.keyUp(k)
        auto = (dodges - manual) / max(1, dodges) * 100 if dodges else 0
        self.parent.log(f"🏁 Evasion stopped – Dodges: {dodges}, Overrides: {manual}, Autonomy: {auto:.1f}%")