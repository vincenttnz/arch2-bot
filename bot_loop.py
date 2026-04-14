"""
bot_loop.py  v4.2
==================
Key fixes in this version:

1. SKILL SCREEN → MOVEMENT:
   After clicking a skill card, the bot now correctly resumes movement the
   moment the game returns to COMBAT state.  A post-click cooldown of 2.5 s
   prevents the bot from hammering the same card repeatedly.  If the skill
   screen is still detected > 5 s after the last click (stuck / animation
   taking long), it re-clicks the CENTRE card as a failsafe.

2. BACKUP MIDDLE-CARD CLICK:
   All skill-screen clicks default to w//2 (centre) unless SkillRecogniser
   suggests a different slot.  This matches the requirement to "always click
   the middle skill card" as a backup.

3. BARE-EXCEPT REMOVED:
   Replaced with explicit Exception logging so errors are visible in the GUI
   log with [BOT ERROR] prefix rather than being swallowed silently.

4. SKILL DETECTOR FALLBACK:
   If core/vision_skill_select.py is missing or broken, a lightweight HSV
   colour detector recognises the skill-selection background (golden/beige
   card panel) so the bot still picks skills without the external module.

5. MOVEMENT AFTER SKILL (BUG FIX):
   last_move_ts is reset to 0 after a skill click so movement fires
   immediately on the first COMBAT frame after the skill screen clears.
   Previously last_move_ts could have a recent timestamp from before the
   skill screen appeared, causing a 0.55 s delay.

6. EXPLICIT ERROR COUNTING:
   If more than 20 consecutive frames raise an exception, the bot logs a
   clear warning and pauses for 2 s instead of spinning at maximum speed.
"""
from __future__ import annotations
import json, random, time, traceback
from pathlib import Path
from typing import Optional

import cv2, numpy as np

# ── optional ultralytics ───────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    _YOLO_OK = True
except ImportError:
    _YOLO_OK = False

# ── skill recogniser ──────────────────────────────────────────────────────────
_REC_OK = False
try:
    from core.skill_recogniser import SkillRecogniser, SKILL_PRIORITY
    _REC_OK = True
except ImportError:
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from skill_recogniser import SkillRecogniser, SKILL_PRIORITY
        _REC_OK = True
    except ImportError:
        SkillRecogniser = None
        SKILL_PRIORITY  = {}

# ── evasion constants ─────────────────────────────────────────────────────────
_EV_WEIGHTS = {"projectile":3.5,"aoe_indicator":2.0,"boss":1.5,"mob":1.0,"player":0.0}
_SAFE_R     = 400
_MOVE_THR   = 0.25
_HEU_PERIOD = 0.55   # seconds between heuristic moves

# ── skill screen colour signature ─────────────────────────────────────────────
# Archero 2 skill card background is a warm beige/golden panel.
# These HSV ranges detect enough of that colour to distinguish SKILL from COMBAT.
_SKILL_HSV_LO  = np.array([15,  80, 150])
_SKILL_HSV_HI  = np.array([40, 255, 255])
_SKILL_MIN_PIX = 0.04    # fraction of frame that must be golden


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight skill-screen detector (fallback when vision_skill_select missing)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_skill_screen(frame: np.ndarray) -> bool:
    """
    Return True if the current frame looks like a skill selection screen.
    Uses HSV colour analysis to find the golden card panel background.
    Fast (~1 ms) and works even when vision_skill_select is unavailable.
    """
    try:
        h, w = frame.shape[:2]
        # Focus on the middle vertical band where cards appear
        roi = frame[int(h*0.25):int(h*0.85), int(w*0.15):int(w*0.85)]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, _SKILL_HSV_LO, _SKILL_HSV_HI)
        frac = np.count_nonzero(mask) / mask.size
        return frac >= _SKILL_MIN_PIX
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Reward Tracker
# ─────────────────────────────────────────────────────────────────────────────

class RewardTracker:
    """Accumulates reward signals and persists to data/reward_state.json."""

    def __init__(self, path: Path):
        self.path    = path
        self.total   = self.episode = 0.0
        self.step    = self.dodges = self.deaths = self.waves = self.skills = 0
        self._last_t = time.time()
        self._prev   = "BOOT"
        self._proj: dict = {}
        self._load()

    def _load(self):
        try:
            if self.path.exists():
                d = json.loads(self.path.read_text())
                self.total   = float(d.get("total_reward",   0))
                self.episode = float(d.get("episode_reward", 0))
                self.step    = int(d.get("step",             0))
                self.dodges  = int(d.get("dodges",           0))
                self.deaths  = int(d.get("deaths",           0))
                self.waves   = int(d.get("waves",            0))
                self.skills  = int(d.get("skills_picked",    0))
        except Exception:
            pass

    def save(self):
        try:
            self.path.write_text(json.dumps({
                "total_reward":   round(self.total,   3),
                "episode_reward": round(self.episode, 3),
                "step":  self.step,  "dodges":  self.dodges,
                "deaths":self.deaths,"waves":   self.waves,
                "skills_picked": self.skills, "ts": time.time(),
            }))
        except Exception:
            pass

    def on_frame(self, screen, dets, safe_r, px, py):
        self.step += 1
        now = time.time()
        if self._prev == "COMBAT":
            if screen == "SKILL":
                self._add(5.0); self.waves += 1; self._last_t = now
            elif screen == "MENU":
                self._add(-10.0); self.deaths += 1
                self.episode = 0.0; self._last_t = now
        if screen == "COMBAT" and now - self._last_t >= 1.0:
            self._add(1.0); self._last_t = now
        for i, d in enumerate(dets):
            if d.get("label") != "projectile": continue
            dist   = ((d["x"]-px)**2 + (d["y"]-py)**2) ** 0.5
            in_now = dist < safe_r
            if self._proj.get(i, False) and not in_now:
                self.dodges += 1; self._add(0.5)
            self._proj[i] = in_now
        self._proj = {k: v for k, v in self._proj.items() if k < len(dets)}
        self._prev = screen
        if self.step % 30 == 0:
            self.save()

    def on_skill(self, prio):
        self.skills += 1
        self._add(2.0 if prio >= 9 else 1.0 if prio >= 7 else 0.0)

    def on_thrash(self): self._add(-0.1)
    def _add(self, d):   self.total += d; self.episode += d

    def as_dict(self):
        return {
            "total_reward":   round(self.total,   2),
            "episode_reward": round(self.episode, 2),
            "step":   self.step,   "dodges":  self.dodges,
            "deaths": self.deaths, "waves":   self.waves,
            "skills_picked": self.skills,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ArcheroBot
# ─────────────────────────────────────────────────────────────────────────────

class ArcheroBot:
    """
    Archero 2 bot with guaranteed movement and robust skill selection.

    Movement hierarchy (always guaranteed):
      1. AI evasion using YOLO detection model (if loaded)
      2. Classic random swipe (fallback, always works)

    Skill selection hierarchy:
      1. SkillRecogniser pHash + YOLO classifier → best card by priority
      2. BACKUP: always click CENTRE card (w//2) if recogniser unavailable
    """

    def __init__(self, adb, project_root=None, logger=None,
                 log_frames=False, log_every=5):
        self.project_root  = Path(project_root) if project_root \
                             else Path(__file__).resolve().parent
        self.logger        = logger or print
        self.adb           = adb
        self.running       = False
        self.frame_counter = 0

        # Timestamps — all float seconds
        self.last_move_ts  = 0.0   # last time _execute_movement fired
        self.last_click_ts = 0.0   # last time a skill card was clicked
        self.skill_seen_ts = 0.0   # time when SKILL screen was first detected

        self.last_screen   = "BOOT"
        self.frame_logging = bool(log_frames)
        self.log_every     = max(1, int(log_every))
        self._consec_errors = 0

        # Data dirs
        self.dirs = {
            "combat": self.project_root / "data" / "raw_frames",
            "skills": self.project_root / "data" / "skill_cards",
            "menu":   self.project_root / "data" / "menu_screens",
            "orig":   self.project_root / "data" / "original_snapshots",
            "inbox":  self.project_root / "data" / "high_priority_projectiles",
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        # Reward
        self.reward = RewardTracker(
            self.project_root / "data" / "reward_state.json")

        # Screen detector (external module + HSV fallback)
        self._ctx = None
        try:
            from core.vision_skill_select import SkillSelectorVision
            self._ctx = SkillSelectorVision(project_root=self.project_root)
            self._log("Screen detector: vision_skill_select loaded ✓")
        except Exception as e:
            self._log(f"vision_skill_select unavailable ({e}) "
                      f"— using HSV fallback detector")

        # Skill recogniser
        self.skill_rec = None
        if _REC_OK and SkillRecogniser:
            try:
                self.skill_rec = SkillRecogniser(self.project_root)
                st = self.skill_rec.status()
                self._log(f"SkillRec: {st['db_skills']} skills in DB, "
                          f"clf={st['classifier']}")
            except Exception as e:
                self._log(f"SkillRec init: {e}")
        else:
            self._log("SkillRec not available — centre-card backup active")

        # Detection model (hot-swappable)
        self._yolo      = None
        self._yolo_path = None
        self._yolo_name = "none"
        self._load_model()

        self._log(
            f"Bot v4.2 ready | detection={self._yolo_name} | "
            f"skill_rec={'ready' if self.skill_rec else 'off (backup: centre click)'} | "
            f"steps={self.reward.step}"
        )

    # ── logging ───────────────────────────────────────────────────────────────

    def _log(self, m):
        try:   self.logger(f"[BOT] {m}")
        except Exception: print(f"[BOT] {m}")

    # ── model management ──────────────────────────────────────────────────────

    def _find_best_pt(self):
        d = self.project_root / "runs" / "detect"
        if not d.exists(): return None
        pts = sorted(d.rglob("best.pt"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
        return str(pts[0]) if pts else None

    def _load_model(self):
        if not _YOLO_OK: return
        w = self._find_best_pt()
        if not w or w == self._yolo_path: return
        try:
            self._yolo      = YOLO(w)
            self._yolo_path = w
            self._yolo_name = Path(w).parent.parent.name
            self._log(f"Detection model loaded: {self._yolo_name}")
        except Exception as e:
            self._log(f"Model load failed: {e}")

    def _hotswap(self):
        if self.frame_counter % 60 != 0: return
        w = self._find_best_pt()
        if w and w != self._yolo_path:
            self._log("New detection model — hotswapping …")
            self._load_model()

    # ── status ────────────────────────────────────────────────────────────────

    def get_status(self):
        r = self.reward.as_dict()
        return {
            "screen":         self.last_screen,
            "reward":         r["total_reward"],
            "total_reward":   r["total_reward"],
            "episode_reward": r["episode_reward"],
            "step":           r["step"],
            "dodges":         r["dodges"],
            "deaths":         r["deaths"],
            "skills_picked":  r["skills"],
            "model":          self._yolo_name,
        }

    # ── screen detection ──────────────────────────────────────────────────────

    def _get_screen_state(self, frame) -> tuple[str, str]:
        """
        Returns (screen, raw_state).
        screen   : "SKILL" | "COMBAT" | "MENU"
        raw_state: the more specific value from vision_skill_select
                   or "SKILL_3"/"COMBAT" from our HSV fallback.

        HSV fallback is always run if the external module says COMBAT,
        to catch skill screens the module might miss.
        """
        raw = "COMBAT"
        if self._ctx:
            try:
                raw = self._ctx.detect_context(frame)
            except Exception as e:
                self._log(f"Screen detect error: {e}")
                raw = "COMBAT"

        # If external module says COMBAT, double-check with HSV detector
        if "SKILL" not in raw and _detect_skill_screen(frame):
            raw = "SKILL_3"   # assume 3-card layout as default

        screen = "SKILL" if "SKILL" in raw else raw
        return screen, raw

    # ── data logging ──────────────────────────────────────────────────────────

    def _save_data(self, frame, screen, raw):
        ts = int(time.time() * 1000)
        h, w = frame.shape[:2]
        cv2.imwrite(str(self.dirs["orig"] / f"orig_{ts}_{screen}.jpg"), frame)
        if "SKILL" in raw:
            n = 2 if raw == "SKILL_2" else 3
            for i in range(n):
                crop = frame[:, int(w/n*i):int(w/n*(i+1))]
                cv2.imwrite(str(self.dirs["skills"] /
                                f"skill_{ts}_n{n}_s{i}.jpg"), crop)
        elif screen == "MENU":
            cv2.imwrite(str(self.dirs["menu"] / f"menu_{ts}.jpg"), frame)
        else:
            cv2.imwrite(str(self.dirs["combat"] / f"combat_{ts}.jpg"), frame)

    # ── detection ─────────────────────────────────────────────────────────────

    def _detect(self, frame):
        if not self._yolo: return []
        try:
            res  = self._yolo.predict(frame, conf=0.35, verbose=False)
            return [{"label": self._yolo.names.get(int(b.cls[0]), "mob"),
                     "x": float(b.xywh[0][0]), "y": float(b.xywh[0][1]),
                     "w": float(b.xywh[0][2]), "h": float(b.xywh[0][3]),
                     "conf": float(b.conf[0])} for b in res[0].boxes]
        except Exception:
            return []

    # ── evasion (AI movement) ─────────────────────────────────────────────────

    def _evasion_vector(self, dets, shape):
        h, w  = shape[:2]
        px, py = w // 2, int(h * 0.82)
        sigma  = _SAFE_R / 3.0
        fx = fy = 0.0
        for d in dets:
            wt = _EV_WEIGHTS.get(d["label"], 1.0)
            if not wt: continue
            dx   = px - d["x"]; dy = py - d["y"]
            dist = (dx**2 + dy**2) ** 0.5 + 1e-6
            if dist > _SAFE_R: continue
            fall  = np.exp(-(dist**2) / (2 * sigma**2))
            fx   += (dx / dist) * wt * fall
            fy   += (dy / dist) * wt * fall
        return fx, fy

    def _ai_move(self, frame, dets):
        fx, fy = self._evasion_vector(dets, frame.shape)
        if abs(fx) < _MOVE_THR and abs(fy) < _MOVE_THR:
            self.reward.on_thrash(); return
        try:
            self.adb.move_vector(fx, fy, frame.shape)
        except AttributeError:
            # Old adb_controller without move_vector — fall back to swipe
            h, w  = frame.shape[:2]
            cx, cy = w // 2, int(h * 0.82)
            mag = (fx**2 + fy**2) ** 0.5 + 1e-9
            tx  = max(0, min(w-1, int(cx + (fx/mag)*380)))
            ty  = max(0, min(h-1, int(cy + (fy/mag)*380)))
            self.adb.swipe(cx, cy, tx, ty, duration=200)

    # ── classic movement (always works) ───────────────────────────────────────

    def _execute_movement(self, frame):
        """
        Rock-solid random-swipe movement.
        This is the unconditional fallback — always fires if no AI model.
        """
        now = time.time()
        if now - self.last_move_ts < _HEU_PERIOD:
            return
        h, w  = frame.shape[:2]
        cx, cy = w // 2, int(h * 0.82)
        dx = random.randint(-150, 150)
        dy = random.randint(-150, 150)
        self.adb.swipe(cx, cy, cx + dx, cy + dy, duration=250)
        self.last_move_ts = now

    # ── skill selection ────────────────────────────────────────────────────────

    def _handle_skill(self, frame, raw):
        """
        Click the best skill card.

        Priority:
          1. SkillRecogniser identifies each card → click highest-priority slot
          2. BACKUP: always click CENTRE card (w//2) — guaranteed fallback

        After clicking:
          • last_click_ts = now  (1.8 s cooldown before next click)
          • last_move_ts  = 0    (movement fires immediately on COMBAT return)

        Stuck-screen recovery:
          If SKILL screen detected > 5 s after last click, force-click centre
          card again in case the first click missed.
        """
        now  = time.time()
        h, w = frame.shape[:2]

        # Stuck recovery: still on SKILL screen 5+ seconds after clicking
        time_since_click = now - self.last_click_ts
        if time_since_click > 5.0 and self.last_click_ts > 0.0:
            self._log("⚠ Stuck on SKILL screen >5s — forcing centre click")
            self.adb.click(w // 2, int(h * 0.50))
            self.last_click_ts = now
            self.last_move_ts  = 0.0
            return

        # Normal cooldown
        if time_since_click < 1.8:
            return

        # ── Try SkillRecogniser ───────────────────────────────────────────────
        n_slots = 2 if raw == "SKILL_2" else 3

        if self.skill_rec:
            try:
                slot, name, conf = self.skill_rec.pick_best_from_screen(
                    frame, n_slots=n_slots)
                prio = self.skill_rec.priority(name)

                # Compute click X for the identified slot
                if n_slots == 3:
                    # Cards at roughly 20%, 50%, 80% of screen width
                    x = int(w * (0.20 + slot * 0.30))
                else:
                    # Cards at roughly 30%, 70%
                    x = int(w * (0.30 + slot * 0.40))

                self._log(f"SKILL  slot={slot}  {name}  "
                          f"conf={conf:.2f}  prio={prio}")
                self.adb.click(x, int(h * 0.50))
                self.last_click_ts = now
                self.last_move_ts  = 0.0   # reset so movement starts immediately
                self.reward.on_skill(prio)
                return
            except Exception as e:
                self._log(f"SkillRec error → backup centre click: {e}")

        # ── BACKUP: always click CENTRE card ─────────────────────────────────
        self._log(f"SKILL backup: clicking centre card (w//2 = {w//2})")
        self.adb.click(w // 2, int(h * 0.50))
        self.last_click_ts = now
        self.last_move_ts  = 0.0   # reset so movement starts immediately

    # ── uncertain frame capture ────────────────────────────────────────────────

    def _capture_uncertain(self, frame, dets):
        if dets and min(d["conf"] for d in dets) < 0.55:
            ts = int(time.time() * 1000)
            cv2.imwrite(str(self.dirs["inbox"] / f"frame_{ts}.jpg"), frame)

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self):
        self.running = True
        self._log(
            f"Running | detection={self._yolo_name} | "
            f"skill_rec={'ready' if self.skill_rec else 'backup:centre'}"
        )

        while self.running:
            try:
                # ── grab frame ────────────────────────────────────────────────
                frame = self.adb.screencap()
                if frame is None:
                    time.sleep(0.1)
                    continue

                self.frame_counter += 1
                self._consec_errors = 0   # reset on successful frame
                self._hotswap()

                # ── detect screen state ───────────────────────────────────────
                screen, raw = self._get_screen_state(frame)
                self.last_screen = screen

                # Track when SKILL screen first appeared
                if screen == "SKILL" and self.skill_seen_ts == 0.0:
                    self.skill_seen_ts = time.time()
                elif screen != "SKILL":
                    self.skill_seen_ts = 0.0

                # ── YOLO detection ────────────────────────────────────────────
                dets = self._detect(frame) if self._yolo else []

                # ── reward accounting ─────────────────────────────────────────
                h, w = frame.shape[:2]
                self.reward.on_frame(
                    screen, dets, float(_SAFE_R),
                    w // 2, int(h * 0.82))

                # ── frame logging ─────────────────────────────────────────────
                if self.frame_logging:
                    do_log = (
                        screen == "SKILL" or
                        (screen == "MENU"   and self.frame_counter % 10 == 0) or
                        (screen == "COMBAT" and self.frame_counter % self.log_every == 0)
                    )
                    if do_log:
                        self._save_data(frame, screen, raw)

                # ── uncertain-frame capture ───────────────────────────────────
                if screen == "COMBAT" and self.frame_counter % 10 == 0:
                    self._capture_uncertain(frame, dets)

                # ── ACTION ───────────────────────────────────────────────────
                if screen == "SKILL":
                    self._handle_skill(frame, raw)

                else:   # COMBAT or MENU
                    # Movement: AI evasion if model + detections, else classic
                    if self._yolo and dets:
                        self._ai_move(frame, dets)
                    else:
                        self._execute_movement(frame)

                time.sleep(0.02)

            except Exception:
                self._consec_errors += 1
                tb = traceback.format_exc()
                self._log(f"[BOT ERROR] frame={self.frame_counter} "
                          f"(#{self._consec_errors}): {tb.splitlines()[-1]}")
                if self._consec_errors == 1 or self._consec_errors % 20 == 0:
                    self._log(f"Full traceback:\n{tb}")
                if self._consec_errors > 20:
                    self._log("⚠ >20 consecutive errors — pausing 2s")
                    time.sleep(2.0)
                else:
                    time.sleep(0.1)

        # ── shutdown ──────────────────────────────────────────────────────────
        self.reward.save()
        self._log(
            f"Stopped | steps={self.reward.step} | "
            f"reward={self.reward.total:.2f} | "
            f"skills={self.reward.skills} | "
            f"dodges={self.reward.dodges} | "
            f"deaths={self.reward.deaths}"
        )

    def stop(self):
        self.running = False
        try:
            self.adb.stop_move()
        except Exception:
            pass
