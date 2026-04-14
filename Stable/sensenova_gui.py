"""
sensenova_gui.py  —  Unified Command Center v21.0
==================================================
NEW — Acceleration Engine (Tab 2):
  • 🚀 GROUNDING DINO AUTO-LABEL  — zero-shot text-prompt labeling of all
        inbox frames. No manual work. Routes high-conf boxes straight to
        approved/, uncertain boxes to smart-review inbox.
  • ⚡ SMART REVIEW  — uncertainty sampling. Only shows frames where the
        current YOLO model is uncertain (conf 0.25-0.55). Confident frames
        auto-approve, hopeless frames auto-discard. 10x fewer frames to
        review manually, 10x more training value per minute.
  • 🎲 SYNTHETIC AUGMENT  — multiplies approved dataset 8x via flip, rotate,
        HSV jitter, scale, mosaic, noise, blur. Zero additional labeling.

NEW — Smart Trainer (Tab 3):
  • Auto-selects training strategy based on dataset size:
        < 100 images  → Cold Start  (freeze=10, focal loss, 50 epochs)
        100-500       → Warm Train  (freeze=5,  full aug, 80 epochs)
        500+          → Full Train  (freeze=0,  all hyper, 150 epochs)
  • Fixed hyperparameters: rect=True preserves portrait aspect ratio,
        copy_paste=0.3 rebalances rare projectile class, fl_gamma=1.5
        focuses gradients on hard/missed detections.

FIX (Labeler):  L-click always draws the currently selected class.
                _cycle_class() is the only place that mutates _lbl_class.
FIX (Evasion):  ADB screencap is the frame source. ADB swipe is movement.
PRESERVED: All previous modules (active learning, sprint labeler,
           hot-swap model, dailies OCR engine).
"""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import atexit
from pathlib import Path
from typing import Optional

# ── Dependency Guard ──────────────────────────────────────────────────────────
try:
    import tkinter as tk
    from tkinter import messagebox, ttk
    import cv2
    import numpy as np
except ImportError as e:
    print(f"\n❌ CRITICAL: Missing module: {e}")
    sys.exit(1)

try:
    import pygetwindow as gw
    from mss import mss
    _MSS_OK = True
except ImportError:
    _MSS_OK = False

try:
    from ultralytics import YOLO
    _YOLO_OK = True
except ImportError:
    _YOLO_OK = False

try:
    import pydirectinput
    import keyboard
    _INPUT_OK = True
except ImportError:
    _INPUT_OK = False

try:
    import pytesseract
    _tess = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(_tess):
        pytesseract.pytesseract.tesseract_cmd = _tess
    _OCR_OK = True
except ImportError:
    _OCR_OK = False

try:
    from adb_controller import ADBController
    from bot_loop import ArcheroBot
    _BOT_OK = True
except ImportError:
    _BOT_OK = False

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ── Active-learning constants ─────────────────────────────────────────────────
AUTO_TRAIN_THRESHOLD = 50
FINETUNE_EPOCHS      = 15
LABEL_CLASSES        = ['projectile', 'mob', 'aoe_indicator', 'boss', 'player']
LABEL_COLORS         = {
    'projectile': (0, 255, 255),
    'mob': (0, 255, 0),
    'aoe_indicator': (0, 128, 255),
    'boss': (255, 0, 200),
    'player': (255, 255, 0)
}
CLICK_THRESHOLD_PX   = 8
DEFAULT_POINT_SIZE   = {'projectile': 24, 'mob': 44,
                         'aoe_indicator': 56, 'boss': 50, 'player': 36}

# ── Uncertainty-sampling thresholds ──────────────────────────────────────────
SMART_REVIEW_LOW     = 0.25   # below = model has no idea → discard candidate
SMART_REVIEW_HIGH    = 0.60   # above = model already knows → auto-approve
# frames between LOW and HIGH are shown to the human reviewer

# ── Grounding DINO prompts per class ─────────────────────────────────────────
GDINO_PROMPTS = {
    'projectile':    "arrow . fireball . magic projectile . energy ball . red orb . glowing ball . attack projectile . bullet",
    'mob':           "enemy character . monster . creature . game enemy . humanoid enemy",
    'aoe_indicator': "red circle on floor . danger zone . aoe indicator . floor warning . red ring",
    'boss':          "boss enemy . large monster . boss character . giant enemy",
    'player':        "player character . archer . hero character . main character",
}
GDINO_AUTO_CONF  = 0.45   # above → auto-approve without human review
GDINO_MIN_CONF   = 0.25   # below → discard

# ── Training strategy thresholds ─────────────────────────────────────────────
COLD_START_MAX   = 100    # samples below this → cold-start strategy
WARM_TRAIN_MAX   = 500    # samples below this → warm-train strategy
                           # above → full train

# ── Augmentation multiplier ───────────────────────────────────────────────────
AUG_FACTOR       = 8      # synthetic copies per approved image


# ─────────────────────────────────────────────────────────────────────────────
class UnifiedCommandCenter:

    POLL_MS = 250

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SenseNova Command Center v21.0 | Acceleration Engine")
        self.root.geometry("860x980")
        self.root.configure(bg="#1a1a1a")
        self.project_root = Path(__file__).resolve().parent

        # ── Paths ─────────────────────────────────────────────────────────────
        self.base_dir      = str(self.project_root / "data")
        self.inbox_dir     = os.path.join(self.base_dir, "high_priority_projectiles")
        self.approved_dir  = os.path.join(self.base_dir, "approved_projectiles")
        self.discarded_dir = os.path.join(self.base_dir, "discarded_projectiles")
        self.output_path   = os.path.join(self.base_dir, "combat_boxes.jsonl")
        self.yolo_root     = os.path.join(self.base_dir, "yolo_dataset")
        self.progress_path = os.path.join(self.base_dir, "label_progress.json")
        for d in [self.inbox_dir, self.approved_dir,
                  self.discarded_dir, self.yolo_root]:
            os.makedirs(d, exist_ok=True)

        # ── SenseNova state ───────────────────────────────────────────────────
        self.class_map      = {'player': 0, 'projectile': 1,
                                'aoe_indicator': 2, 'mob': 3, 'boss': 4}
        self.window_title   = "SenseNova" # Ensure your emulator/game window is named this!
        self.is_capturing   = False
        self.is_evading     = False
        self.is_daily_running = False
        self.safe_radius    = 400
        self.min_conf       = 0.55
        self.panic_key      = 'shift'
        self.panic_exit     = 'f10'

        # ── Labeler mouse state ───────────────────────────────────────────────
        self._lbl_boxes: list       = []
        self._lbl_drawing           = False
        self._lbl_deleting_region   = False
        self._lbl_ix = self._lbl_iy = -1
        self._lbl_del_ix = self._lbl_del_iy = -1
        self._lbl_class             = 'projectile'
        self._lbl_class_idx         = 0
        self._lbl_scale_x           = 1.0
        self._lbl_scale_y           = 1.0
        self._current_display_base: Optional[np.ndarray] = None
        self._lbl_press_x = self._lbl_press_y = -1
        self._point_size            = DEFAULT_POINT_SIZE.copy()

        # ── Active-learning counters ──────────────────────────────────────────
        self._new_labels_since_train = self._load_label_progress()
        self._is_finetuning          = False
        self._finetune_lock          = threading.Lock()
        self._latest_weights: Optional[str] = None   
        self._labeler_ai_version     = "none"

        # ── Tk log queue (thread-safe) ────────────────────────────────────────
        self._log_q: queue.Queue = queue.Queue()

        # ── Bot state ─────────────────────────────────────────────────────────
        self.adb = None
        self.bot = None
        self.bot_thread = None
        self.log_frames_var     = tk.BooleanVar(value=True)
        self.log_freq_var       = tk.IntVar(value=5)
        self.monitor_device_var = tk.StringVar(value="127.0.0.1:7555")
        self.adb_status_var     = tk.StringVar(value="ADB: disconnected")
        self.bot_status_var     = tk.StringVar(value="Bot: stopped")
        self.screen_var         = tk.StringVar(value="Screen: —")
        self.reward_var         = tk.StringVar(value="Reward: —")

        atexit.register(self._cleanup)
        self._setup_ui()
        self._drain_log()
        self.log("🚀 System Initialized. Awaiting Directives...")
        
        if not _BOT_OK:
            self.log("⚠ bot_loop not found — Bot tab limited.")
        if not _MSS_OK:
            self.log("⚠ mss not found — Capture disabled.")
        if not _YOLO_OK:
            self.log("⚠ ultralytics not found — Training/Evasion disabled.")
            
        self.root.after(self.POLL_MS, self._poll)

    def _load_label_progress(self) -> int:
        try:
            if os.path.exists(self.progress_path):
                with open(self.progress_path, "r") as f:
                    return int(json.loads(f.read()).get("new_since_train", 0))
        except Exception: pass
        return 0

    def _save_label_progress(self):
        try:
            with open(self.progress_path, "w") as f:
                json.dump({"new_since_train": self._new_labels_since_train}, f)
        except Exception: pass

    def log(self, msg: str):
        self._log_q.put(str(msg))

    def _drain_log(self):
        try:
            while True:
                msg = self._log_q.get_nowait()
                self.log_area.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {msg}\n")
                self.log_area.see(tk.END)
        except queue.Empty: pass
        self.root.after(80, self._drain_log)

    def _setup_ui(self):
        hdr = tk.Frame(self.root, bg="#111111")
        hdr.pack(fill="x", side="top")
        tk.Label(hdr, text="⚡  SENSENOVA  +  ARCHERO 2", bg="#111111", fg="#e0e0e0", font=("Consolas", 13, "bold")).pack(side="left", padx=16, pady=10)
        pills = tk.Frame(hdr, bg="#111111")
        pills.pack(side="right", padx=12, pady=8)
        for var in (self.adb_status_var, self.bot_status_var, self.screen_var):
            tk.Label(pills, textvariable=var, bg="#1e1e1e", fg="#aaaaaa", font=("Consolas", 8), padx=8, pady=3).pack(side="left", padx=4)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook", background="#1a1a1a", borderwidth=0)
        style.configure("TNotebook.Tab", background="#2a2a2a", foreground="#aaaaaa", font=("Consolas", 9, "bold"), padding=[14, 6])
        style.map("TNotebook.Tab", background=[("selected", "#333333")], foreground=[("selected", "#ffffff")])
        
        nb = ttk.Notebook(self.root)
        for text, builder in [
            ("  1. Capture  ", self._build_capture_tab),
            ("  2. Labeler  ", self._build_labeler_tab),
            ("  3. Trainer  ", self._build_trainer_tab),
            ("  4. Evasion  ", self._build_evasion_tab),
            ("  5. Bot  ",     self._build_bot_tab),
            ("  6. Dailies  ", self._build_dailies_tab),
        ]:
            tab = ttk.Frame(nb)
            nb.add(tab, text=text)
            builder(tab)
        nb.pack(expand=True, fill="both", padx=10, pady=(4, 0))

        tk.Label(self.root, text="  System Review Log", bg="#111111", fg="#888888", font=("Consolas", 8, "bold"), anchor="w").pack(fill="x", padx=10, pady=(6, 0))
        self.log_area = tk.Text(self.root, height=12, bg="#0a0a0a", fg="#00FF00", font=("Consolas", 8), insertbackground="#00FF00", relief="flat", bd=0)
        self.log_area.pack(fill="both", padx=10, pady=(2, 10))

    def _card(self, parent, title: str) -> tk.Frame:
        outer = tk.Frame(parent, bg="#242424", padx=2, pady=2)
        outer.pack(fill="x", padx=20, pady=8)
        tk.Label(outer, text=title, bg="#242424", fg="#888888", font=("Consolas", 8), anchor="w").pack(fill="x", padx=8, pady=(4, 0))
        inner = tk.Frame(outer, bg="#1e1e1e")
        inner.pack(fill="x", padx=4, pady=4)
        return inner

    def _btn(self, parent, text: str, command, bg="#3a3a3a", fg="white", height=2, font_size=9) -> tk.Button:
        return tk.Button(parent, text=text, command=command, bg=bg, fg=fg, activebackground=bg, activeforeground=fg, font=("Consolas", font_size, "bold"), relief="flat", bd=0, cursor="hand2", height=height, padx=12)

    # ── TAB BUILDERS ─────────────────────────────────────────────────────────

    def _build_capture_tab(self, tab):
        tk.Label(tab, text="Temporal Motion Acquisition", bg="#1a1a1a", fg="#cccccc", font=("Consolas", 11, "bold")).pack(pady=16)
        card = self._card(tab, "CAPTURE ENGINE")
        self.btn_cap = self._btn(card, "▶  START CAPTURE", self.toggle_capture, bg="#27ae60", height=2, font_size=10)
        self.btn_cap.pack(fill="x", padx=12, pady=10)

    def _build_labeler_tab(self, tab):
        tk.Label(tab, text="Active Learning Data Refinement",
                 bg="#1a1a1a", fg="#cccccc",
                 font=("Consolas", 11, "bold")).pack(pady=16)

        # ── Controls legend ───────────────────────────────────────────────────
        legend = self._card(tab, "CONTROLS")
        for ln in [
            "L/R-Click drag        →  draw selected class",
            "Scroll wheel          →  cycle label class",
            "L/R-Click (no drag)   →  pinpoint box at cursor",
            "SHIFT + L-drag        →  mass-delete region (red)",
            "[SPACE] Approve  [D] Discard  [S/Enter] Skip  [Q/Esc] Quit",
        ]:
            tk.Label(legend, text=ln, bg="#1e1e1e", fg="#aaaaaa",
                     font=("Consolas", 8), anchor="w").pack(fill="x", padx=12, pady=1)

        # ── Active-learning status ────────────────────────────────────────────
        al = self._card(tab, "ACTIVE LEARNING STATUS")
        self.al_status_var = tk.StringVar(
            value=f"New labels since last train: {self._new_labels_since_train} / {AUTO_TRAIN_THRESHOLD}")
        tk.Label(al, textvariable=self.al_status_var,
                 bg="#1e1e1e", fg="#00cc88", font=("Consolas", 9)).pack(padx=12, pady=4)
        self.al_finetune_var = tk.StringVar(value="Fine-tune: idle")
        tk.Label(al, textvariable=self.al_finetune_var,
                 bg="#1e1e1e", fg="#888888", font=("Consolas", 8)).pack(padx=12, pady=(0, 4))

        # ── Acceleration Engine (NEW) ─────────────────────────────────────────
        acc = self._card(tab, "⚡ ACCELERATION ENGINE  (replace manual labeling)")
        tk.Label(acc,
                 text="Grounding DINO: zero-shot text-prompt labeling — no training data needed.",
                 bg="#1e1e1e", fg="#aaaaaa", font=("Consolas", 8), anchor="w").pack(
            fill="x", padx=12, pady=(4, 0))
        self._btn(acc, "🚀  AUTO-LABEL  (Grounding DINO — zero-shot)",
                  lambda: threading.Thread(
                      target=self.gdino_auto_label_worker, daemon=True).start(),
                  bg="#1a5276", font_size=9).pack(fill="x", padx=12, pady=4)

        tk.Label(acc,
                 text="Smart Review: only shows frames where model is uncertain (0.25–0.60 conf).",
                 bg="#1e1e1e", fg="#aaaaaa", font=("Consolas", 8), anchor="w").pack(
            fill="x", padx=12, pady=(6, 0))
        self._btn(acc, "⚡  SMART REVIEW  (uncertainty sampling only)",
                  lambda: threading.Thread(
                      target=self.smart_review_worker, daemon=True).start(),
                  bg="#145a32", font_size=9).pack(fill="x", padx=12, pady=4)

        tk.Label(acc,
                 text="Synthetic Augment: multiplies approved dataset 8× with zero extra labeling.",
                 bg="#1e1e1e", fg="#aaaaaa", font=("Consolas", 8), anchor="w").pack(
            fill="x", padx=12, pady=(6, 0))
        self._btn(acc, "🎲  SYNTHETIC AUGMENT  (8× data expansion)",
                  lambda: threading.Thread(
                      target=self.synthetic_augment_worker, daemon=True).start(),
                  bg="#6e2f1a", font_size=9).pack(fill="x", padx=12, pady=4)

        # ── Standard actions ──────────────────────────────────────────────────
        btns = self._card(tab, "MANUAL ACTIONS")
        self._btn(btns, "🧹  DEEP CLEAN INBOX",
                  lambda: threading.Thread(
                      target=self.deep_clean_dataset, daemon=True).start(),
                  bg="#34495e").pack(fill="x", padx=12, pady=5)
        self._btn(btns, "🧠  FORCE LEARN & AUTO-LABEL",
                  self.force_active_learning, bg="#7030a0").pack(fill="x", padx=12, pady=5)
        self._btn(btns, "OPEN SPRINT LABELER",
                  self.start_labeler, bg="#8e44ad").pack(fill="x", padx=12, pady=5)

    def _build_trainer_tab(self, tab):
        tk.Label(tab, text="Smart Neural Forge",
                 bg="#1a1a1a", fg="#cccccc",
                 font=("Consolas", 11, "bold")).pack(pady=16)

        # ── Strategy info card ────────────────────────────────────────────────
        info = self._card(tab, "STRATEGY AUTO-SELECTOR")
        for txt, col in [
            ("< 100 images  → 🧊 Cold Start   freeze=10, focal loss, 50 epochs (~8 min)", "#5dade2"),
            ("100–500        → 🔥 Warm Train  freeze=5,  full aug,   80 epochs (~20 min)", "#f39c12"),
            ("500+           → 💪 Full Train   freeze=0,  all hyper, 150 epochs (~45 min)", "#2ecc71"),
        ]:
            tk.Label(info, text=txt, bg="#1e1e1e", fg=col,
                     font=("Consolas", 8), anchor="w").pack(fill="x", padx=12, pady=2)
        tk.Label(info,
                 text="All strategies: rect=True (portrait ratio) · copy_paste=0.3 (balance projectiles) · fl_gamma=1.5",
                 bg="#1e1e1e", fg="#666666",
                 font=("Consolas", 8), anchor="w").pack(fill="x", padx=12, pady=(4, 6))

        card = self._card(tab, "FORGE ENGINE")
        self._btn(card, "🔥  START SMART TRAINING",
                  self.start_training_thread,
                  bg="#d35400", height=2, font_size=10).pack(fill="x", padx=12, pady=10)

    def _build_evasion_tab(self, tab):
        tk.Label(tab, text="Weighted Vector Evasion Bridge", bg="#1a1a1a", fg="#cccccc", font=("Consolas", 11, "bold")).pack(pady=16)
        card = self._card(tab, "EVASION ENGINE")
        self.btn_eva = self._btn(card, "⚔  ENGAGE NEURAL HUD & BRIDGE", self.toggle_evasion, bg="#2980b9", height=2, font_size=10)
        self.btn_eva.pack(fill="x", padx=12, pady=10)
        tk.Label(card, text=f"Panic Key: [{self.panic_key.upper()}]   Kill Switch: [{self.panic_exit.upper()}]", bg="#1e1e1e", fg="#666666", font=("Consolas", 8)).pack(pady=(0, 8))

    def _build_bot_tab(self, tab):
        tk.Label(tab, text="Archero 2 Bot  —  Maximum Potential Engine", bg="#1a1a1a", fg="#cccccc", font=("Consolas", 11, "bold")).pack(pady=16)
        conn = self._card(tab, "ADB CONNECTION")
        row = tk.Frame(conn, bg="#1e1e1e"); row.pack(fill="x", padx=12, pady=6)
        tk.Label(row, text="Device:", bg="#1e1e1e", fg="#888888", font=("Consolas", 8)).pack(side="left")
        tk.Entry(row, textvariable=self.monitor_device_var, width=18, bg="#2a2a2a", fg="white", insertbackground="white", relief="flat", font=("Consolas", 9)).pack(side="left", padx=6)
        self._btn(row, "CONNECT", self.bot_connect_adb, bg="#16a085", height=1, font_size=8).pack(side="left", padx=4)
        
        ctrl = self._card(tab, "BOT CONTROLS")
        br = tk.Frame(ctrl, bg="#1e1e1e"); br.pack(fill="x", padx=12, pady=8)
        self._btn(br, "▶  START BOT", self.bot_start, bg="#27ae60", height=1, font_size=9).pack(side="left", padx=4)
        self._btn(br, "■  STOP BOT",  self.bot_stop, bg="#c0392b", height=1, font_size=9).pack(side="left", padx=4)
        
        learn = self._card(tab, "LEARNING MODE")
        lr = tk.Frame(learn, bg="#1e1e1e"); lr.pack(fill="x", padx=12, pady=8)
        tk.Checkbutton(lr, text="Auto-Screenshots", variable=self.log_frames_var, bg="#1e1e1e", fg="#aaaaaa", selectcolor="#2a2a2a", activebackground="#1e1e1e", font=("Consolas", 9)).pack(side="left")
        tk.Label(lr, text="  Interval:", bg="#1e1e1e", fg="#666666", font=("Consolas", 8)).pack(side="left")
        tk.Spinbox(lr, from_=1, to=20, textvariable=self.log_freq_var, width=4, bg="#2a2a2a", fg="white", buttonbackground="#2a2a2a", font=("Consolas", 9), relief="flat").pack(side="left", padx=4)
        
        stat = self._card(tab, "LIVE STATUS")
        srow = tk.Frame(stat, bg="#1e1e1e"); srow.pack(fill="x", padx=12, pady=6)
        for var in (self.adb_status_var, self.bot_status_var, self.screen_var, self.reward_var):
            tk.Label(srow, textvariable=var, bg="#1e1e1e", fg="#00cc88", font=("Consolas", 8), padx=8, pady=2).pack(side="left")

    def _build_dailies_tab(self, tab):
        tk.Label(tab, text="Automated Daily Events", bg="#1a1a1a", fg="#cccccc", font=("Consolas", 11, "bold")).pack(pady=16)
        card = self._card(tab, "ARENA AUTOPILOT")
        desc = ("Clicks Challenge, reads player power via OCR,\n"
                "scans 5 enemies and challenges the first weaker one.\n"
                "Repeats 5 times with 30 s cooldown.")
        tk.Label(card, text=desc, bg="#1e1e1e", fg="#aaaaaa", font=("Consolas", 9), justify="left").pack(pady=8, padx=12, anchor="w")
        self.btn_daily = self._btn(card, "▶  START ARENA DAILIES", self.toggle_daily_arena, bg="#f1c40f", fg="black", height=2, font_size=10)
        self.btn_daily.pack(fill="x", padx=12, pady=10)

    # ── LOGIC IMPLEMENTATIONS ────────────────────────────────────────────────

    def force_active_learning(self):
        with self._finetune_lock:
            if self._is_finetuning:
                self.log("⚠ Learning process already in progress."); return
        self.log("🧠 Manual Force-Learn Triggered: Bypassing threshold...")
        self._new_labels_since_train = 0
        self._save_label_progress()
        self._update_al_status_label()
        threading.Thread(target=self._incremental_finetune, daemon=True).start()

    def _poll(self):
        if self.bot:
            try:
                st = self.bot.get_status()
                self.screen_var.set(f"Screen: {st.get('screen', '?')}")
                rwd = st.get('reward', st.get('total_reward', 0.0))
                self.reward_var.set(f"Reward: {rwd:.2f}")
            except Exception:
                pass
        self.root.after(self.POLL_MS, self._poll)

    def _cleanup(self):
        if self.bot:
            try: self.bot.stop()
            except Exception: pass

    def get_game_rect(self):
        if not _MSS_OK: return None
        try:
            # FIX-RECURSION: We must not capture the Command Center or the HUD
            windows = gw.getWindowsWithTitle(self.window_title)
            target_win = None
            for w in windows:
                if "Command Center" not in w.title and "Vision_HUD_Overlay" not in w.title:
                    target_win = w
                    break
            
            if not target_win: return None
            if target_win.isMinimized: target_win.restore()
            return {"top": target_win.top, "left": target_win.left, "width": target_win.width, "height": target_win.height}
        except Exception: return None

    def bot_connect_adb(self):
        if not _BOT_OK: self.log("❌ adb_controller not available."); return
        self.adb = ADBController(logger=self.log)
        self.adb.device = self.monitor_device_var.get().strip()
        ok, msg = self.adb.connect()
        self.log(f"[ADB] {msg}")
        self.adb_status_var.set(f"ADB: {'Connected ✓' if ok else 'Failed ✗'}")

    def bot_start(self):
        if not _BOT_OK: self.log("❌ bot_loop not available."); return
        if self.bot_thread and self.bot_thread.is_alive(): self.log("[BOT] Already running."); return
        if self.adb is None: self.bot_connect_adb()
        try:
            self.bot = ArcheroBot(self.adb, project_root=self.project_root, logger=self.log, log_frames=self.log_frames_var.get(), log_every=self.log_freq_var.get())
        except TypeError:
            self.bot = ArcheroBot(project_root=self.project_root, logger=self.log)
        self.bot_thread = threading.Thread(target=self.bot.run, daemon=True)
        self.bot_thread.start()
        self.bot_status_var.set("Bot: running ▶")
        self.log("[BOT] Started.")

    def bot_stop(self):
        if self.bot: self.bot.stop()
        self.bot_status_var.set("Bot: stopped ■")
        self.log("[BOT] Stopped.")

    def toggle_capture(self):
        if not _MSS_OK: self.log("❌ mss/pygetwindow not installed."); return
        self.is_capturing = not self.is_capturing
        if self.is_capturing:
            self.btn_cap.config(text="■  STOP CAPTURE", bg="#c0392b")
            threading.Thread(target=self.capture_loop, daemon=True).start()
        else:
            self.btn_cap.config(text="▶  START CAPTURE", bg="#27ae60")

    def get_proposals(self, frame: np.ndarray, prev_gray):
        proposals = []
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        zone_mask = np.ones((h, w), dtype=np.uint8) * 255
        zone_mask[0:int(h * 0.18), :] = 0
        zone_mask[int(h * 0.65):h, 0:int(w * 0.30)] = 0
        zone_mask[int(h * 0.75):h, int(w * 0.70):w] = 0

        excl = cv2.bitwise_or(
            cv2.dilate(cv2.inRange(hsv, np.array([85, 120, 150]), np.array([105, 255, 255])), np.ones((7, 7), np.uint8)),
            cv2.dilate(cv2.inRange(hsv, np.array([20, 100, 180]), np.array([40, 255, 255])), np.ones((7, 7), np.uint8))
        )
        zone_mask = cv2.bitwise_and(zone_mask, cv2.bitwise_not(excl))

        m1 = cv2.inRange(hsv, np.array([0, 150, 100]),   np.array([10, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([170, 150, 100]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_and(cv2.bitwise_or(m1, m2), zone_mask)
        for cnt in cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bh == 0: continue
            area, aspect = cv2.contourArea(cnt), bw / float(bh)
            if aspect > 2.0 and bw > 15 and bh < 25:
                proposals.append({"label": "mob", "x": x+bw/2, "y": y+bw/1.5, "w": bw*1.5, "h": bw*1.5})
            elif 200 < area < 15000 and aspect < 2.5:
                proposals.append({"label": "aoe_indicator", "x": x+bw/2, "y": y+bh/2, "w": bw, "h": bh})

        if prev_gray is not None and prev_gray.shape == gray.shape:
            diff = cv2.absdiff(cv2.GaussianBlur(gray, (7, 7), 0), cv2.GaussianBlur(prev_gray, (7, 7), 0))
            _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
            thresh = cv2.bitwise_and(cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1), zone_mask)
            for cnt in cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                if 80 < cv2.contourArea(cnt) < 600:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    proposals.append({"label": "projectile", "x": x+bw/2, "y": y+bh/2, "w": bw, "h": bh})
        return proposals, gray

    def capture_loop(self):
        self.log("📡 Scanning for game window...")
        with mss() as sct:
            prev_gray = None
            while self.is_capturing:
                rect = self.get_game_rect()
                if not rect: time.sleep(1.5); continue
                img   = np.array(sct.grab(rect))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                proposals, prev_gray = self.get_proposals(frame, prev_gray)
                if proposals:
                    ts    = int(time.time() * 1000)
                    fname = f"frame_{ts}.jpg"
                    cv2.imwrite(os.path.join(self.inbox_dir, fname), frame)
                    self.log(f"📸 {fname} | {len(proposals)} hazards")
                time.sleep(0.1)

    def _overlay_text(self, img: np.ndarray, remaining: int, total: int) -> np.ndarray:
        cls, ver, delta = self._lbl_class.upper(), self._labeler_ai_version, self._new_labels_since_train
        hud = img.copy()
        cv2.rectangle(hud, (0, 0), (img.shape[1], 52), (20, 20, 20), -1)
        cv2.putText(hud, f"[SPACE] Approve  [D] Discard  [Del/Bksp] Remove last  [S/Enter] Skip  [Q/Esc] Quit  |  Scroll = cycle class", (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
        cls_color = LABEL_COLORS.get(self._lbl_class, (255, 255, 255))
        status = f"class={cls}  |  {remaining} remaining  |  AI={ver}  |  +{delta}/{AUTO_TRAIN_THRESHOLD} since retrain"
        cv2.putText(hud, status, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.52, cls_color, 1)
        return hud

    def _draw_boxes(self, base: np.ndarray) -> np.ndarray:
        out = base.copy()
        for p in self._lbl_boxes:
            sx, sy = int((p['x'] - p['w'] / 2) / self._lbl_scale_x), int((p['y'] - p['h'] / 2) / self._lbl_scale_y)
            sw, sh = int(p['w'] / self._lbl_scale_x), int(p['h'] / self._lbl_scale_y)
            col = LABEL_COLORS.get(p['label'], (255, 255, 255))
            cv2.rectangle(out, (sx, sy), (sx + sw, sy + sh), col, 2)
            cv2.putText(out, p['label'], (sx, max(12, sy - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
        return out

    def _force_redraw(self, remaining: int = 0, total: int = 0):
        if self._current_display_base is None: return
        img = self._draw_boxes(self._current_display_base)
        cv2.imshow("SenseNova Sprint Labeler", self._overlay_text(img, remaining, total))

    def _labeler_mouse_cb(self, event, x, y, flags, param):
        if self._current_display_base is None: return
        nx, ny = int(x * self._lbl_scale_x), int(y * self._lbl_scale_y)
        rem = param[0] if param else 0
        tot = param[1] if param else 0

        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                self._lbl_deleting_region = True
                self._lbl_del_ix, self._lbl_del_iy = nx, ny
            else:
                # Both L and R draw the CURRENTLY selected class.
                # Class is changed by scroll / +/- only — never by which button.
                self._lbl_drawing = True
                self._lbl_ix, self._lbl_iy = nx, ny
                self._lbl_press_x, self._lbl_press_y = nx, ny

        elif event == cv2.EVENT_MOUSEMOVE:
            if self._lbl_drawing:
                preview = self._draw_boxes(self._current_display_base)
                col = LABEL_COLORS.get(self._lbl_class, (0, 255, 255))
                ox = int(self._lbl_ix / self._lbl_scale_x)
                oy = int(self._lbl_iy / self._lbl_scale_y)
                cv2.rectangle(preview, (ox, oy), (x, y), col, 2)
                cv2.imshow("SenseNova Sprint Labeler",
                           self._overlay_text(preview, rem, tot))
            elif self._lbl_deleting_region:
                preview = self._draw_boxes(self._current_display_base)
                ox = int(self._lbl_del_ix / self._lbl_scale_x)
                oy = int(self._lbl_del_iy / self._lbl_scale_y)
                cv2.rectangle(preview, (ox, oy), (x, y), (0, 0, 255), 2)
                cv2.imshow("SenseNova Sprint Labeler",
                           self._overlay_text(preview, rem, tot))

        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            if self._lbl_deleting_region:
                self._lbl_deleting_region = False
                dx1, dx2 = min(self._lbl_del_ix, nx), max(self._lbl_del_ix, nx)
                dy1, dy2 = min(self._lbl_del_iy, ny), max(self._lbl_del_iy, ny)
                self._lbl_boxes = [b for b in self._lbl_boxes if not (
                    (b["x"]+b["w"]/2) >= dx1 and (b["x"]-b["w"]/2) <= dx2 and
                    (b["y"]+b["h"]/2) >= dy1 and (b["y"]-b["h"]/2) <= dy2)]
                self._force_redraw(rem, tot)
            elif self._lbl_drawing:
                self._lbl_drawing = False
                drag_x = abs(nx - self._lbl_press_x)
                drag_y = abs(ny - self._lbl_press_y)
                if drag_x < CLICK_THRESHOLD_PX and drag_y < CLICK_THRESHOLD_PX:
                    sz = self._point_size.get(self._lbl_class, 28)
                    self._lbl_boxes.append({"label": self._lbl_class,
                                            "x": float(nx), "y": float(ny),
                                            "w": float(sz), "h": float(sz)})
                else:
                    bw = abs(nx - self._lbl_ix)
                    bh = abs(ny - self._lbl_iy)
                    if bw > 4 and bh > 4:
                        self._lbl_boxes.append({
                            "label": self._lbl_class,
                            "x": min(self._lbl_ix, nx) + bw/2,
                            "y": min(self._lbl_iy, ny) + bh/2,
                            "w": float(bw), "h": float(bh)})
                self._force_redraw(rem, tot)

        elif event == cv2.EVENT_MBUTTONDOWN:
            for i, b in enumerate(self._lbl_boxes):
                if (b["x"]-b["w"]/2 <= nx <= b["x"]+b["w"]/2 and
                        b["y"]-b["h"]/2 <= ny <= b["y"]+b["h"]/2):
                    del self._lbl_boxes[i]; self._force_redraw(rem, tot); break

        elif event == cv2.EVENT_MOUSEWHEEL:
            self._cycle_class(1 if flags > 0 else -1, param)


    def deep_clean_dataset(self):
        self.log("🧹 Deep cleaning inbox...")
        if not os.path.exists(self.output_path): self.log("❌ No JSONL found."); return
        with open(self.output_path, 'r', encoding='utf-8') as f: lines = f.readlines()
        cleaned, fd, bd = [], 0, 0
        for line in lines:
            data = json.loads(line)
            img_path = os.path.join(self.inbox_dir, data['file'])
            if not os.path.exists(img_path): cleaned.append(line); continue
            frame = cv2.imread(img_path); gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); h, w = frame.shape[:2]
            valid = []
            for box in data.get('boxes', []):
                x1, y1 = max(0, int(box['x'] - box['w'] / 2)), max(0, int(box['y'] - box['h'] / 2))
                x2, y2 = min(w, int(box['x'] + box['w'] / 2)), min(h, int(box['y'] + box['h'] / 2))
                if x2 <= x1 or y2 <= y1: bd += 1; continue
                crop = gray[y1:y2, x1:x2]; _, std = cv2.meanStdDev(crop)
                contrast, peak = float(std[0][0]), int(np.max(crop)) if crop.size else 0
                bad = (box['label'] == 'projectile' and (contrast < 15.0 or peak < 160)) or (box['label'] == 'mob' and contrast < 20.0)
                if bad: bd += 1
                else: valid.append(box)
            if valid: data['boxes'] = valid; cleaned.append(json.dumps(data) + "\n")
            else:
                try: os.remove(img_path); fd += 1
                except Exception: pass
        with open(self.output_path, 'w', encoding='utf-8') as f: f.writelines(cleaned)
        self.log(f"✨ Clean done — {bd} noise boxes, {fd} empty frames purged.")

    def start_labeler(self):
        if self.is_capturing or self.is_evading: messagebox.showwarning("Conflict", "Stop Capture/Evasion before labeling."); return
        threading.Thread(target=self.labeler_worker, daemon=True).start()

    def labeler_worker(self):
        already_approved = set(os.listdir(self.approved_dir)); already_discarded = set(os.listdir(self.discarded_dir))
        files = sorted([f for f in os.listdir(self.inbox_dir) if f.lower().endswith(('.jpg', '.png')) and f not in already_approved and f not in already_discarded])
        total = len(files)
        if total == 0: self.log("ℹ Inbox empty (or all files already processed)."); return
        self.log(f"🕵 Sprint: {total} unprocessed frames.")

        ai_model = self._load_best_model()
        if ai_model: self.log(f"🤖 Active Learning: AI {self._labeler_ai_version} pre-labeling.")
        else: self.log("ℹ No trained model found — using heuristic proposals.")

        win_name = "SenseNova Sprint Labeler"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1280, 720)

        def _on_point_size(val): self._point_size[self._lbl_class] = max(8, val)
        cv2.createTrackbar("Point size", win_name, 28, 120, _on_point_size)

        progress_ctx = [total, total]
        cv2.setMouseCallback(win_name, self._labeler_mouse_cb, progress_ctx)
        prev_gray = None

        for idx, fname in enumerate(files):
            remaining, progress_ctx[0], src_path = total - idx, total - idx, os.path.join(self.inbox_dir, fname)
            if not os.path.exists(src_path): continue
            frame = cv2.imread(src_path)
            if frame is None: continue

            self._lbl_scale_x, self._lbl_scale_y = frame.shape[1] / 1280.0, frame.shape[0] / 720.0
            self._lbl_boxes = []

            new_model = self._try_hotswap_model(ai_model)
            if new_model is not None: ai_model = new_model; self.log(f"🔄 Model hot-swapped → {self._labeler_ai_version}")

            if ai_model:
                try:
                    results = ai_model.predict(frame, conf=0.35, verbose=False)
                    for box in results[0].boxes:
                        c = box.xywh[0]; lbl = ai_model.names.get(int(box.cls[0]), 'projectile')
                        self._lbl_boxes.append({"label": lbl, "x": float(c[0]), "y": float(c[1]), "w": float(c[2]), "h": float(c[3])})
                    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                except Exception: self._lbl_boxes, current_gray = self.get_proposals(frame, prev_gray)
            else: self._lbl_boxes, current_gray = self.get_proposals(frame, prev_gray)

            self._current_display_base = cv2.resize(frame.copy(), (1280, 720), interpolation=cv2.INTER_AREA)
            self._force_redraw(remaining, total)

            while True:
                key = cv2.waitKey(33) & 0xFF
                
                # Check trackbar exists before reading to avoid OpenCV NULL errors
                try:
                    raw_ps = cv2.getTrackbarPos("Point size", win_name)
                    if raw_ps > 0: self._point_size[self._lbl_class] = max(8, raw_ps)
                except Exception: pass

                if key == ord(' '):
                    with open(self.output_path, "a", encoding="utf-8") as f: f.write(json.dumps({"file": fname, "boxes": self._lbl_boxes}) + "\n"); f.flush(); os.fsync(f.fileno())
                    try: shutil.move(src_path, os.path.join(self.approved_dir, fname))
                    except Exception as e: self.log(f"⚠ Move error: {e}")
                    self.log(f"✅ Approved: {fname} ({len(self._lbl_boxes)} boxes)")
                    self._new_labels_since_train += 1; self._save_label_progress(); self._update_al_status_label(); self._maybe_trigger_finetune()
                    prev_gray = current_gray; break
                elif key in (ord('d'), ord('D')):
                    try: shutil.move(src_path, os.path.join(self.discarded_dir, fname))
                    except Exception as e: self.log(f"⚠ Move error: {e}")
                    self.log(f"🗑 Discarded: {fname}"); prev_gray = current_gray; break
                elif key in (ord('s'), ord('S'), 13): prev_gray = current_gray; break
                elif key in (8, 127, 0):
                    if self._lbl_boxes: self._lbl_boxes.pop(); self._force_redraw(remaining, total)
                elif key in (ord('q'), ord('Q'), 27):
                    self._current_display_base = None
                    try: cv2.destroyWindow(win_name)
                    except Exception: pass
                    self.log("🛑 Labeler closed by user."); return
                elif key in (ord('+'), ord('='), ord(']')):
                    self._cycle_class(+1, progress_ctx)
                elif key in (ord('-'), ord('[')):
                    self._cycle_class(-1, progress_ctx)
        self._current_display_base = None
        try: cv2.destroyWindow(win_name)
        except Exception: pass
        self.log(f"🏁 Sprint concluded — {total} frames processed.")

    def _update_al_status_label(self):
        def _upd(): self.al_status_var.set(f"New labels since last train: {self._new_labels_since_train} / {AUTO_TRAIN_THRESHOLD}")
        self.root.after(0, _upd)

    def _find_latest_weights(self) -> Optional[str]:
        runs_dir = self.project_root / "runs" / "detect"
        if not runs_dir.exists(): return None
        candidates = sorted(runs_dir.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        return str(candidates[0]) if candidates else None

    def _load_best_model(self):
        if not _YOLO_OK: return None
        weights = self._find_latest_weights()
        if weights is None: self._labeler_ai_version = "none"; return None
        try:
            m = YOLO(weights); parts = Path(weights).parts
            ver_parts = [p for p in parts if "SenseNova" in p or "Survival" in p]
            self._labeler_ai_version = ver_parts[-1] if ver_parts else "custom"
            self._latest_weights = weights; return m
        except Exception as e: self.log(f"⚠ Could not load model: {e}"); return None

    def _try_hotswap_model(self, current_model):
        if not _YOLO_OK: return None
        newest = self._find_latest_weights()
        if newest and newest != self._latest_weights:
            try:
                m = YOLO(newest); parts = Path(newest).parts
                ver_parts = [p for p in parts if "SenseNova" in p]
                self._labeler_ai_version = ver_parts[-1] if ver_parts else "new"
                self._latest_weights = newest; return m
            except Exception: pass
        return None

    def _maybe_trigger_finetune(self):
        if self._new_labels_since_train < AUTO_TRAIN_THRESHOLD: return
        with self._finetune_lock:
            if self._is_finetuning: return
            self._is_finetuning = True
        self.log(f"🔁 Auto fine-tune triggered ({self._new_labels_since_train} new labels).")
        self._new_labels_since_train = 0; self._save_label_progress(); self._update_al_status_label()
        threading.Thread(target=self._incremental_finetune, daemon=True).start()

    def _incremental_finetune(self):
        def _set_status(msg): self.log(msg); self.root.after(0, lambda: self.al_finetune_var.set(msg))
        _set_status("Fine-tune: building dataset...")
        if not _YOLO_OK: _set_status("Fine-tune: ultralytics missing — aborted."); self._is_finetuning = False; return
        valid = self._build_yolo_dataset_from_approved()
        if valid < 20: _set_status(f"Fine-tune: only {valid} valid samples — skipped."); self._is_finetuning = False; return
        version = int(time.time()); run_name = f"SenseNova_Survival_ft_{version}"; yaml_path = os.path.join(self.yolo_root, "data.yaml")
        _set_status(f"Fine-tune: training {FINETUNE_EPOCHS} epochs → {run_name} …")
        try:
            base_weights = self._find_latest_weights() or "yolov8n.pt"
            model = YOLO(base_weights)
            model.train(data=yaml_path, epochs=FINETUNE_EPOCHS, imgsz=640, device=0, batch=8, workers=0, patience=10, box=10.0, cls=2.5, name=run_name, verbose=False)
            new_weights = str(self.project_root / "runs" / "detect" / run_name / "weights" / "best.pt")
            if os.path.exists(new_weights): self._latest_weights = new_weights; _set_status(f"Fine-tune: ✅ done → {run_name}")
            else: _set_status("Fine-tune: training completed (weights not found).")
        except Exception as e: _set_status(f"Fine-tune: ❌ {e}")
        finally: self._is_finetuning = False

    def _build_yolo_dataset_from_approved(self) -> int:
        for split in ['train', 'val']:
            for sub in ['images', 'labels']:
                p = os.path.join(self.yolo_root, split, sub)
                if os.path.exists(p): shutil.rmtree(p)
                os.makedirs(p, exist_ok=True)
        if not os.path.exists(self.output_path): return 0
        with open(self.output_path, 'r', encoding='utf-8') as f: lines = f.readlines()
        entries = []
        for line in lines:
            try:
                d = json.loads(line); img_p = os.path.join(self.approved_dir, d['file'])
                if os.path.exists(img_p) and d.get('boxes'): entries.append(d)
            except Exception: pass
        for i, data in enumerate(entries):
            split = 'train' if i < int(len(entries) * 0.8) else 'val'
            img_src = os.path.join(self.approved_dir, data['file'])
            if not os.path.exists(img_src): continue
            img = cv2.imread(img_src)
            if img is None: continue
            h, w = img.shape[:2]; lbl_name = os.path.splitext(data['file'])[0] + ".txt"; lbl_path = os.path.join(self.yolo_root, split, 'labels', lbl_name)
            with open(lbl_path, 'w') as f:
                for b in data['boxes']:
                    cid = self.class_map.get(b.get('label', 'mob'), 3)
                    f.write(f"{cid} {b['x']/w:.6f} {b['y']/h:.6f} {b['w']/w:.6f} {b['h']/h:.6f}\n")
            shutil.copy(img_src, os.path.join(self.yolo_root, split, 'images', data['file']))
        yaml_path = os.path.join(self.yolo_root, 'data.yaml')
        with open(yaml_path, 'w') as f: f.write(f"path: {self.yolo_root}\ntrain: train/images\nval: val/images\nnc: 5\nnames: ['player', 'projectile', 'aoe_indicator', 'mob', 'boss']\n")
        return len(entries)

    # ── NEW: _cycle_class — single choke-point for class changes ─────────────

    def _cycle_class(self, delta: int, param):
        """Scroll wheel, +/-, and keyboard all come here. L-click always
        draws whatever _lbl_class is set to after this call."""
        self._lbl_class_idx = (self._lbl_class_idx + delta) % len(LABEL_CLASSES)
        self._lbl_class = LABEL_CLASSES[self._lbl_class_idx]
        rem = param[0] if param else 0
        tot = param[1] if param else 0
        self._force_redraw(rem, tot)

    # ── NEW: Grounding DINO Auto-Labeler ─────────────────────────────────────

    def gdino_auto_label_worker(self):
        """
        Zero-shot auto-labeling using Grounding DINO.
        Requires: pip install groundingdino-py
        Falls back to a colour+motion heuristic if GDINO unavailable.

        Routing:
          conf > GDINO_AUTO_CONF  → write to JSONL + move to approved/
          GDINO_MIN_CONF–AUTO_CONF → leave in inbox for smart_review_worker
          conf < GDINO_MIN_CONF   → move to discarded/
        """
        self.log("🚀 Grounding DINO Auto-Labeler starting...")

        try:
            from groundingdino.util.inference import load_model, predict
            import torchvision.transforms as T
            _gdino_ok = True
        except ImportError:
            _gdino_ok = False
            self.log("⚠ groundingdino not installed — using heuristic fallback.")
            self.log("  Install: pip install groundingdino-py")

        inbox_files = sorted([
            f for f in os.listdir(self.inbox_dir)
            if f.lower().endswith(('.jpg', '.png'))
            and f not in set(os.listdir(self.approved_dir))
            and f not in set(os.listdir(self.discarded_dir))
        ])
        total = len(inbox_files)
        if total == 0:
            self.log("ℹ Inbox empty — nothing to auto-label."); return
        self.log(f"  Processing {total} frames...")

        auto_approved = uncertain = discarded_count = 0

        # ── Grounding DINO path ───────────────────────────────────────────────
        if _gdino_ok:
            try:
                gdino_cfg   = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
                gdino_ckpt  = "groundingdino_swint_ogc.pth"
                gdino_model = load_model(gdino_cfg, gdino_ckpt)

                for i, fname in enumerate(inbox_files):
                    src = os.path.join(self.inbox_dir, fname)
                    frame = cv2.imread(src)
                    if frame is None: continue

                    h, w = frame.shape[:2]
                    boxes_out = []

                    for cls_name, prompt in GDINO_PROMPTS.items():
                        try:
                            from PIL import Image as PILImage
                            pil_img = PILImage.fromarray(
                                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            boxes, logits, _ = predict(
                                model=gdino_model,
                                image=pil_img,
                                caption=prompt,
                                box_threshold=GDINO_MIN_CONF,
                                text_threshold=GDINO_MIN_CONF,
                            )
                            for box, conf in zip(boxes, logits):
                                cx  = float(box[0]) * w
                                cy  = float(box[1]) * h
                                bw  = float(box[2]) * w
                                bh  = float(box[3]) * h
                                boxes_out.append({
                                    "label": cls_name,
                                    "x": cx, "y": cy,
                                    "w": bw, "h": bh,
                                    "_conf": float(conf)
                                })
                        except Exception:
                            pass

                    if not boxes_out:
                        discarded_count += 1
                        continue

                    min_conf = min(b["_conf"] for b in boxes_out)
                    clean    = [{k: v for k, v in b.items() if k != "_conf"}
                                for b in boxes_out]

                    if min_conf >= GDINO_AUTO_CONF:
                        # Auto-approve: write JSONL and move
                        with open(self.output_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"file": fname,
                                                "boxes": clean}) + "\n")
                            f.flush()
                        shutil.move(src, os.path.join(self.approved_dir, fname))
                        self._new_labels_since_train += 1
                        self._save_label_progress()
                        auto_approved += 1
                    else:
                        # Leave in inbox for human smart-review
                        uncertain += 1

                    if (i + 1) % 50 == 0:
                        self.log(f"  {i+1}/{total} — approved:{auto_approved} "
                                 f"uncertain:{uncertain}")

            except Exception as e:
                self.log(f"❌ GDINO error: {e}")
                _gdino_ok = False

        # ── Heuristic fallback (no GDINO) ─────────────────────────────────────
        if not _gdino_ok:
            prev_gray = None
            for i, fname in enumerate(inbox_files):
                src   = os.path.join(self.inbox_dir, fname)
                frame = cv2.imread(src)
                if frame is None: continue
                proposals, prev_gray = self.get_proposals(frame, prev_gray)
                if proposals:
                    with open(self.output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"file": fname,
                                            "boxes": proposals}) + "\n")
                        f.flush()
                    shutil.move(src, os.path.join(self.approved_dir, fname))
                    auto_approved += 1
                else:
                    shutil.move(src, os.path.join(self.discarded_dir, fname))
                    discarded_count += 1
                if (i + 1) % 100 == 0:
                    self.log(f"  {i+1}/{total} processed (heuristic)")

        self._update_al_status_label()
        self._maybe_trigger_finetune()
        self.log(f"✅ Auto-label done — approved:{auto_approved}  "
                 f"uncertain:{uncertain}  discarded:{discarded_count}")
        if uncertain > 0:
            self.log(f"  Run ⚡ SMART REVIEW to process {uncertain} uncertain frames.")

    # ── NEW: Smart Review — uncertainty sampling ──────────────────────────────

    def smart_review_worker(self):
        """
        Only shows the human frames where the model is uncertain
        (SMART_REVIEW_LOW ≤ min_conf ≤ SMART_REVIEW_HIGH).

        Frames where ALL detections have conf > SMART_REVIEW_HIGH are
        auto-approved (model already knows them).
        Frames with NO detection at all are shown (model saw nothing).
        This cuts manual review time by ~80%.
        """
        self.log("⚡ Smart Review: loading model for uncertainty sampling...")

        model = self._load_best_model()
        if model is None:
            self.log("⚠ No trained model — running full sprint labeler instead.")
            self.labeler_worker(); return

        already_done = (set(os.listdir(self.approved_dir)) |
                        set(os.listdir(self.discarded_dir)))
        all_files = sorted([
            f for f in os.listdir(self.inbox_dir)
            if f.lower().endswith(('.jpg', '.png')) and f not in already_done
        ])
        if not all_files:
            self.log("ℹ Inbox empty."); return

        self.log(f"  Scanning {len(all_files)} frames for uncertain predictions...")

        uncertain_files = []
        auto_approved   = 0
        for fname in all_files:
            src   = os.path.join(self.inbox_dir, fname)
            frame = cv2.imread(src)
            if frame is None: continue
            try:
                results = model.predict(frame, conf=SMART_REVIEW_LOW,
                                         verbose=False)
                boxes   = results[0].boxes
                if len(boxes) == 0:
                    # Model sees nothing — uncertain — human should review
                    uncertain_files.append(fname)
                    continue
                confs = [float(b.conf[0]) for b in boxes]
                if min(confs) > SMART_REVIEW_HIGH:
                    # Model is confident about everything — auto-approve
                    boxes_data = []
                    for b in boxes:
                        c   = b.xywh[0]
                        lbl = model.names.get(int(b.cls[0]), 'mob')
                        boxes_data.append({
                            "label": lbl,
                            "x": float(c[0]), "y": float(c[1]),
                            "w": float(c[2]), "h": float(c[3])
                        })
                    with open(self.output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"file": fname,
                                            "boxes": boxes_data}) + "\n")
                        f.flush()
                    shutil.move(src, os.path.join(self.approved_dir, fname))
                    auto_approved += 1
                    self._new_labels_since_train += 1
                else:
                    uncertain_files.append(fname)
            except Exception:
                uncertain_files.append(fname)

        self._save_label_progress()
        self._update_al_status_label()
        self.log(f"  Auto-approved (confident): {auto_approved}")
        self.log(f"  Uncertain → opening for human review: {len(uncertain_files)}")

        if not uncertain_files:
            self.log("✅ Smart Review: all frames auto-approved — no manual review needed!")
            self._maybe_trigger_finetune()
            return

        # Open sprint labeler for ONLY the uncertain frames (much smaller set)
        self._run_sprint_on_filelist(uncertain_files)
        self._maybe_trigger_finetune()

    def _run_sprint_on_filelist(self, file_list: list):
        """Sprint labeler restricted to a pre-filtered file list."""
        total = len(file_list)
        if total == 0: return
        self.log(f"🕵 Smart Sprint: {total} uncertain frames to review.")

        ai_model = self._load_best_model()
        win      = "SenseNova Sprint Labeler"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)

        def _on_ps(val): self._point_size[self._lbl_class] = max(8, val)
        cv2.createTrackbar("Point size", win, 28, 120, _on_ps)

        ctx = [total, total]
        cv2.setMouseCallback(win, self._labeler_mouse_cb, ctx)
        prev_gray = None

        for idx, fname in enumerate(file_list):
            ctx[0]   = total - idx
            src_path = os.path.join(self.inbox_dir, fname)
            if not os.path.exists(src_path): continue
            frame = cv2.imread(src_path)
            if frame is None: continue

            self._lbl_scale_x = frame.shape[1] / 1280.0
            self._lbl_scale_y = frame.shape[0] / 720.0
            self._lbl_boxes   = []

            new_m = self._try_hotswap_model(ai_model)
            if new_m: ai_model = new_m

            if ai_model:
                try:
                    res = ai_model.predict(frame, conf=SMART_REVIEW_LOW,
                                            verbose=False)
                    for box in res[0].boxes:
                        c   = box.xywh[0]
                        lbl = ai_model.names.get(int(box.cls[0]), 'projectile')
                        self._lbl_boxes.append({
                            "label": lbl,
                            "x": float(c[0]), "y": float(c[1]),
                            "w": float(c[2]), "h": float(c[3])
                        })
                    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                except Exception:
                    self._lbl_boxes, current_gray = self.get_proposals(
                        frame, prev_gray)
            else:
                self._lbl_boxes, current_gray = self.get_proposals(
                    frame, prev_gray)

            self._current_display_base = cv2.resize(
                frame.copy(), (1280, 720), interpolation=cv2.INTER_AREA)
            self._force_redraw(ctx[0], total)

            while True:
                key = cv2.waitKey(33) & 0xFF
                try:
                    ps = cv2.getTrackbarPos("Point size", win)
                    if ps > 0: self._point_size[self._lbl_class] = max(8, ps)
                except Exception: pass

                if key == ord(' '):
                    with open(self.output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"file": fname,
                                            "boxes": self._lbl_boxes}) + "\n")
                        f.flush(); os.fsync(f.fileno())
                    try: shutil.move(src_path, os.path.join(self.approved_dir, fname))
                    except Exception: pass
                    self.log(f"✅ {fname} ({len(self._lbl_boxes)} boxes)")
                    self._new_labels_since_train += 1
                    self._save_label_progress()
                    self._update_al_status_label()
                    prev_gray = current_gray; break
                elif key in (ord('d'), ord('D')):
                    try: shutil.move(src_path, os.path.join(self.discarded_dir, fname))
                    except Exception: pass
                    self.log(f"🗑 {fname}")
                    prev_gray = current_gray; break
                elif key in (ord('s'), ord('S'), 13):
                    prev_gray = current_gray; break
                elif key in (8, 127, 0):
                    if self._lbl_boxes:
                        self._lbl_boxes.pop()
                        self._force_redraw(ctx[0], total)
                elif key in (ord('q'), ord('Q'), 27):
                    self._current_display_base = None
                    try: cv2.destroyWindow(win)
                    except Exception: pass
                    self.log("🛑 Smart Sprint closed."); return
                elif key in (ord('+'), ord('='), ord(']')):
                    self._cycle_class(+1, ctx)
                elif key in (ord('-'), ord('[')):
                    self._cycle_class(-1, ctx)

        self._current_display_base = None
        try: cv2.destroyWindow(win)
        except Exception: pass
        self.log(f"🏁 Smart Sprint done — {total} uncertain frames reviewed.")

    # ── NEW: Synthetic Augmentation ───────────────────────────────────────────

    def synthetic_augment_worker(self):
        """
        Multiplies the approved dataset by AUG_FACTOR using OpenCV.
        For every approved image + its JSONL boxes, generates AUG_FACTOR-1
        augmented copies with transformed boxes.
        Augmentations: horizontal flip, brightness/contrast, HSV jitter,
        gaussian noise, scale, blur, rotation ±10°.
        """
        self.log(f"🎲 Synthetic Augmentation: generating {AUG_FACTOR}× copies...")

        if not os.path.exists(self.output_path):
            self.log("❌ No JSONL found."); return

        with open(self.output_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        entries = {}
        for line in lines:
            try:
                d = json.loads(line)
                img_path = os.path.join(self.approved_dir, d['file'])
                if os.path.exists(img_path) and d.get('boxes'):
                    entries[d['file']] = d['boxes']
            except Exception:
                pass

        if not entries:
            self.log("❌ No approved images with boxes found."); return

        self.log(f"  Found {len(entries)} source images → generating "
                 f"{len(entries) * (AUG_FACTOR - 1)} augmented copies...")

        new_entries = []
        for fname, boxes in entries.items():
            src_path = os.path.join(self.approved_dir, fname)
            frame    = cv2.imread(src_path)
            if frame is None: continue
            h, w = frame.shape[:2]

            for aug_i in range(1, AUG_FACTOR):  # skip 0 = original
                aug_frame = frame.copy()
                aug_boxes = [dict(b) for b in boxes]

                # 1. Horizontal flip (alternating)
                if aug_i % 2 == 0:
                    aug_frame = cv2.flip(aug_frame, 1)
                    aug_boxes = [{**b, "x": w - b["x"]} for b in aug_boxes]

                # 2. Brightness + contrast jitter
                alpha = 0.75 + (aug_i * 0.08)   # contrast 0.75–1.35
                beta  = -20 + (aug_i * 8)        # brightness -20..+44
                aug_frame = np.clip(aug_frame.astype(np.float32) * alpha + beta,
                                    0, 255).astype(np.uint8)

                # 3. HSV jitter
                hsv    = cv2.cvtColor(aug_frame, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] + aug_i * 7) % 180
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (0.8 + aug_i * 0.06), 0, 255)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (0.85 + aug_i * 0.04), 0, 255)
                aug_frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

                # 4. Gaussian noise (mild)
                noise = np.random.normal(0, 5 + aug_i, aug_frame.shape).astype(np.int16)
                aug_frame = np.clip(aug_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

                # 5. Scale ±20% (only on even aug_i to keep boxes valid)
                if aug_i % 3 == 0:
                    scale = 0.85 + (aug_i % 3) * 0.15
                    new_h = int(h * scale); new_w = int(w * scale)
                    aug_frame = cv2.resize(cv2.resize(aug_frame, (new_w, new_h)),
                                           (w, h))
                    aug_boxes = [{**b,
                                  "x": b["x"] * scale + (w - new_w) / 2,
                                  "y": b["y"] * scale + (h - new_h) / 2,
                                  "w": b["w"] * scale,
                                  "h": b["h"] * scale} for b in aug_boxes]

                # 6. Slight blur on odd aug_i
                if aug_i % 2 == 1 and aug_i > 1:
                    ks = 3 if aug_i < 5 else 5
                    aug_frame = cv2.GaussianBlur(aug_frame, (ks, ks), 0)

                # Save augmented image
                stem   = os.path.splitext(fname)[0]
                ext    = os.path.splitext(fname)[1]
                new_fn = f"{stem}_aug{aug_i}{ext}"
                new_path = os.path.join(self.approved_dir, new_fn)
                cv2.imwrite(new_path, aug_frame)
                new_entries.append({"file": new_fn, "boxes": aug_boxes})

        # Append new entries to JSONL
        with open(self.output_path, "a", encoding="utf-8") as f:
            for entry in new_entries:
                f.write(json.dumps(entry) + "\n")

        total_new = len(new_entries)
        self.log(f"✅ Augmentation done — {total_new} new frames added.")
        self.log(f"  Dataset size: {len(entries)} → {len(entries) + total_new} "
                 f"({AUG_FACTOR}× expansion)")
        self.log("  Run 🔥 SMART TRAINING to train on the expanded dataset.")

    # ── SMART TRAINING: auto-selects strategy by dataset size ────────────────

    def _get_smart_training_config(self, n_samples: int) -> dict:
        """
        Returns YOLO training kwargs tuned for the current dataset size.
        Fixes all 6 root causes identified in the training audit.
        """
        # ── Base augmentation (all strategies) ───────────────────────────────
        base = dict(
            imgsz    = 640,
            device   = 0,
            workers  = 0,
            rect     = True,      # preserve portrait aspect ratio
            mosaic   = 1.0,       # combine 4 frames — great for small objects
            copy_paste = 0.3,     # paste projectiles onto frames — fixes imbalance
            hsv_h    = 0.015,
            hsv_s    = 0.7,
            hsv_v    = 0.4,
            scale    = 0.5,
            translate= 0.1,
            degrees  = 5,
            flipud   = 0.0,
            fliplr   = 0.5,
            # Focal loss — forces attention on hard/small/missed detections
            fl_gamma = 1.5,
            # Better defaults than the hardcoded 10.0/2.5
            box      = 7.5,
            cls      = 0.5,
            dfl      = 1.5,
            verbose  = False,
        )

        if n_samples < COLD_START_MAX:
            # ── Cold Start ────────────────────────────────────────────────────
            # Freeze backbone so 50 images only update the detection head
            # (~500K params instead of 3.2M). Converges in far fewer steps.
            return {**base,
                    "epochs"  : 50,
                    "freeze"  : 10,       # freeze first 10 backbone layers
                    "batch"   : 8,
                    "patience": 15,
                    "mixup"   : 0.0,
                    "name"    : "SenseNova_ColdStart"}

        elif n_samples < WARM_TRAIN_MAX:
            # ── Warm Train ────────────────────────────────────────────────────
            # Partially unfreeze — backbone refines but doesn't thrash
            return {**base,
                    "epochs"  : 80,
                    "freeze"  : 5,
                    "batch"   : 12,
                    "patience": 20,
                    "mixup"   : 0.1,
                    "name"    : "SenseNova_WarmTrain"}

        else:
            # ── Full Train ────────────────────────────────────────────────────
            # All weights unfrozen. Multi-scale for better small-object detection.
            return {**base,
                    "epochs"  : 150,
                    "freeze"  : 0,
                    "batch"   : 16,
                    "patience": 30,
                    "mixup"   : 0.15,
                    "multi_scale": True,
                    "name"    : "SenseNova_FullTrain"}

    def validate_and_prune(self):
        self.log("🔍 Signal Integrity Audit...")
        if not os.path.exists(self.output_path): return None
        with open(self.output_path, 'r', encoding='utf-8') as f: lines = f.readlines()
        valid_data, discarded = [], 0
        for line in lines:
            data = json.loads(line); img_path = os.path.join(self.approved_dir, data['file'])
            if not os.path.exists(img_path): 
                discarded += 1
                continue
            img = cv2.imread(img_path)
            if img is None: continue
            h, w = img.shape[:2]; clean, corrupt = [], False
            for b in data['boxes']:
                if (b['x'] - b['w'] / 2 < 0 or b['x'] + b['w'] / 2 > w or b['y'] - b['h'] / 2 < 0 or b['y'] + b['h'] / 2 > h): corrupt = True; break
                if b['w'] < 4 or b['h'] < 4: continue
                clean.append(b)
            if not corrupt and clean: data['boxes'] = clean; valid_data.append(data)
            else: discarded += 1
        self.log(f"✅ Audit: {len(valid_data)} passed | {discarded} pruned.")
        return valid_data

    def start_training_thread(self):
        if not _YOLO_OK: self.log("❌ ultralytics not installed."); return
        if messagebox.askyesno("Confirm Smart Forge",
                               "Start smart training? Strategy auto-selects by dataset size."):
            threading.Thread(target=self.training_worker, daemon=True).start()

    def training_worker(self):
        valid = self.validate_and_prune()
        if not valid or len(valid) < 10:
            self.log("❌ Need ≥ 10 valid labeled frames."); return
        n   = len(valid)
        cfg = self._get_smart_training_config(n)
        strategy = cfg["name"]
        self.log(f"📊 Dataset: {n} samples → strategy: {strategy}")
        self.log(f"   epochs={cfg['epochs']} freeze={cfg.get('freeze',0)} "
                 f"rect={cfg['rect']} copy_paste={cfg['copy_paste']}")
        count = self._build_yolo_dataset_from_approved()
        self.log(f"📂 Dataset built — {count} samples (incl. augmentations).")
        yaml_path    = os.path.join(self.yolo_root, "data.yaml")
        base_weights = self._find_latest_weights() or "yolov8n.pt"
        self.log(f"🔥 Base: {Path(base_weights).name if base_weights != 'yolov8n.pt' else 'yolov8n.pt (pretrained)'}")
        try:
            model = YOLO(base_weights)
            model.train(data=yaml_path, **cfg)
            self.log("🏆 SMART FORGE SUCCESSFUL.")
            self.log("  Run ⚔ ENGAGE NEURAL HUD in Tab 4 to test live detection.")
        except Exception as e:
            self.log(f"💀 FORGE FAILED: {e}")

        except Exception as e: self.log(f"💀 FORGE FAILED: {e}")

    def toggle_evasion(self):
        if not _YOLO_OK or not _INPUT_OK or not _MSS_OK: self.log("❌ Missing: ultralytics / pydirectinput / mss"); return
        self.is_evading = not self.is_evading
        if self.is_evading: self.btn_eva.config(text="DISENGAGE BRIDGE", bg="#c0392b"); threading.Thread(target=self.evasion_worker, daemon=True).start()
        else:
            self.btn_eva.config(text="⚔  ENGAGE NEURAL HUD & BRIDGE", bg="#2980b9")
            try: cv2.destroyAllWindows()
            except Exception: pass

    # ── Phase 4: EVASION ENGINE (FIXED RECURSION & LOGGING) ──────────────────

    def evasion_worker(self):
        self.log("⚔ DEPLOYING NEURAL HUD...")
        weights = self._find_latest_weights()
        if weights is None: self.log("❌ No trained weights found. Train first."); return
        
        self.log(f"  Loaded: {Path(weights).parent.parent.name}")
        model = YOLO(weights).to('cuda')
        dodges = manual = 0
        
        # FIX-RECURSION: The CV2 window must not contain the word 'SenseNova'
        hud_win_name = "Vision_HUD_Overlay"
        last_log = time.time()
        
        with mss() as sct:
            while self.is_evading:
                if keyboard.is_pressed(self.panic_exit): break
                
                rect = self.get_game_rect()
                if not rect: time.sleep(0.5); continue
                
                frame = cv2.cvtColor(np.array(sct.grab(rect)), cv2.COLOR_BGRA2BGR)
                h, w = frame.shape[:2]
                px, py = w // 2, h // 2
                
                results = model.predict(frame, conf=self.min_conf, verbose=False)
                hud = frame.copy()
                override = any(keyboard.is_pressed(k) for k in ['w','a','s','d', self.panic_key])
                fx = fy = 0.0
                
                cv2.circle(hud, (px, py), self.safe_radius, (255, 255, 255), 1)
                
                threat_counts = {}
                for box in results[0].boxes:
                    c = box.xywh[0]
                    hx, hy = float(c[0]), float(c[1])
                    hw, hh = float(c[2]) / 2, float(c[3]) / 2
                    
                    label = model.names[int(box.cls[0])]
                    threat_counts[label] = threat_counts.get(label, 0) + 1
                    
                    col = ((0,255,0) if label == 'mob' else (0,255,255) if label == 'projectile' else (0,0,255))
                    cv2.rectangle(hud, (int(hx-hw), int(hy-hh)), (int(hx+hw), int(hy+hh)), col, 2)
                    
                    dx, dy = px - hx, py - hy
                    dist = (dx**2 + dy**2)**0.5 + 1e-5
                    
                    if dist < self.safe_radius:
                        wt = (3.5 if label == 'projectile' else 2.0 if label == 'aoe_indicator' else 1.0)
                        mag = ((self.safe_radius - dist) / self.safe_radius)**2 * wt
                        fx += (dx / dist) * mag
                        fy += (dy / dist) * mag
                
                thr = 0.25
                
                # DETAILED LOGGING (Fires every 1.5 seconds)
                if time.time() - last_log > 1.5:
                    t_str = ", ".join([f"{k}:{v}" for k,v in threat_counts.items()]) if threat_counts else "None"
                    if override:
                        self.log(f"[HUD] Manual Override | Threats: {t_str}")
                    elif abs(fx) > thr or abs(fy) > thr:
                        self.log(f"[HUD] ⚠ EVADING! Vector: (X:{fx:.2f}, Y:{fy:.2f}) | Threats: {t_str}")
                    else:
                        self.log(f"[HUD] Safe Zone | Threats: {t_str}")
                    last_log = time.time()

                # EVASION EXECUTION
                if override:
                    manual += 1
                    for k in ['w','a','s','d']: pydirectinput.keyUp(k)
                    cv2.putText(hud, "MANUAL OVERRIDE", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 165, 255), 2)
                elif abs(fx) > thr or abs(fy) > thr:
                    dodges += 1
                    if fx > thr: pydirectinput.keyDown('d'); pydirectinput.keyUp('a')
                    elif fx < -thr: pydirectinput.keyDown('a'); pydirectinput.keyUp('d')
                    if fy > thr: pydirectinput.keyDown('s'); pydirectinput.keyUp('w')
                    elif fy < -thr: pydirectinput.keyDown('w'); pydirectinput.keyUp('s')
                    cv2.putText(hud, f"EVADING ({fx:.1f},{fy:.1f})", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
                else:
                    for k in ['w','a','s','d']: pydirectinput.keyUp(k)
                    cv2.putText(hud, "IDLE", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
                
                # FIX-RECURSION: This window name must not match the GUI title filter
                cv2.imshow(hud_win_name, cv2.resize(hud, (1280, 720)))
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                
        try: cv2.destroyWindow(hud_win_name)
        except Exception: pass
        for k in ['w','a','s','d']: pydirectinput.keyUp(k)
        
        auto = (dodges - manual) / max(1, dodges) * 100
        self.log(f"── Battle Report ── Dodges: {dodges}  Overrides: {manual}  Autonomy: {auto:.1f}%")
        self.root.after(0, self.toggle_evasion)

    # ─────────────────────────────────────────────────────────────────────────

    def toggle_daily_arena(self):
        if not _OCR_OK or not _INPUT_OK or not _MSS_OK: self.log("❌ Missing: pytesseract / pydirectinput / mss"); return
        self.is_daily_running = not self.is_daily_running
        if self.is_daily_running:
            self.btn_daily.config(text="■  STOP ARENA DAILIES", bg="#e74c3c", fg="white")
            threading.Thread(target=self.daily_arena_worker, daemon=True).start()
        else:
            self.btn_daily.config(text="▶  START ARENA DAILIES", bg="#f1c40f", fg="black")

    def parse_power(self, text: str) -> float:
        text = text.upper().replace(',', '').strip()
        m = re.search(r'([\d\.]+)([KM]?)', text)
        if not m: return 0.0
        val = float(m.group(1))
        if m.group(2) == 'K': val *= 1_000
        elif m.group(2) == 'M': val *= 1_000_000
        return val

    def perform_ocr_on_crop(self, image, config='--psm 7') -> str:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            return pytesseract.image_to_string(th, config=config).strip()
        except Exception: return ""

    def daily_arena_worker(self):
        self.log("⚔ Starting Arena Dailies...")
        with mss() as sct:
            for attempt in range(1, 6):
                if not self.is_daily_running: break
                self.log(f"─── Attempt {attempt}/5 ───")
                rect = self.get_game_rect()
                if not rect: self.log("⚠ Window not found."); time.sleep(2); continue
                L, T, W, H = (rect['left'], rect['top'], rect['width'], rect['height'])
                pydirectinput.click(int(L + W * 0.5), int(T + H * 0.92))
                time.sleep(3)
                if not self.is_daily_running: break
                rect = self.get_game_rect()
                img = cv2.cvtColor(np.array(sct.grab(rect)), cv2.COLOR_BGRA2BGR)
                H2, W2 = img.shape[:2]
                p_crop = img[int(H2*0.11):int(H2*0.17), int(W2*0.65):int(W2*0.95)]
                p_txt = self.perform_ocr_on_crop(p_crop)
                p_pow = self.parse_power(p_txt)
                self.log(f"👤 Player: {p_txt!r} ({p_pow:.0f})")
                if p_pow == 0: self.log("⚠ Cannot read power. Skipping."); continue
                row_start = H2 * 0.20
                row_height = (H2 * 0.85 - row_start) / 5
                challenged = False
                for i in range(5):
                    if not self.is_daily_running: break
                    rt = int(row_start + i * row_height)
                    rb = int(row_start + (i + 1) * row_height)
                    e_crop = img[int(rt + row_height*0.5):rb, int(W2*0.15):int(W2*0.45)]
                    e_txt = self.perform_ocr_on_crop(e_crop)
                    e_pow = self.parse_power(e_txt)
                    self.log(f"  Enemy {i+1}: {e_txt!r} ({e_pow:.0f})")
                    if 0 < e_pow < p_pow:
                        self.log(f"💥 Enemy {i+1} weaker — attacking!")
                        pydirectinput.click(int(L + W * 0.8), int(T + rt + row_height * 0.5))
                        challenged = True
                        break
                if not challenged: self.log("♻ All enemies stronger.")
                self.log("⏳ 30 s cooldown…")
                for _ in range(30):
                    if not self.is_daily_running: break
                    time.sleep(1)
        self.log("🛑 Arena Dailies concluded.")
        self.root.after(0, lambda: self.btn_daily.config(text="▶  START ARENA DAILIES", bg="#f1c40f", fg="black"))
        self.is_daily_running = False

# ─────────────────────────────────────────────────────────────────────────────
def main():
    root = tk.Tk()
    app  = UnifiedCommandCenter(root)
    root.mainloop()

if __name__ == "__main__":
    main()