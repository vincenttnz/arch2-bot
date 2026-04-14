#!/usr/bin/env python3
"""
sensenova_gui.py  —  Unified Command Center v25.0
==================================================
- Removed unsupported 'fl_gamma' argument.
- Training uses subprocess with live logging.
- Evasion: only Shift for manual override; bot stops automatically.
- Full Acceleration Engine (Grounding DINO, Smart Review, Synthetic Augment).
- Capture button moved to Bot tab.
"""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import sys
import threading
import time
import atexit
import random
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

# ── Constants ─────────────────────────────────────────────────────────────────
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
SMART_REVIEW_LOW     = 0.25
SMART_REVIEW_HIGH    = 0.60
GDINO_PROMPTS = {
    'projectile':    "arrow . fireball . magic projectile . energy ball . red orb . glowing ball . attack projectile . bullet",
    'mob':           "enemy character . monster . creature . game enemy . humanoid enemy",
    'aoe_indicator': "red circle on floor . danger zone . aoe indicator . floor warning . red ring",
    'boss':          "boss enemy . large monster . boss character . giant enemy",
    'player':        "player character . archer . hero character . main character",
}
GDINO_AUTO_CONF  = 0.45
GDINO_MIN_CONF   = 0.25
COLD_START_MAX   = 100
WARM_TRAIN_MAX   = 500
AUG_FACTOR       = 8

# ─────────────────────────────────────────────────────────────────────────────
class UnifiedCommandCenter:
    POLL_MS = 250

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SenseNova v27.0 | Skill Intelligence")
        self.root.geometry("860x920")
        self.root.configure(bg="#1a1a1a")
        self.project_root = Path(__file__).resolve().parent

        # Paths
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

        # State
        self.class_map = {'player': 0, 'projectile': 1,
                          'aoe_indicator': 2, 'mob': 3, 'boss': 4}
        self.game_window_title = tk.StringVar(value="MuMuPlayer")
        self.is_capturing = False
        self.is_evading   = False
        self.safe_radius  = 400
        self.min_conf     = 0.55
        self.panic_key    = 'shift'
        self.panic_exit   = 'f10'

        # Labeler state
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

        # Active learning
        self._new_labels_since_train = self._load_label_progress()
        self._is_finetuning          = False
        self._finetune_lock          = threading.Lock()
        self._latest_weights: Optional[str] = None
        self._labeler_ai_version     = "none"

        # Log queue
        self._log_q: queue.Queue = queue.Queue()

        # Bot
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

        # Capture thread handle
        self.capture_thread = None

        # New v27 state vars
        self.reward_state_path    = os.path.join(self.base_dir, "reward_state.json")
        self.reward_detail_var    = tk.StringVar(value="")
        self.mumu_mode_var        = tk.StringVar(value="adb")
        self.skill_db_status_var  = tk.StringVar(value="Hash DB: not built")
        self.skill_clf_status_var = tk.StringVar(value="Classifier: not trained")
        self.skill_train_log_var  = tk.StringVar(value="")
        self.skill_scan_log_var   = tk.StringVar(value="")
        self.skill_epochs_var     = tk.IntVar(value=20)
        self.skill_batch_var      = tk.IntVar(value=32)
        self._skill_rec           = None
        self._scanner_running     = False

        atexit.register(self._cleanup)
        self._setup_ui()
        self._drain_log()
        self.log("🚀 System Ready")

    # --------------------------------------------------------------------------
    # Logging
    # --------------------------------------------------------------------------
    def log(self, msg: str):
        self._log_q.put(str(msg))

    def _drain_log(self):
        try:
            while True:
                msg = self._log_q.get_nowait()
                self.log_area.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {msg}\n")
                self.log_area.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(80, self._drain_log)

    def _load_label_progress(self) -> int:
        try:
            if os.path.exists(self.progress_path):
                with open(self.progress_path, "r") as f:
                    return int(json.loads(f.read()).get("new_since_train", 0))
        except Exception:
            pass
        return 0

    def _save_label_progress(self):
        try:
            with open(self.progress_path, "w") as f:
                json.dump({"new_since_train": self._new_labels_since_train}, f)
        except Exception:
            pass

    # --------------------------------------------------------------------------
    # UI Setup
    # --------------------------------------------------------------------------
    def _setup_ui(self):
        hdr = tk.Frame(self.root, bg="#111111")
        hdr.pack(fill="x", side="top")
        tk.Label(hdr, text="⚡ SENSENOVA + ARCHERO 2", bg="#111111", fg="#e0e0e0",
                 font=("Consolas", 13, "bold")).pack(side="left", padx=16, pady=10)
        pills = tk.Frame(hdr, bg="#111111")
        pills.pack(side="right", padx=12, pady=8)
        for var in (self.adb_status_var, self.bot_status_var, self.screen_var):
            tk.Label(pills, textvariable=var, bg="#1e1e1e", fg="#aaaaaa",
                     font=("Consolas", 8), padx=8, pady=3).pack(side="left", padx=4)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook", background="#1a1a1a", borderwidth=0)
        style.configure("TNotebook.Tab", background="#2a2a2a", foreground="#aaaaaa",
                        font=("Consolas", 9, "bold"), padding=[14, 6])
        style.map("TNotebook.Tab", background=[("selected", "#333333")],
                  foreground=[("selected", "#ffffff")])

        nb = ttk.Notebook(self.root)
        tabs = [
            ("Labeler", self._build_labeler_tab),
            ("Trainer", self._build_trainer_tab),
            ("Evasion", self._build_evasion_tab),
            ("Bot",     self._build_bot_tab),
            ("Skills",  self._build_skills_tab),
        ]
        for text, builder in tabs:
            tab = ttk.Frame(nb)
            nb.add(tab, text=text)
            builder(tab)
        nb.pack(expand=True, fill="both", padx=10, pady=(4, 0))

        tk.Label(self.root, text="System Log", bg="#111111", fg="#888888",
                 font=("Consolas", 8, "bold"), anchor="w").pack(fill="x", padx=10, pady=(6, 0))
        self.log_area = tk.Text(self.root, height=12, bg="#0a0a0a", fg="#00FF00",
                                font=("Consolas", 8), insertbackground="#00FF00",
                                relief="flat", bd=0)
        self.log_area.pack(fill="both", padx=10, pady=(2, 10))

    def _card(self, parent, title: str) -> tk.Frame:
        outer = tk.Frame(parent, bg="#242424", padx=2, pady=2)
        outer.pack(fill="x", padx=20, pady=8)
        tk.Label(outer, text=title, bg="#242424", fg="#888888",
                 font=("Consolas", 8), anchor="w").pack(fill="x", padx=8, pady=(4, 0))
        inner = tk.Frame(outer, bg="#1e1e1e")
        inner.pack(fill="x", padx=4, pady=4)
        return inner

    def _btn(self, parent, text: str, command, bg="#3a3a3a", fg="white",
             height=2, font_size=9) -> tk.Button:
        return tk.Button(parent, text=text, command=command, bg=bg, fg=fg,
                         activebackground=bg, activeforeground=fg,
                         font=("Consolas", font_size, "bold"), relief="flat",
                         bd=0, cursor="hand2", height=height, padx=12)

    # --------------------------------------------------------------------------
    # Tab: Labeler
    # --------------------------------------------------------------------------
    def _build_labeler_tab(self, tab):
        tk.Label(tab, text="Active Learning Data Refinement", bg="#1a1a1a", fg="#cccccc",
                 font=("Consolas", 11, "bold")).pack(pady=16)

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

        al = self._card(tab, "ACTIVE LEARNING STATUS")
        self.al_status_var = tk.StringVar(
            value=f"New labels since last train: {self._new_labels_since_train} / {AUTO_TRAIN_THRESHOLD}")
        tk.Label(al, textvariable=self.al_status_var, bg="#1e1e1e", fg="#00cc88",
                 font=("Consolas", 9)).pack(padx=12, pady=4)
        self.al_finetune_var = tk.StringVar(value="Fine-tune: idle")
        tk.Label(al, textvariable=self.al_finetune_var, bg="#1e1e1e", fg="#888888",
                 font=("Consolas", 8)).pack(padx=12, pady=(0, 4))

        acc = self._card(tab, "⚡ ACCELERATION ENGINE")
        self._btn(acc, "🚀 AUTO-LABEL (Grounding DINO)", lambda: threading.Thread(
            target=self.gdino_auto_label_worker, daemon=True).start(),
                  bg="#1a5276", font_size=9).pack(fill="x", padx=12, pady=4)
        self._btn(acc, "⚡ SMART REVIEW (Uncertainty Sampling)", lambda: threading.Thread(
            target=self.smart_review_worker, daemon=True).start(),
                  bg="#145a32", font_size=9).pack(fill="x", padx=12, pady=4)
        self._btn(acc, "🎲 SYNTHETIC AUGMENT (8x Data)", lambda: threading.Thread(
            target=self.synthetic_augment_worker, daemon=True).start(),
                  bg="#6e2f1a", font_size=9).pack(fill="x", padx=12, pady=4)

        btns = self._card(tab, "MANUAL ACTIONS")
        self._btn(btns, "🧹 DEEP CLEAN INBOX", lambda: threading.Thread(
            target=self.deep_clean_dataset, daemon=True).start(),
                  bg="#34495e").pack(fill="x", padx=12, pady=5)
        self._btn(btns, "🧠 FORCE LEARN", self.force_active_learning,
                  bg="#7030a0").pack(fill="x", padx=12, pady=5)
        self._btn(btns, "OPEN SPRINT LABELER", self.start_labeler,
                  bg="#8e44ad").pack(fill="x", padx=12, pady=5)

    # --------------------------------------------------------------------------
    # Tab: Trainer
    # --------------------------------------------------------------------------
    def _build_trainer_tab(self, tab):
        tk.Label(tab, text="Smart Neural Forge", bg="#1a1a1a", fg="#cccccc",
                 font=("Consolas", 11, "bold")).pack(pady=16)
        info = self._card(tab, "STRATEGY AUTO-SELECTOR")
        for txt, col in [
            ("< 100 images  → 🧊 Cold Start   freeze=10, 50 epochs", "#5dade2"),
            ("100–500       → 🔥 Warm Train  freeze=5,  80 epochs", "#f39c12"),
            ("500+          → 💪 Full Train   freeze=0, 150 epochs", "#2ecc71"),
        ]:
            tk.Label(info, text=txt, bg="#1e1e1e", fg=col,
                     font=("Consolas", 8), anchor="w").pack(fill="x", padx=12, pady=2)
        card = self._card(tab, "FORGE ENGINE")
        self._btn(card, "🔥 START SMART TRAINING", self.start_training_thread,
                  bg="#d35400", height=2, font_size=10).pack(fill="x", padx=12, pady=10)

    # --------------------------------------------------------------------------
    # Tab: Evasion
    # --------------------------------------------------------------------------
    def _build_evasion_tab(self, tab):
        tk.Label(tab, text="Weighted Vector Evasion Bridge", bg="#1a1a1a", fg="#cccccc",
                 font=("Consolas", 11, "bold")).pack(pady=16)
        card = self._card(tab, "EVASION ENGINE")
        win_frame = tk.Frame(card, bg="#1e1e1e")
        win_frame.pack(fill="x", padx=12, pady=5)
        tk.Label(win_frame, text="Game Window Title:", bg="#1e1e1e", fg="#888888",
                 font=("Consolas", 8)).pack(side="left")
        tk.Entry(win_frame, textvariable=self.game_window_title, width=20,
                 bg="#2a2a2a", fg="white", insertbackground="white",
                 relief="flat", font=("Consolas", 9)).pack(side="left", padx=6)
        self.btn_eva = self._btn(card, "⚔ ENGAGE EVASION", self.toggle_evasion,
                                 bg="#2980b9", height=2, font_size=10)
        self.btn_eva.pack(fill="x", padx=12, pady=10)
        tk.Label(card, text=f"Manual Override Key: [{self.panic_key.upper()}]   Kill Switch: [{self.panic_exit.upper()}]",
                 bg="#1e1e1e", fg="#666666", font=("Consolas", 8)).pack(pady=(0, 8))

    # --------------------------------------------------------------------------
    # Tab: Bot (includes Capture button)
    # --------------------------------------------------------------------------
    def _build_bot_tab(self, tab):
        tk.Label(tab, text="Archero 2 Bot & Capture", bg="#1a1a1a", fg="#cccccc",
                 font=("Consolas", 11, "bold")).pack(pady=16)
        cap_card = self._card(tab, "CAPTURE ENGINE")
        self.btn_cap = self._btn(cap_card, "▶ START CAPTURE", self.toggle_capture,
                                 bg="#27ae60", height=2, font_size=10)
        self.btn_cap.pack(fill="x", padx=12, pady=10)

        conn = self._card(tab, "ADB CONNECTION")
        row = tk.Frame(conn, bg="#1e1e1e")
        row.pack(fill="x", padx=12, pady=6)
        tk.Label(row, text="Device:", bg="#1e1e1e", fg="#888888",
                 font=("Consolas", 8)).pack(side="left")
        tk.Entry(row, textvariable=self.monitor_device_var, width=18,
                 bg="#2a2a2a", fg="white", insertbackground="white",
                 relief="flat", font=("Consolas", 9)).pack(side="left", padx=6)
        self._btn(row, "CONNECT", self.bot_connect_adb, bg="#16a085",
                  height=1, font_size=8).pack(side="left", padx=4)

        ctrl = self._card(tab, "BOT CONTROLS")
        br = tk.Frame(ctrl, bg="#1e1e1e")
        br.pack(fill="x", padx=12, pady=8)
        self._btn(br, "▶ START BOT", self.bot_start, bg="#27ae60",
                  height=1, font_size=9).pack(side="left", padx=4)
        self._btn(br, "■ STOP BOT", self.bot_stop, bg="#c0392b",
                  height=1, font_size=9).pack(side="left", padx=4)

        learn = self._card(tab, "LEARNING MODE")
        lr = tk.Frame(learn, bg="#1e1e1e")
        lr.pack(fill="x", padx=12, pady=8)
        tk.Checkbutton(lr, text="Auto-Screenshots", variable=self.log_frames_var,
                       bg="#1e1e1e", fg="#aaaaaa", selectcolor="#2a2a2a",
                       activebackground="#1e1e1e", font=("Consolas", 9)).pack(side="left")
        tk.Label(lr, text="  Interval:", bg="#1e1e1e", fg="#666666",
                 font=("Consolas", 8)).pack(side="left")
        tk.Spinbox(lr, from_=1, to=20, textvariable=self.log_freq_var, width=4,
                   bg="#2a2a2a", fg="white", buttonbackground="#2a2a2a",
                   font=("Consolas", 9), relief="flat").pack(side="left", padx=4)

        stat = self._card(tab, "LIVE STATUS")
        srow = tk.Frame(stat, bg="#1e1e1e")
        srow.pack(fill="x", padx=12, pady=6)
        for var in (self.adb_status_var, self.bot_status_var, self.screen_var, self.reward_var):
            tk.Label(srow, textvariable=var, bg="#1e1e1e", fg="#00cc88",
                     font=("Consolas", 8), padx=8, pady=2).pack(side="left")
        tk.Label(stat, textvariable=self.reward_detail_var, bg="#1e1e1e", fg="#555555",
                 font=("Consolas", 7), anchor="w").pack(fill="x", padx=12, pady=(0,4))

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # Tab: Skills — Scanner + Hash DB + Classifier Trainer
    # --------------------------------------------------------------------------
    def _build_skills_tab(self, tab):
        tk.Label(tab, text="Skill Card Intelligence", bg="#1a1a1a", fg="#cccccc",
                 font=("Consolas", 11, "bold")).pack(pady=10)

        # ── How it works ──────────────────────────────────────────────────────
        info = self._card(tab, "HOW IT WORKS")
        for txt, col in [
            ("Step 1  AUTO-SCAN skill_cards/ — AI extracts names from 78k images", "#aaaaaa"),
            ("Step 2  BUILD HASH DB — instant pHash lookup table (no GPU needed)", "#5dade2"),
            ("Step 3  TRAIN CLASSIFIER — YOLOv8 on GPU, ~15 min", "#f39c12"),
            ("Bot auto-picks highest-priority skill card every SKILL screen.", "#aaaaaa"),
        ]:
            tk.Label(info, text=txt, bg="#1e1e1e", fg=col,
                     font=("Consolas", 8), anchor="w").pack(fill="x", padx=12, pady=1)

        # ── Status ────────────────────────────────────────────────────────────
        st_card = self._card(tab, "RECOGNISER STATUS")
        tk.Label(st_card, textvariable=self.skill_db_status_var,
                 bg="#1e1e1e", fg="#00cc88", font=("Consolas", 9), anchor="w"
                 ).pack(fill="x", padx=12, pady=2)
        tk.Label(st_card, textvariable=self.skill_clf_status_var,
                 bg="#1e1e1e", fg="#f39c12", font=("Consolas", 9), anchor="w"
                 ).pack(fill="x", padx=12, pady=(0, 4))

        # ── Auto-scan ─────────────────────────────────────────────────────────
        scan_card = self._card(tab, "AUTO-SCAN  data/skill_cards/  (78 k images → templates + DB)")
        tk.Label(scan_card,
                 text="Reads every image in data/skill_cards/, identifies skill names via\n"
                      "pHash matching + Tesseract OCR, builds templates in core/templates/skills/,\n"
                      "writes skill_card_labels.json + skill_train_manifest.csv.",
                 bg="#1e1e1e", fg="#888888", font=("Consolas", 8), justify="left"
                 ).pack(fill="x", padx=12, pady=(4, 0))

        scan_cfg = tk.Frame(scan_card, bg="#1e1e1e")
        scan_cfg.pack(fill="x", padx=12, pady=(4, 0))
        tk.Label(scan_cfg, text="Max images (0=all):", bg="#1e1e1e", fg="#888888",
                 font=("Consolas", 8)).pack(side="left")
        self.scan_max_var = tk.IntVar(value=0)
        tk.Spinbox(scan_cfg, from_=0, to=100000, textvariable=self.scan_max_var,
                   width=7, bg="#2a2a2a", fg="white", buttonbackground="#2a2a2a",
                   font=("Consolas", 9), relief="flat").pack(side="left", padx=6)

        self._btn(scan_card, "🔍  AUTO-SCAN SKILL CARDS",
                  self._start_skill_scan, bg="#1a3a5c", height=1, font_size=9
                  ).pack(fill="x", padx=12, pady=(6, 2))
        tk.Label(scan_card, textvariable=self.skill_scan_log_var,
                 bg="#1e1e1e", fg="#aaaaaa", font=("Consolas", 7), anchor="w"
                 ).pack(fill="x", padx=12, pady=(0, 8))

        # ── Hash DB ───────────────────────────────────────────────────────────
        h_card = self._card(tab, "STAGE 1 — HASH DATABASE  (pHash, instant, no GPU)")
        tk.Label(h_card,
                 text="Hashes templates in core/templates/skills/ → models/skill_hash_db.pkl.\n"
                      "Runs in < 1 second. Re-run whenever templates change.",
                 bg="#1e1e1e", fg="#888888", font=("Consolas", 8), justify="left"
                 ).pack(fill="x", padx=12, pady=(4, 0))
        self._btn(h_card, "🗂  BUILD HASH DB",
                  lambda: threading.Thread(target=self._build_hash_db_worker, daemon=True).start(),
                  bg="#1a4c6e", height=1, font_size=9
                  ).pack(fill="x", padx=12, pady=(6, 8))

        # ── Classifier ────────────────────────────────────────────────────────
        c_card = self._card(tab, "STAGE 2 — YOLO CLASSIFIER  (data/skill_cards/, ~15 min GPU)")
        tk.Label(c_card,
                 text="Trains YOLOv8n-cls on data/skill_cards/ (subfolder = class).\n"
                      "Output: models/skill_classifier.pt",
                 bg="#1e1e1e", fg="#888888", font=("Consolas", 8), justify="left"
                 ).pack(fill="x", padx=12, pady=(4, 0))
        cfg = tk.Frame(c_card, bg="#1e1e1e")
        cfg.pack(fill="x", padx=12, pady=(4, 0))
        tk.Label(cfg, text="Epochs:", bg="#1e1e1e", fg="#888888",
                 font=("Consolas", 8)).pack(side="left")
        tk.Spinbox(cfg, from_=5, to=100, textvariable=self.skill_epochs_var,
                   width=5, bg="#2a2a2a", fg="white", buttonbackground="#2a2a2a",
                   font=("Consolas", 9), relief="flat").pack(side="left", padx=4)
        tk.Label(cfg, text="  Batch:", bg="#1e1e1f", fg="#888888",
                 font=("Consolas", 8)).pack(side="left")
        tk.Spinbox(cfg, from_=8, to=128, textvariable=self.skill_batch_var,
                   width=5, bg="#2a2a2a", fg="white", buttonbackground="#2a2a2a",
                   font=("Consolas", 9), relief="flat").pack(side="left", padx=4)
        self._btn(c_card, "🧠  START CLASSIFIER TRAINING",
                  self._start_skill_clf_training, bg="#6b2fa0", height=1, font_size=9
                  ).pack(fill="x", padx=12, pady=(6, 2))
        tk.Label(c_card, textvariable=self.skill_train_log_var,
                 bg="#1e1e1e", fg="#aaaaaa", font=("Consolas", 7), anchor="w"
                 ).pack(fill="x", padx=12, pady=(0, 8))

        # ── Priority table ────────────────────────────────────────────────────
        p_card = self._card(tab, "SKILL PRIORITY  (allclash.com + progameguides.com)")
        rows = [
            ("S 10", "#e74c3c", "ricochet, multishot, front_arrow, diagonal_arrow, piercing_arrow"),
            ("S  9", "#e74c3c", "charged_arrow, lightwing_arrow, instant_strike, blitz_strike, bolt,\n"
                               "               fairy_of_the_wing, venom, super_venom, toxic_meteror"),
            ("A  8", "#f39c12", "beam_strike, beam_circle"),
            ("A  7", "#f39c12", "fire_circle, energy_ring, lightning/laser/ice_spike/bomb_sprite,\n"
                               "               vine_pursuit, warriors_breath, wind_blessing, cloudfooted,\n"
                               "               soul_of_swiftness, frenzy_potion, sprite_frenzy"),
            ("B 5/4","#2ecc71", "plant_guardian, insect_lure, breath_of_wind, slow/vampiric/poison_circle,\n"
                               "               stand_strong, demon_slayer, life_conversion, restore_hp"),
            ("C  2", "#95a5a6", "atk_increase, combat_data, jump, perilous_recovery, bolt_meteor, super_freeze"),
            ("SKIP", "#555555", "error, junk, none, start"),
        ]
        for lbl, col, skills in rows:
            r = tk.Frame(p_card, bg="#1e1e1e"); r.pack(fill="x", padx=8, pady=1)
            tk.Label(r, text=lbl, bg="#1e1e1e", fg=col,
                     font=("Consolas", 8, "bold"), width=5, anchor="w").pack(side="left")
            tk.Label(r, text=skills, bg="#1e1e1e", fg="#666666",
                     font=("Consolas", 7), justify="left", anchor="w").pack(side="left", padx=4)

    # ── skill scan worker ─────────────────────────────────────────────────────
    def _start_skill_scan(self):
        if getattr(self, '_scanner_running', False):
            self.log("⚠ Scanner already running.")
            return
        threading.Thread(target=self._skill_scan_worker, daemon=True).start()

    def _skill_scan_worker(self):
        self._scanner_running = True
        def _upd(msg):
            self.log(msg)
            self.root.after(0, lambda: self.skill_scan_log_var.set(
                msg.replace("[Scanner] ","")[:80]))
        try:
            from skill_card_scanner import SkillCardScanner
        except ImportError:
            try:
                import sys as _sys
                _sys.path.insert(0, str(self.project_root))
                from skill_card_scanner import SkillCardScanner
            except ImportError:
                _upd("❌ skill_card_scanner.py not found in project root")
                self._scanner_running = False
                return
        try:
            scanner = SkillCardScanner(self.project_root)
            max_n   = getattr(self, 'scan_max_var', None)
            max_n   = max_n.get() if max_n else 0
            results = scanner.scan(progress_callback=_upd, max_images=max_n)
            stats   = results.get("stats", {})
            msg = (f"✅ Scan done: {stats.get('total_scanned',0)} images, "
                   f"{stats.get('skills_identified',0)} skills identified")
            _upd(msg)
            self.root.after(0, lambda: self.skill_db_status_var.set(
                f"Hash DB built from scan — {stats.get('skills_identified',0)} skills"))
        except Exception as e:
            _upd(f"❌ Scan error: {e}")
        finally:
            self._scanner_running = False

    # ── hash DB builder ───────────────────────────────────────────────────────
    def _build_hash_db_worker(self):
        self.log("🗂 Building skill hash DB …")
        def _upd(msg):
            self.root.after(0, lambda: self.skill_db_status_var.set(msg))
        try:
            _rc = None
            try:
                from core.skill_recogniser import SkillRecogniser as _RC; _rc = _RC
            except ImportError:
                pass
            if _rc is None:
                import sys as _sys; _sys.path.insert(0, str(self.project_root))
                try:
                    from skill_recogniser import SkillRecogniser as _RC; _rc = _RC
                except ImportError:
                    pass
            if _rc is None:
                _upd("Hash DB: ❌ skill_recogniser.py not found")
                self.log("❌ Copy skill_recogniser.py to project root or core/")
                return
            rec = _rc(self.project_root)
            n   = rec.build_hash_db()
            st  = rec.status()
            msg = f"Hash DB: {st['db_skills']} skills, {st['db_hashes']} hashes ✓"
            _upd(msg); self.log(f"✅ {msg}")
            self._skill_rec = rec
        except Exception as e:
            msg = f"Hash DB error: {e}"
            _upd(msg); self.log(f"❌ {msg}")

    # ── classifier trainer ────────────────────────────────────────────────────
    def _start_skill_clf_training(self):
        if not _YOLO_OK:
            self.log("❌ ultralytics not installed."); return
        with self._finetune_lock:
            if self._is_finetuning:
                self.log("⚠ Training already in progress."); return
        ep = self.skill_epochs_var.get()
        bt = self.skill_batch_var.get()
        if messagebox.askyesno("Train Skill Classifier",
                               f"Train YOLOv8 classifier on data/skill_cards/?\n\n"
                               f"  Epochs={ep}   Batch={bt}\n  ~15 min on RTX 3060"):
            threading.Thread(target=self._skill_clf_worker, daemon=True).start()

    def _skill_clf_worker(self):
        with self._finetune_lock:
            if self._is_finetuning:
                self.log("⚠ Already training."); return
            self._is_finetuning = True
        def _upd(msg):
            self.log(msg)
            self.root.after(0, lambda: self.skill_train_log_var.set(
                msg.replace("[Scanner] ","")[:80]))
        try:
            from skill_card_scanner import SkillCardScanner
            scanner = SkillCardScanner(self.project_root)
            ok = scanner.train_classifier(
                self.project_root,
                epochs=self.skill_epochs_var.get(),
                batch=self.skill_batch_var.get(),
                progress_callback=_upd,
            )
            if ok:
                self.root.after(0, lambda: self.skill_clf_status_var.set(
                    "Classifier: trained ✓  (models/skill_classifier.pt)"))
                self._build_hash_db_worker()
        except Exception as e:
            _upd(f"❌ {e}")
        finally:
            self._is_finetuning = False

    # Capture Logic
    # --------------------------------------------------------------------------
    def get_game_rect(self):
        if not _MSS_OK:
            return None
        try:
            title = self.game_window_title.get().strip()
            if not title:
                return None
            windows = gw.getWindowsWithTitle(title)
            for w in windows:
                if "Command Center" not in w.title and "Vision_HUD_Overlay" not in w.title:
                    if w.isMinimized:
                        w.restore()
                    return {"top": w.top, "left": w.left, "width": w.width, "height": w.height}
            return None
        except Exception:
            return None

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
            if bh == 0:
                continue
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

    def toggle_capture(self):
        if not _MSS_OK:
            self.log("❌ mss/pygetwindow not installed.")
            return
        self.is_capturing = not self.is_capturing
        if self.is_capturing:
            self.btn_cap.config(text="■ STOP CAPTURE", bg="#c0392b")
            self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
            self.capture_thread.start()
        else:
            self.btn_cap.config(text="▶ START CAPTURE", bg="#27ae60")

    def capture_loop(self):
        self.log("📡 Capture started. Looking for game window...")
        with mss() as sct:
            prev_gray = None
            while self.is_capturing:
                rect = self.get_game_rect()
                if not rect:
                    time.sleep(1.5)
                    continue
                img = np.array(sct.grab(rect))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                proposals, prev_gray = self.get_proposals(frame, prev_gray)
                if proposals:
                    ts = int(time.time() * 1000)
                    fname = f"frame_{ts}.jpg"
                    cv2.imwrite(os.path.join(self.inbox_dir, fname), frame)
                    self.log(f"📸 {fname} | {len(proposals)} hazards")
                time.sleep(0.1)

    # --------------------------------------------------------------------------
    # Labeler Worker (full)
    # --------------------------------------------------------------------------
    def start_labeler(self):
        if self.is_capturing or self.is_evading:
            messagebox.showwarning("Conflict", "Stop Capture/Evasion before labeling.")
            return
        threading.Thread(target=self.labeler_worker, daemon=True).start()

    def labeler_worker(self):
        already_approved = set(os.listdir(self.approved_dir))
        already_discarded = set(os.listdir(self.discarded_dir))
        files = sorted([f for f in os.listdir(self.inbox_dir)
                        if f.lower().endswith(('.jpg', '.png')) and f not in already_approved and f not in already_discarded])
        total = len(files)
        if total == 0:
            self.log("ℹ Inbox empty (or all files already processed).")
            return
        self.log(f"🕵 Sprint: {total} unprocessed frames.")

        ai_model = self._load_best_model()
        if ai_model:
            self.log(f"🤖 Active Learning: AI {self._labeler_ai_version} pre-labeling.")
        else:
            self.log("ℹ No trained model found — using heuristic proposals.")

        win_name = "SenseNova Sprint Labeler"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1280, 720)

        def _on_point_size(val):
            self._point_size[self._lbl_class] = max(8, val)
        cv2.createTrackbar("Point size", win_name, 28, 120, _on_point_size)

        progress_ctx = [total, total]
        cv2.setMouseCallback(win_name, self._labeler_mouse_cb, progress_ctx)
        prev_gray = None

        for idx, fname in enumerate(files):
            remaining = total - idx
            progress_ctx[0] = remaining
            src_path = os.path.join(self.inbox_dir, fname)
            if not os.path.exists(src_path):
                continue
            frame = cv2.imread(src_path)
            if frame is None:
                continue

            self._lbl_scale_x = frame.shape[1] / 1280.0
            self._lbl_scale_y = frame.shape[0] / 720.0
            self._lbl_boxes = []

            new_model = self._try_hotswap_model(ai_model)
            if new_model is not None:
                ai_model = new_model
                self.log(f"🔄 Model hot-swapped → {self._labeler_ai_version}")

            if ai_model:
                try:
                    results = ai_model.predict(frame, conf=0.35, verbose=False)
                    for box in results[0].boxes:
                        c = box.xywh[0]
                        lbl = ai_model.names.get(int(box.cls[0]), 'projectile')
                        self._lbl_boxes.append({"label": lbl, "x": float(c[0]), "y": float(c[1]),
                                                "w": float(c[2]), "h": float(c[3])})
                    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                except Exception:
                    self._lbl_boxes, current_gray = self.get_proposals(frame, prev_gray)
            else:
                self._lbl_boxes, current_gray = self.get_proposals(frame, prev_gray)

            self._current_display_base = cv2.resize(frame.copy(), (1280, 720), interpolation=cv2.INTER_AREA)
            self._force_redraw(remaining, total)

            while True:
                key = cv2.waitKey(33) & 0xFF
                try:
                    raw_ps = cv2.getTrackbarPos("Point size", win_name)
                    if raw_ps > 0:
                        self._point_size[self._lbl_class] = max(8, raw_ps)
                except Exception:
                    pass

                if key == ord(' '):
                    with open(self.output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"file": fname, "boxes": self._lbl_boxes}) + "\n")
                        f.flush()
                    try:
                        shutil.move(src_path, os.path.join(self.approved_dir, fname))
                    except Exception as e:
                        self.log(f"⚠ Move error: {e}")
                    self.log(f"✅ Approved: {fname} ({len(self._lbl_boxes)} boxes)")
                    self._new_labels_since_train += 1
                    self._save_label_progress()
                    self._update_al_status_label()
                    self._maybe_trigger_finetune()
                    prev_gray = current_gray
                    break
                elif key in (ord('d'), ord('D')):
                    try:
                        shutil.move(src_path, os.path.join(self.discarded_dir, fname))
                    except Exception as e:
                        self.log(f"⚠ Move error: {e}")
                    self.log(f"🗑 Discarded: {fname}")
                    prev_gray = current_gray
                    break
                elif key in (ord('s'), ord('S'), 13):
                    prev_gray = current_gray
                    break
                elif key in (8, 127, 0):
                    if self._lbl_boxes:
                        self._lbl_boxes.pop()
                        self._force_redraw(remaining, total)
                elif key in (ord('q'), ord('Q'), 27):
                    self._current_display_base = None
                    try:
                        cv2.destroyWindow(win_name)
                    except Exception:
                        pass
                    self.log("🛑 Labeler closed by user.")
                    return
                elif key in (ord('+'), ord('='), ord(']')):
                    self._cycle_class(+1, progress_ctx)
                elif key in (ord('-'), ord('[')):
                    self._cycle_class(-1, progress_ctx)

        self._current_display_base = None
        try:
            cv2.destroyWindow(win_name)
        except Exception:
            pass
        self.log(f"🏁 Sprint concluded — {total} frames processed.")

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
            sx = int((p['x'] - p['w'] / 2) / self._lbl_scale_x)
            sy = int((p['y'] - p['h'] / 2) / self._lbl_scale_y)
            sw = int(p['w'] / self._lbl_scale_x)
            sh = int(p['h'] / self._lbl_scale_y)
            col = LABEL_COLORS.get(p['label'], (255, 255, 255))
            cv2.rectangle(out, (sx, sy), (sx + sw, sy + sh), col, 2)
            cv2.putText(out, p['label'], (sx, max(12, sy - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
        return out

    def _force_redraw(self, remaining: int = 0, total: int = 0):
        if self._current_display_base is None:
            return
        img = self._draw_boxes(self._current_display_base)
        cv2.imshow("SenseNova Sprint Labeler", self._overlay_text(img, remaining, total))

    def _labeler_mouse_cb(self, event, x, y, flags, param):
        if self._current_display_base is None:
            return
        nx, ny = int(x * self._lbl_scale_x), int(y * self._lbl_scale_y)
        rem = param[0] if param else 0
        tot = param[1] if param else 0

        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                self._lbl_deleting_region = True
                self._lbl_del_ix, self._lbl_del_iy = nx, ny
            else:
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
                cv2.imshow("SenseNova Sprint Labeler", self._overlay_text(preview, rem, tot))
            elif self._lbl_deleting_region:
                preview = self._draw_boxes(self._current_display_base)
                ox = int(self._lbl_del_ix / self._lbl_scale_x)
                oy = int(self._lbl_del_iy / self._lbl_scale_y)
                cv2.rectangle(preview, (ox, oy), (x, y), (0, 0, 255), 2)
                cv2.imshow("SenseNova Sprint Labeler", self._overlay_text(preview, rem, tot))

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
                    del self._lbl_boxes[i]
                    self._force_redraw(rem, tot)
                    break

        elif event == cv2.EVENT_MOUSEWHEEL:
            self._cycle_class(1 if flags > 0 else -1, param)

    def _cycle_class(self, delta: int, param):
        self._lbl_class_idx = (self._lbl_class_idx + delta) % len(LABEL_CLASSES)
        self._lbl_class = LABEL_CLASSES[self._lbl_class_idx]
        rem = param[0] if param else 0
        tot = param[1] if param else 0
        self._force_redraw(rem, tot)

    # --------------------------------------------------------------------------
    # Active Learning Helpers
    # --------------------------------------------------------------------------
    def _update_al_status_label(self):
        def _upd():
            self.al_status_var.set(f"New labels since last train: {self._new_labels_since_train} / {AUTO_TRAIN_THRESHOLD}")
        self.root.after(0, _upd)

    def _find_latest_weights(self) -> Optional[str]:
        runs_dir = self.project_root / "runs" / "detect"
        if not runs_dir.exists():
            return None
        candidates = sorted(runs_dir.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        return str(candidates[0]) if candidates else None

    def _load_best_model(self):
        if not _YOLO_OK:
            return None
        weights = self._find_latest_weights()
        if weights is None:
            self._labeler_ai_version = "none"
            return None
        try:
            m = YOLO(weights)
            self._labeler_ai_version = "custom"
            self._latest_weights = weights
            return m
        except Exception as e:
            self.log(f"⚠ Could not load model: {e}")
            return None

    def _try_hotswap_model(self, current_model):
        if not _YOLO_OK:
            return None
        newest = self._find_latest_weights()
        if newest and newest != self._latest_weights:
            try:
                m = YOLO(newest)
                self._labeler_ai_version = "new"
                self._latest_weights = newest
                return m
            except Exception:
                pass
        return None

    def _maybe_trigger_finetune(self):
        if self._new_labels_since_train < AUTO_TRAIN_THRESHOLD:
            return
        with self._finetune_lock:
            if self._is_finetuning:
                return
            self._is_finetuning = True
        self.log(f"🔁 Auto fine-tune triggered ({self._new_labels_since_train} new labels).")
        self._new_labels_since_train = 0
        self._save_label_progress()
        self._update_al_status_label()
        threading.Thread(target=self._incremental_finetune, daemon=True).start()

    def _incremental_finetune(self):
        def _set_status(msg):
            self.log(msg)
            self.root.after(0, lambda: self.al_finetune_var.set(msg))
        _set_status("Fine-tune: building dataset...")
        if not _YOLO_OK:
            _set_status("Fine-tune: ultralytics missing — aborted.")
            self._is_finetuning = False
            return
        valid = self._build_yolo_dataset_from_approved()
        if valid < 20:
            _set_status(f"Fine-tune: only {valid} valid samples — skipped.")
            self._is_finetuning = False
            return
        self.log(f"Fine-tune: built YOLO dataset with {valid} labeled images.")
        n_samples = len(list(Path(self.yolo_root).rglob("*.jpg")))
        self.log(f"Fine-tune: total images in dataset folder = {n_samples}")
        epochs = 10 if n_samples < 100 else 12 if n_samples < 300 else 15
        version = int(time.time())
        run_name = f"SenseNova_Survival_ft_{version}"
        yaml_path = os.path.join(self.yolo_root, "data.yaml")
        _set_status(f"Fine-tune: training {epochs} epochs → {run_name} …")
        try:
            base_weights = self._find_latest_weights() or "yolov8n.pt"
            model = YOLO(base_weights)
            model.train(data=yaml_path, epochs=epochs, imgsz=640, device=0,
                        batch=8, workers=2, patience=10, box=10.0, cls=2.5,
                        name=run_name, verbose=False)
            new_weights = str(self.project_root / "runs" / "detect" / run_name / "weights" / "best.pt")
            if os.path.exists(new_weights):
                self._latest_weights = new_weights
                _set_status(f"Fine-tune: ✅ done → {run_name}")
            else:
                _set_status("Fine-tune: training completed (weights not found).")
        except Exception as e:
            _set_status(f"Fine-tune: ❌ {e}")
        finally:
            self._is_finetuning = False

    def _build_yolo_dataset_from_approved(self) -> int:
        self.log("Building YOLO dataset from approved images...")
        for split in ['train', 'val']:
            for sub in ['images', 'labels']:
                p = os.path.join(self.yolo_root, split, sub)
                if os.path.exists(p):
                    shutil.rmtree(p)
                os.makedirs(p, exist_ok=True)
        if not os.path.exists(self.output_path):
            return 0
        with open(self.output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        entries = []
        for line in lines:
            try:
                d = json.loads(line)
                img_p = os.path.join(self.approved_dir, d['file'])
                if os.path.exists(img_p) and d.get('boxes'):
                    entries.append(d)
            except Exception:
                pass
        self.log(f"   Found {len(entries)} approved images with boxes")
        for i, data in enumerate(entries):
            split = 'train' if i < int(len(entries) * 0.8) else 'val'
            img_src = os.path.join(self.approved_dir, data['file'])
            if not os.path.exists(img_src):
                continue
            img = cv2.imread(img_src)
            if img is None:
                continue
            h, w = img.shape[:2]
            lbl_name = os.path.splitext(data['file'])[0] + ".txt"
            lbl_path = os.path.join(self.yolo_root, split, 'labels', lbl_name)
            with open(lbl_path, 'w') as f:
                for b in data['boxes']:
                    cid = self.class_map.get(b.get('label', 'mob'), 3)
                    f.write(f"{cid} {b['x']/w:.6f} {b['y']/h:.6f} {b['w']/w:.6f} {b['h']/h:.6f}\n")
            shutil.copy(img_src, os.path.join(self.yolo_root, split, 'images', data['file']))
        yaml_path = os.path.join(self.yolo_root, 'data.yaml')
        with open(yaml_path, 'w') as f:
            f.write(f"path: {self.yolo_root}\ntrain: train/images\nval: val/images\nnc: 5\nnames: ['player', 'projectile', 'aoe_indicator', 'mob', 'boss']\n")
        return len(entries)

    def force_active_learning(self):
        with self._finetune_lock:
            if self._is_finetuning:
                self.log("⚠ Learning already in progress.")
                return
        self.log("🧠 Manual Force-Learn Triggered")
        self._new_labels_since_train = 0
        self._save_label_progress()
        self._update_al_status_label()
        threading.Thread(target=self._incremental_finetune, daemon=True).start()

    def deep_clean_dataset(self):
        self.log("🧹 Deep cleaning inbox...")
        if not os.path.exists(self.output_path):
            self.log("❌ No JSONL found.")
            return
        with open(self.output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        cleaned, fd, bd = [], 0, 0
        for line in lines:
            data = json.loads(line)
            img_path = os.path.join(self.inbox_dir, data['file'])
            if not os.path.exists(img_path):
                cleaned.append(line)
                continue
            frame = cv2.imread(img_path)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = frame.shape[:2]
            valid = []
            for box in data.get('boxes', []):
                x1 = max(0, int(box['x'] - box['w'] / 2))
                y1 = max(0, int(box['y'] - box['h'] / 2))
                x2 = min(w, int(box['x'] + box['w'] / 2))
                y2 = min(h, int(box['y'] + box['h'] / 2))
                if x2 <= x1 or y2 <= y1:
                    bd += 1
                    continue
                crop = gray[y1:y2, x1:x2]
                _, std = cv2.meanStdDev(crop)
                contrast, peak = float(std[0][0]), int(np.max(crop)) if crop.size else 0
                bad = (box['label'] == 'projectile' and (contrast < 15.0 or peak < 160)) or (box['label'] == 'mob' and contrast < 20.0)
                if bad:
                    bd += 1
                else:
                    valid.append(box)
            if valid:
                data['boxes'] = valid
                cleaned.append(json.dumps(data) + "\n")
            else:
                try:
                    os.remove(img_path)
                    fd += 1
                except Exception:
                    pass
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.writelines(cleaned)
        self.log(f"✨ Clean done — {bd} noise boxes, {fd} empty frames purged.")

    # --------------------------------------------------------------------------
    # Grounding DINO Auto-Labeler
    # --------------------------------------------------------------------------
    def gdino_auto_label_worker(self):
        self.log("🚀 Grounding DINO Auto-Labeler starting...")
        try:
            from groundingdino.util.inference import load_model, predict
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
            self.log("ℹ Inbox empty — nothing to auto-label.")
            return
        self.log(f"  Processing {total} frames...")

        auto_approved = uncertain = discarded_count = 0

        if _gdino_ok:
            try:
                gdino_cfg = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
                gdino_ckpt = "groundingdino_swint_ogc.pth"
                gdino_model = load_model(gdino_cfg, gdino_ckpt)

                for i, fname in enumerate(inbox_files):
                    src = os.path.join(self.inbox_dir, fname)
                    frame = cv2.imread(src)
                    if frame is None:
                        continue
                    h, w = frame.shape[:2]
                    boxes_out = []

                    for cls_name, prompt in GDINO_PROMPTS.items():
                        try:
                            from PIL import Image as PILImage
                            pil_img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            boxes, logits, _ = predict(
                                model=gdino_model,
                                image=pil_img,
                                caption=prompt,
                                box_threshold=GDINO_MIN_CONF,
                                text_threshold=GDINO_MIN_CONF,
                            )
                            for box, conf in zip(boxes, logits):
                                cx = float(box[0]) * w
                                cy = float(box[1]) * h
                                bw = float(box[2]) * w
                                bh = float(box[3]) * h
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
                    clean = [{k: v for k, v in b.items() if k != "_conf"} for b in boxes_out]

                    if min_conf >= GDINO_AUTO_CONF:
                        with open(self.output_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"file": fname, "boxes": clean}) + "\n")
                            f.flush()
                        shutil.move(src, os.path.join(self.approved_dir, fname))
                        self._new_labels_since_train += 1
                        self._save_label_progress()
                        auto_approved += 1
                    else:
                        uncertain += 1

                    if (i + 1) % 50 == 0:
                        self.log(f"  {i+1}/{total} — approved:{auto_approved} uncertain:{uncertain}")
            except Exception as e:
                self.log(f"❌ GDINO error: {e}")
                _gdino_ok = False

        if not _gdino_ok:
            prev_gray = None
            for i, fname in enumerate(inbox_files):
                src = os.path.join(self.inbox_dir, fname)
                frame = cv2.imread(src)
                if frame is None:
                    continue
                proposals, prev_gray = self.get_proposals(frame, prev_gray)
                if proposals:
                    with open(self.output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"file": fname, "boxes": proposals}) + "\n")
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
        self.log(f"✅ Auto-label done — approved:{auto_approved} uncertain:{uncertain} discarded:{discarded_count}")
        if uncertain > 0:
            self.log(f"  Run ⚡ SMART REVIEW to process {uncertain} uncertain frames.")

    # --------------------------------------------------------------------------
    # Smart Review
    # --------------------------------------------------------------------------
    def _predict_with_tta(self, model, frame):
        try:
            res = model(frame, conf=0.25, verbose=False)[0].boxes
            if len(res) == 0:
                return 0.0
            base_confs = [float(b.conf[0]) for b in res]
            flipped = cv2.flip(frame, 1)
            res_flip = model(flipped, conf=0.25, verbose=False)[0].boxes
            if len(res) != len(res_flip):
                return 0.0
            flip_confs = [float(b.conf[0]) for b in res_flip]
            diff = np.mean([abs(a - b) for a, b in zip(base_confs, flip_confs)])
            return 1.0 - min(1.0, diff / 0.3)
        except:
            return 0.0

    def smart_review_worker(self):
        self.log("⚡ Smart Review: loading model for uncertainty sampling...")
        model = self._load_best_model()
        if model is None:
            self.log("⚠ No trained model — running full sprint labeler instead.")
            self.labeler_worker()
            return

        already_done = (set(os.listdir(self.approved_dir)) |
                        set(os.listdir(self.discarded_dir)))
        all_files = sorted([
            f for f in os.listdir(self.inbox_dir)
            if f.lower().endswith(('.jpg', '.png')) and f not in already_done
        ])
        if not all_files:
            self.log("ℹ Inbox empty.")
            return

        self.log(f"  Scanning {len(all_files)} frames for uncertain predictions...")
        uncertain_files = []
        auto_approved = 0

        for i, fname in enumerate(all_files):
            src = os.path.join(self.inbox_dir, fname)
            frame = cv2.imread(src)
            if frame is None:
                continue
            try:
                results = model.predict(frame, conf=SMART_REVIEW_LOW, verbose=False)
                boxes = results[0].boxes
                if len(boxes) == 0:
                    uncertain_files.append(fname)
                    continue
                confs = [float(b.conf[0]) for b in boxes]
                consistency = self._predict_with_tta(model, frame)
                if consistency > 0.8 and min(confs) > SMART_REVIEW_HIGH:
                    boxes_data = []
                    for b in boxes:
                        c = b.xywh[0]
                        lbl = model.names.get(int(b.cls[0]), 'mob')
                        boxes_data.append({
                            "label": lbl,
                            "x": float(c[0]), "y": float(c[1]),
                            "w": float(c[2]), "h": float(c[3])
                        })
                    with open(self.output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"file": fname, "boxes": boxes_data}) + "\n")
                        f.flush()
                    shutil.move(src, os.path.join(self.approved_dir, fname))
                    auto_approved += 1
                    self._new_labels_since_train += 1
                else:
                    uncertain_files.append(fname)
            except Exception as e:
                self.log(f"Error processing {fname}: {e}")
                uncertain_files.append(fname)

            if (i+1) % 50 == 0:
                self.log(f"  Processed {i+1}/{len(all_files)} frames. Auto-approved: {auto_approved}")

        self._save_label_progress()
        self._update_al_status_label()
        self.log(f"Smart Review: {auto_approved} auto-approved, {len(uncertain_files)} uncertain.")
        if not uncertain_files:
            self.log("✅ All frames auto-approved — no manual review needed!")
            self._maybe_trigger_finetune()
            return

        self.log(f"Starting manual review for {len(uncertain_files)} uncertain frames...")
        self._run_sprint_on_filelist(uncertain_files)
        self._maybe_trigger_finetune()

    def _run_sprint_on_filelist(self, file_list: list):
        total = len(file_list)
        if total == 0:
            return
        self.log(f"🕵 Smart Sprint: {total} uncertain frames to review.")
        ai_model = self._load_best_model()
        win = "SenseNova Sprint Labeler"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)

        def _on_ps(val):
            self._point_size[self._lbl_class] = max(8, val)
        cv2.createTrackbar("Point size", win, 28, 120, _on_ps)

        ctx = [total, total]
        cv2.setMouseCallback(win, self._labeler_mouse_cb, ctx)
        prev_gray = None

        for idx, fname in enumerate(file_list):
            ctx[0] = total - idx
            src_path = os.path.join(self.inbox_dir, fname)
            if not os.path.exists(src_path):
                continue
            frame = cv2.imread(src_path)
            if frame is None:
                continue

            self._lbl_scale_x = frame.shape[1] / 1280.0
            self._lbl_scale_y = frame.shape[0] / 720.0
            self._lbl_boxes = []

            new_m = self._try_hotswap_model(ai_model)
            if new_m:
                ai_model = new_m

            if ai_model:
                try:
                    res = ai_model.predict(frame, conf=SMART_REVIEW_LOW, verbose=False)
                    for box in res[0].boxes:
                        c = box.xywh[0]
                        lbl = ai_model.names.get(int(box.cls[0]), 'projectile')
                        self._lbl_boxes.append({
                            "label": lbl,
                            "x": float(c[0]), "y": float(c[1]),
                            "w": float(c[2]), "h": float(c[3])
                        })
                    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                except Exception:
                    self._lbl_boxes, current_gray = self.get_proposals(frame, prev_gray)
            else:
                self._lbl_boxes, current_gray = self.get_proposals(frame, prev_gray)

            self._current_display_base = cv2.resize(frame.copy(), (1280, 720), interpolation=cv2.INTER_AREA)
            self._force_redraw(ctx[0], total)

            while True:
                key = cv2.waitKey(33) & 0xFF
                try:
                    ps = cv2.getTrackbarPos("Point size", win)
                    if ps > 0:
                        self._point_size[self._lbl_class] = max(8, ps)
                except Exception:
                    pass

                if key == ord(' '):
                    with open(self.output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"file": fname, "boxes": self._lbl_boxes}) + "\n")
                        f.flush()
                    try:
                        shutil.move(src_path, os.path.join(self.approved_dir, fname))
                    except Exception:
                        pass
                    self.log(f"✅ Approved: {fname} ({len(self._lbl_boxes)} boxes)")
                    self._new_labels_since_train += 1
                    self._save_label_progress()
                    self._update_al_status_label()
                    prev_gray = current_gray
                    break
                elif key in (ord('d'), ord('D')):
                    try:
                        shutil.move(src_path, os.path.join(self.discarded_dir, fname))
                    except Exception:
                        pass
                    self.log(f"🗑 Discarded: {fname}")
                    prev_gray = current_gray
                    break
                elif key in (ord('s'), ord('S'), 13):
                    prev_gray = current_gray
                    break
                elif key in (8, 127, 0):
                    if self._lbl_boxes:
                        self._lbl_boxes.pop()
                        self._force_redraw(ctx[0], total)
                elif key in (ord('q'), ord('Q'), 27):
                    self._current_display_base = None
                    try:
                        cv2.destroyWindow(win)
                    except Exception:
                        pass
                    self.log("🛑 Smart Sprint closed.")
                    return
                elif key in (ord('+'), ord('='), ord(']')):
                    self._cycle_class(+1, ctx)
                elif key in (ord('-'), ord('[')):
                    self._cycle_class(-1, ctx)

        self._current_display_base = None
        try:
            cv2.destroyWindow(win)
        except Exception:
            pass
        self.log(f"🏁 Smart Sprint done — {total} uncertain frames reviewed.")

    # --------------------------------------------------------------------------
    # Synthetic Augmentation
    # --------------------------------------------------------------------------
    def synthetic_augment_worker(self):
        self.log(f"🎲 Synthetic Augmentation: generating {AUG_FACTOR}× copies...")
        if not os.path.exists(self.output_path):
            self.log("❌ No JSONL found.")
            return

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
            self.log("❌ No approved images with boxes found.")
            return

        self.log(f"  Found {len(entries)} source images → generating {len(entries) * (AUG_FACTOR - 1)} augmented copies...")
        new_entries = []

        for fname, boxes in entries.items():
            src_path = os.path.join(self.approved_dir, fname)
            frame = cv2.imread(src_path)
            if frame is None:
                continue
            h, w = frame.shape[:2]

            for aug_i in range(1, AUG_FACTOR):
                aug_frame = frame.copy()
                aug_boxes = [dict(b) for b in boxes]

                # Horizontal flip (alternating)
                if aug_i % 2 == 0:
                    aug_frame = cv2.flip(aug_frame, 1)
                    aug_boxes = [{**b, "x": w - b["x"]} for b in aug_boxes]

                # Brightness + contrast jitter
                alpha = 0.75 + (aug_i * 0.08)
                beta = -20 + (aug_i * 8)
                aug_frame = np.clip(aug_frame.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

                # HSV jitter
                hsv = cv2.cvtColor(aug_frame, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] + aug_i * 7) % 180
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (0.8 + aug_i * 0.06), 0, 255)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (0.85 + aug_i * 0.04), 0, 255)
                aug_frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

                # Gaussian noise
                noise = np.random.normal(0, 5 + aug_i, aug_frame.shape).astype(np.int16)
                aug_frame = np.clip(aug_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

                # Scale ±20%
                if aug_i % 3 == 0:
                    scale = 0.85 + (aug_i % 3) * 0.15
                    new_h = int(h * scale)
                    new_w = int(w * scale)
                    aug_frame = cv2.resize(cv2.resize(aug_frame, (new_w, new_h)), (w, h))
                    aug_boxes = [{**b,
                                  "x": b["x"] * scale + (w - new_w) / 2,
                                  "y": b["y"] * scale + (h - new_h) / 2,
                                  "w": b["w"] * scale,
                                  "h": b["h"] * scale} for b in aug_boxes]

                # Slight blur
                if aug_i % 2 == 1 and aug_i > 1:
                    ks = 3 if aug_i < 5 else 5
                    aug_frame = cv2.GaussianBlur(aug_frame, (ks, ks), 0)

                stem = os.path.splitext(fname)[0]
                ext = os.path.splitext(fname)[1]
                new_fn = f"{stem}_aug{aug_i}{ext}"
                new_path = os.path.join(self.approved_dir, new_fn)
                cv2.imwrite(new_path, aug_frame)
                new_entries.append({"file": new_fn, "boxes": aug_boxes})

        with open(self.output_path, "a", encoding="utf-8") as f:
            for entry in new_entries:
                f.write(json.dumps(entry) + "\n")

        total_new = len(new_entries)
        self.log(f"✅ Augmentation done — {total_new} new frames added.")
        self.log(f"  Dataset size: {len(entries)} → {len(entries) + total_new} ({AUG_FACTOR}× expansion)")
        self.log("  Run 🔥 SMART TRAINING to train on the expanded dataset.")

    # --------------------------------------------------------------------------
    # Training (fixed: removed fl_gamma, uses subprocess with live output)
    # --------------------------------------------------------------------------
    def start_training_thread(self):
        if not _YOLO_OK:
            self.log("❌ ultralytics not installed.")
            return
        if messagebox.askyesno("Confirm Smart Forge", "Start smart training? Strategy auto-selects by dataset size."):
            threading.Thread(target=self.training_worker, daemon=True).start()

    def training_worker(self):
        self.log("Smart training worker started.")
        valid = self.validate_and_prune()
        if not valid or len(valid) < 10:
            self.log(f"❌ Need ≥10 valid labeled frames. Found: {len(valid) if valid else 0}")
            return
        n = len(valid)
        cfg = self._get_smart_training_config(n)
        strategy = cfg["name"]
        self.log(f"📊 Dataset size: {n} → strategy: {strategy}")
        self.log(f"📈 Training config: epochs={cfg['epochs']}, freeze={cfg.get('freeze',0)}, batch={cfg['batch']}")

        count = self._build_yolo_dataset_from_approved()
        self.log(f"📂 YOLO dataset built: {count} labeled images.")
        if count < 10:
            self.log("❌ Still not enough images. Aborting.")
            return

        yaml_path = os.path.join(self.yolo_root, "data.yaml")
        if not os.path.exists(yaml_path):
            self.log("❌ data.yaml not found. Cannot train.")
            return

        base_weights = self._find_latest_weights() or "yolov8n.pt"
        self.log(f"🔥 Base weights: {base_weights}")

        import subprocess
        script_path = os.path.join(self.base_dir, "temp_train.py")
        # Convert cfg dict to a string representation
        cfg_str = repr(cfg)
        with open(script_path, "w") as f:
            f.write(f"""
from ultralytics import YOLO
model = YOLO('{base_weights}')
results = model.train(data='{yaml_path}', **{cfg_str})
print("Training finished successfully.")
""")

        self.log("🚀 Launching YOLO training process...")
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        def log_output():
            for line in iter(process.stdout.readline, ''):
                if line.strip():
                    self.log(f"[YOLO] {line.strip()}")
            process.stdout.close()

        t = threading.Thread(target=log_output, daemon=True)
        t.start()
        process.wait()

        try:
            os.remove(script_path)
        except:
            pass

        if process.returncode == 0:
            self.log("🏆 SMART FORGE SUCCESSFUL.")
            new_weights = self._find_latest_weights()
            if new_weights:
                self._latest_weights = new_weights
                self.log(f"✅ New model saved: {new_weights}")
            else:
                self.log("⚠️ New weights not found, but training may have completed.")
        else:
            self.log(f"💀 FORGE FAILED with exit code {process.returncode}")

    def _get_smart_training_config(self, n_samples: int) -> dict:
        """Return valid YOLO training arguments (no fl_gamma)."""
        base = {
            'imgsz': 640,
            'device': 0,
            'workers': 2,
            'rect': True,
            'mosaic': 1.0,
            'copy_paste': 0.3,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'scale': 0.5,
            'translate': 0.1,
            'degrees': 5,
            'flipud': 0.0,
            'fliplr': 0.5,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'verbose': True,
        }
        if n_samples < COLD_START_MAX:
            return {
                **base,
                'epochs': 50,
                'freeze': 10,
                'batch': 8,
                'patience': 15,
                'mixup': 0.0,
                'name': 'SenseNova_ColdStart'
            }
        elif n_samples < WARM_TRAIN_MAX:
            return {
                **base,
                'epochs': 80,
                'freeze': 5,
                'batch': 12,
                'patience': 20,
                'mixup': 0.1,
                'name': 'SenseNova_WarmTrain'
            }
        else:
            return {
                **base,
                'epochs': 150,
                'freeze': 0,
                'batch': 16,
                'patience': 30,
                'mixup': 0.15,
                'multi_scale': True,
                'name': 'SenseNova_FullTrain'
            }

    def validate_and_prune(self):
        self.log("🔍 Signal Integrity Audit...")
        if not os.path.exists(self.output_path):
            self.log("❌ No combat_boxes.jsonl found.")
            return None
        with open(self.output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.log(f"   Loaded {len(lines)} JSONL entries.")
        valid_data = []
        discarded = 0
        missing_images = 0
        for line in lines:
            try:
                data = json.loads(line)
                img_path = os.path.join(self.approved_dir, data['file'])
                if not os.path.exists(img_path):
                    missing_images += 1
                    continue
                img = cv2.imread(img_path)
                if img is None:
                    missing_images += 1
                    continue
                h, w = img.shape[:2]
                clean_boxes = []
                corrupt = False
                for b in data.get('boxes', []):
                    if (b['x'] - b['w']/2 < 0 or b['x'] + b['w']/2 > w or
                        b['y'] - b['h']/2 < 0 or b['y'] + b['h']/2 > h):
                        corrupt = True
                        break
                    if b['w'] < 4 or b['h'] < 4:
                        continue
                    clean_boxes.append(b)
                if not corrupt and clean_boxes:
                    data['boxes'] = clean_boxes
                    valid_data.append(data)
                else:
                    discarded += 1
            except Exception as e:
                self.log(f"   Skipping invalid JSON: {e}")
                discarded += 1
        self.log(f"✅ Audit: {len(valid_data)} valid, {discarded} discarded, {missing_images} missing images.")
        return valid_data

    # --------------------------------------------------------------------------
    # Evasion Engine
    # --------------------------------------------------------------------------
    def toggle_evasion(self):
        if not _YOLO_OK or not _INPUT_OK or not _MSS_OK:
            self.log("❌ Missing dependencies for evasion.")
            return
        if not self.is_evading:
            if self.bot and self.bot_thread and self.bot_thread.is_alive():
                self.bot.stop()
                self.bot_status_var.set("Bot: stopped (evasion override)")
                self.log("Bot stopped for evasion.")
            for k in ['w','a','s','d']:
                pydirectinput.keyUp(k)
            weights = self._find_latest_weights()
            if not weights:
                self.log("❌ No trained model. Train first.")
                return
            self.log(f"Loading model from {weights}...")
            try:
                model = YOLO(weights).to('cuda')
            except Exception as e:
                self.log(f"Failed to load model: {e}")
                return
            self.is_evading = True
            self.btn_eva.config(text="DISENGAGE EVASION", bg="#c0392b")
            threading.Thread(target=self.evasion_worker, args=(model,), daemon=True).start()
        else:
            self.is_evading = False
            self.btn_eva.config(text="⚔ ENGAGE EVASION", bg="#2980b9")
            self.log("Evasion disengaged.")

    def evasion_worker(self, model):
        self.log("⚔ EVASION ACTIVE – AI dodging enabled")
        prev_gray = None
        flow_scale = 0.5
        flow_predict_frames = 3
        dodges = 0
        manual = 0
        last_log = time.time()

        cv2.startWindowThread()
        hud_win = "Vision_HUD_Overlay"

        with mss() as sct:
            while self.is_evading:
                if keyboard.is_pressed(self.panic_exit):
                    break

                rect = self.get_game_rect()
                if not rect:
                    time.sleep(0.5)
                    continue

                img = np.array(sct.grab(rect))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                h, w = frame.shape[:2]
                px, py = w // 2, h // 2

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow = None
                if prev_gray is not None and prev_gray.shape == gray.shape:
                    small_gray = cv2.resize(gray, (0, 0), fx=flow_scale, fy=flow_scale)
                    small_prev = cv2.resize(prev_gray, (0, 0), fx=flow_scale, fy=flow_scale)
                    flow = cv2.calcOpticalFlowFarneback(small_prev, small_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                prev_gray = gray

                results = model.predict(frame, conf=self.min_conf, verbose=False)
                hud = frame.copy()
                manual_override = keyboard.is_pressed(self.panic_key)
                fx = fy = 0.0

                cv2.circle(hud, (px, py), self.safe_radius, (255, 255, 255), 1)

                threat_counts = {}
                for box in results[0].boxes:
                    c = box.xywh[0]
                    hx, hy = float(c[0]), float(c[1])
                    hw, hh = float(c[2]) / 2, float(c[3]) / 2
                    label = model.names[int(box.cls[0])]
                    threat_counts[label] = threat_counts.get(label, 0) + 1

                    if label == 'projectile' and flow is not None:
                        cx_flow = int(hx * flow_scale)
                        cy_flow = int(hy * flow_scale)
                        if 0 <= cx_flow < flow.shape[1] and 0 <= cy_flow < flow.shape[0]:
                            vx, vy = flow[cy_flow, cx_flow]
                            vx = vx / flow_scale * flow_predict_frames
                            vy = vy / flow_scale * flow_predict_frames
                            hx += vx
                            hy += vy

                    col = (0, 255, 0) if label == 'mob' else (0, 255, 255) if label == 'projectile' else (0, 0, 255)
                    cv2.rectangle(hud, (int(hx - hw), int(hy - hh)), (int(hx + hw), int(hy + hh)), col, 2)

                    dx, dy = px - hx, py - hy
                    dist = np.hypot(dx, dy) + 1e-5
                    if dist < self.safe_radius:
                        sigma = self.safe_radius / 3
                        falloff = np.exp(-(dist ** 2) / (2 * sigma ** 2))
                        base_wt = 3.5 if label == 'projectile' else 2.0 if label == 'aoe_indicator' else 1.0
                        heading = 1.0
                        if label == 'projectile' and flow is not None:
                            vel = np.array([vx, vy]) if 'vx' in locals() else np.array([0, 0])
                            speed = np.linalg.norm(vel)
                            if speed > 0:
                                dir_to_player = np.array([dx, dy]) / dist
                                heading = 1.0 + max(0, np.dot(vel / speed, dir_to_player))
                        weight = base_wt * falloff * heading
                        fx += (dx / dist) * weight
                        fy += (dy / dist) * weight

                thr = 0.25
                if time.time() - last_log > 1.5:
                    t_str = ", ".join(k + ":" + str(v) for k, v in threat_counts.items()) if threat_counts else "None"
                    if manual_override:
                        self.log(f"[HUD] Manual Override (Shift) | Threats: {t_str}")
                    elif abs(fx) > thr or abs(fy) > thr:
                        self.log(f"[HUD] EVADING ({fx:.1f},{fy:.1f}) | Threats: {t_str}")
                    else:
                        self.log(f"[HUD] Safe | Threats: {t_str}")
                    last_log = time.time()

                if manual_override:
                    manual += 1
                    cv2.putText(hud, "MANUAL OVERRIDE (Shift)", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 165, 255), 2)
                elif abs(fx) > thr or abs(fy) > thr:
                    dodges += 1
                    if fx > thr:
                        pydirectinput.keyDown('d')
                        pydirectinput.keyUp('a')
                    elif fx < -thr:
                        pydirectinput.keyDown('a')
                        pydirectinput.keyUp('d')
                    else:
                        pydirectinput.keyUp('a')
                        pydirectinput.keyUp('d')
                    if fy > thr:
                        pydirectinput.keyDown('s')
                        pydirectinput.keyUp('w')
                    elif fy < -thr:
                        pydirectinput.keyDown('w')
                        pydirectinput.keyUp('s')
                    else:
                        pydirectinput.keyUp('w')
                        pydirectinput.keyUp('s')
                    cv2.putText(hud, "EVADING", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
                else:
                    for k in ['w', 'a', 's', 'd']:
                        pydirectinput.keyUp(k)
                    cv2.putText(hud, "IDLE", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

                cv2.imshow(hud_win, cv2.resize(hud, (1280, 720)))
                cv2.waitKey(1)
                time.sleep(0.033 if len(results[0].boxes) > 0 else 0.1)

        cv2.destroyWindow(hud_win)
        for k in ['w', 'a', 's', 'd']:
            pydirectinput.keyUp(k)
        auto = (dodges - manual) / max(1, dodges) * 100 if dodges else 0
        self.log(f"🏁 Evasion stopped – Dodges: {dodges}, Overrides: {manual}, Autonomy: {auto:.1f}%")

    # --------------------------------------------------------------------------
    # Bot
    # --------------------------------------------------------------------------
    def bot_connect_adb(self):
        if not _BOT_OK:
            self.log("❌ adb_controller not available.")
            return
        device = self.monitor_device_var.get().strip()
        mode   = getattr(self, 'mumu_mode_var', None)
        mode   = mode.get() if mode else "adb"
        self.adb = ADBController(
            device_id=device,
            logger=self.log,
            mode=mode,
        )
        self.adb.device    = device
        self.adb.device_id = device
        ok, msg = self.adb.connect()
        self.adb_status_var.set(f"ADB: {'Connected ✓' if ok else 'Failed ✗'}")

    def bot_start(self):
        if not _BOT_OK:
            self.log("❌ bot_loop not available.")
            return
        if self.bot_thread and self.bot_thread.is_alive():
            self.log("[BOT] Already running.")
            return
        if self.adb is None:
            self.bot_connect_adb()
        try:
            self.bot = ArcheroBot(
                self.adb,
                project_root=self.project_root,
                logger=self.log,
                log_frames=self.log_frames_var.get(),
                log_every=self.log_freq_var.get()
            )
        except TypeError as e:
            self.log(f"⚠ Bot init error: {e}. Trying without adb...")
            try:
                self.bot = ArcheroBot(
                    project_root=self.project_root,
                    logger=self.log,
                    log_frames=self.log_frames_var.get(),
                    log_every=self.log_freq_var.get()
                )
            except Exception as e2:
                self.log(f"❌ Cannot start bot: {e2}")
                return
        self.bot_thread = threading.Thread(target=self.bot.run, daemon=True)
        self.bot_thread.start()
        self.bot_status_var.set("Bot: running ▶")
        self.log("[BOT] Started.")

    def bot_stop(self):
        if self.bot:
            self.bot.stop()
        self.bot_status_var.set("Bot: stopped ■")
        self.log("[BOT] Stopped.")

    def _poll(self):
        # Primary: read reward_state.json written by bot every 30 frames
        try:
            rsp = getattr(self, 'reward_state_path', None)
            if rsp and os.path.exists(rsp):
                d    = json.loads(open(rsp).read())
                rwd  = float(d.get("total_reward", 0.0))
                ep   = float(d.get("episode_reward", 0.0))
                step = int(d.get("step", 0))
                dodg = int(d.get("dodges", 0))
                dth  = int(d.get("deaths", 0))
                self.reward_var.set(f"Reward: {rwd:.2f}")
                rd = getattr(self, 'reward_detail_var', None)
                if rd:
                    rd.set(f"ep={ep:.1f}  step={step}  dodges={dodg}  deaths={dth}")
        except Exception:
            pass
        # Secondary: screen label from bot object
        if self.bot:
            try:
                st = self.bot.get_status()
                self.screen_var.set(f"Screen: {st.get('screen', '?')}")
                rsp = getattr(self, 'reward_state_path', None)
                if not rsp or not os.path.exists(rsp):
                    rwd = st.get('reward', st.get('total_reward', 0.0))
                    self.reward_var.set(f"Reward: {rwd:.2f}")
            except Exception:
                pass
        self.root.after(self.POLL_MS, self._poll)

    def _cleanup(self):
        if self.bot:
            try:
                self.bot.stop()
            except Exception:
                pass
        if self.is_capturing:
            self.is_capturing = False

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    root = tk.Tk()
    app = UnifiedCommandCenter(root)
    root.after(app.POLL_MS, app._poll)
    root.mainloop()

if __name__ == "__main__":
    main()