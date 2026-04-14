# core/gui.py – Main window, now uses EvasionEngine and Labeler modules
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from mss import mss

from core.evasion import EvasionEngine
# from core.labeler import Labeler  # if you split labeler

class UnifiedCommandCenter:
    def __init__(self, root):
        self.root = root
        self.root.title("SenseNova v22.0 | Evasion Fixed")
        self.root.geometry("860x920")
        self.root.configure(bg="#1a1a1a")
        self.project_root = Path(__file__).parent.parent

        # Paths
        self.base_dir = str(self.project_root / "data")
        self.inbox_dir = os.path.join(self.base_dir, "high_priority_projectiles")
        self.approved_dir = os.path.join(self.base_dir, "approved_projectiles")
        self.discarded_dir = os.path.join(self.base_dir, "discarded_projectiles")
        self.output_path = os.path.join(self.base_dir, "combat_boxes.jsonl")
        self.yolo_root = os.path.join(self.base_dir, "yolo_dataset")
        for d in [self.inbox_dir, self.approved_dir, self.discarded_dir, self.yolo_root]:
            os.makedirs(d, exist_ok=True)

        # Game window title
        self.game_window_title = tk.StringVar(value="MuMuPlayer")
        self.is_evading = False
        self.safe_radius = 400
        self.min_conf = 0.55

        # Bot and ADB
        self.adb = None
        self.bot = None
        self.bot_thread = None
        self.log_frames_var = tk.BooleanVar(value=True)
        self.log_freq_var = tk.IntVar(value=5)
        self.monitor_device_var = tk.StringVar(value="127.0.0.1:7555")
        self.adb_status_var = tk.StringVar(value="ADB: disconnected")
        self.bot_status_var = tk.StringVar(value="Bot: stopped")

        # Evasion engine
        self.evasion = None
        self.sct = mss()  # shared for screen capture

        self._setup_ui()
        self.log("🚀 System Ready")

    def log(self, msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")  # placeholder – integrate with log area

    def _setup_ui(self):
        nb = ttk.Notebook(self.root)
        # Tabs: Capture, Labeler, Trainer, Evasion, Bot (Dailies removed)
        for text, builder in [
            ("Capture", self._build_capture_tab),
            ("Labeler", self._build_labeler_tab),
            ("Trainer", self._build_trainer_tab),
            ("Evasion", self._build_evasion_tab),
            ("Bot", self._build_bot_tab),
        ]:
            tab = ttk.Frame(nb)
            nb.add(tab, text=text)
            builder(tab)
        nb.pack(expand=True, fill="both", padx=10, pady=10)

    def _build_evasion_tab(self, tab):
        card = self._card(tab, "EVASION ENGINE")
        win_frame = tk.Frame(card, bg="#1e1e1e")
        win_frame.pack(fill="x", padx=12, pady=5)
        tk.Label(win_frame, text="Game Window Title:", bg="#1e1e1e", fg="white").pack(side="left")
        tk.Entry(win_frame, textvariable=self.game_window_title, width=20).pack(side="left", padx=6)
        self.btn_eva = tk.Button(card, text="⚔ ENGAGE EVASION", command=self.toggle_evasion, bg="#2980b9", fg="white")
        self.btn_eva.pack(fill="x", padx=12, pady=10)

    def toggle_evasion(self):
        if not self.is_evading:
            # Stop bot if running
            if self.bot and self.bot_thread and self.bot_thread.is_alive():
                self.bot.stop()
                self.bot_status_var.set("Bot: stopped (evasion override)")
                self.log("Bot stopped for evasion.")
            # Load model
            weights = self._find_latest_weights()
            if not weights:
                self.log("❌ No trained model. Train first.")
                return
            model = YOLO(weights).to('cuda')
            self.is_evading = True
            self.btn_eva.config(text="DISENGAGE EVASION", bg="#c0392b")
            self.evasion = EvasionEngine(self)
            threading.Thread(target=self.evasion.start, args=(model, self.safe_radius, self.min_conf), daemon=True).start()
        else:
            self.is_evading = False
            if self.evasion:
                self.evasion.stop()
            self.btn_eva.config(text="⚔ ENGAGE EVASION", bg="#2980b9")
            self.log("Evasion disengaged.")

    def get_game_rect(self):
        import pygetwindow as gw
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

    def _find_latest_weights(self):
        runs_dir = self.project_root / "runs" / "detect"
        if not runs_dir.exists():
            return None
        candidates = sorted(runs_dir.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        return str(candidates[0]) if candidates else None

    # Stubs for other tabs (Capture, Labeler, Trainer, Bot) – keep original functionality
    def _build_capture_tab(self, tab): pass
    def _build_labeler_tab(self, tab): pass
    def _build_trainer_tab(self, tab): pass
    def _build_bot_tab(self, tab): pass
    def _card(self, parent, title): return tk.Frame(parent)

# Launcher
def main():
    root = tk.Tk()
    app = UnifiedCommandCenter(root)
    root.mainloop()

if __name__ == "__main__":
    main()