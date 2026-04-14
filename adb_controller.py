"""
adb_controller.py  v3.1
========================
Complete rebuild — adds the three methods the bot needs every frame:
  screencap()   — bot_loop calls this each iteration to get the frame
  swipe()       — _execute_movement calls this to move the character
  click()       — skill handler calls this to select skill cards

Also added: move_vector(), stop_move(), detailed connect() logging.
"""
from __future__ import annotations
import math, subprocess, threading, time
from typing import Optional
import cv2, numpy as np

_DPAD = {
    "up":["w"],"down":["s"],"left":["a"],"right":["d"],
    "up_left":["w","a"],"up_right":["w","d"],
    "down_left":["s","a"],"down_right":["s","d"],"idle":[],
}
_JOY_CX, _JOY_CY, _JOY_R = 0.50, 0.82, 380


class ADBController:
    """
    Unified ADB + optional MuMu keyboard controller.
    Compatible with all sensenova_gui.py versions and bot_loop.py.
    """
    def __init__(self, device_id="127.0.0.1:7555", logger=None,
                 mode="adb", mumu_window="MuMu", **kwargs):
        self.device_id   = device_id
        self.device      = device_id   # alias — GUI sets self.adb.device
        self.logger      = logger or print
        self.mode        = mode.lower()
        self.mumu_window = mumu_window
        self._held: set  = set()
        self._lock       = threading.Lock()
        self._pdi        = None
        self._consec_screencap_fails = 0

        if self.mode == "mumu_key":
            try:
                import pydirectinput as p; self._pdi = p
                self._log("Keyboard mode: MuMu WASD dpad (W/A/S/D)")
            except ImportError:
                self._log("pydirectinput not found — using ADB swipe mode")
                self.mode = "adb"
        if self.mode == "adb":
            self._log(f"ADB swipe mode | device: {self.device_id}")

    # ── internal ──────────────────────────────────────────────────────────────

    def _log(self, m):
        try: self.logger(f"[ADB] {m}")
        except Exception: print(f"[ADB] {m}")

    def _exec(self, cmd, timeout=8):
        """Run cmd list, return (stdout, stderr, returncode). Never raises."""
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               timeout=timeout)
            out = r.stdout.decode("utf-8", errors="replace").strip()
            err = r.stderr.decode("utf-8", errors="replace").strip()
            return out, err, r.returncode
        except FileNotFoundError:
            return "", "adb not found — add Android Platform Tools to PATH", -1
        except subprocess.TimeoutExpired:
            return "", f"timed out after {timeout}s", -2
        except Exception as e:
            return "", str(e), -3

    def _run(self, cmd, timeout=5):
        """Fire-and-forget for click/swipe. Returns True on success."""
        try:
            subprocess.run(cmd, capture_output=True, timeout=timeout)
            return True
        except Exception:
            return False

    # ── connection ────────────────────────────────────────────────────────────

    def connect(self):
        """
        Full verbose connection sequence.
        Logs every ADB response line to the GUI log.
        Returns (True, msg) on success, (False, msg) on failure.
        """
        self._log("=" * 44)
        self._log(f"Connecting to MuMu Player [{self.device_id}]")
        self._log("=" * 44)

        # 1. Check adb on PATH
        out, err, rc = self._exec(["adb", "version"])
        if rc != 0:
            self._log(f"❌ {err or 'adb not found'}"); return False, err
        self._log(f"✓  {out.splitlines()[0]}")

        # 2. Start server
        self._log("   Starting ADB server …")
        out, err, _ = self._exec(["adb", "start-server"], timeout=12)
        for ln in (out + "\n" + err).splitlines():
            if ln.strip(): self._log(f"   server: {ln.strip()}")

        # 3. Connect
        self._log(f"   → adb connect {self.device_id}")
        out, err, _ = self._exec(["adb", "connect", self.device_id], timeout=10)
        resp = (out + "\n" + err).strip()
        for ln in resp.splitlines():
            if ln.strip(): self._log(f"   {ln.strip()}")

        if any(k in resp.lower() for k in ["cannot","refused","failed","unable","error:"]):
            self._log("❌ Connect failed")
            self._log("   • MuMu running? Settings → Other → Enable ADB")
            return False, resp

        # 4. Verify in adb devices
        self._log("   → adb devices")
        out, _, _ = self._exec(["adb", "devices"])
        for ln in out.splitlines():
            self._log(f"   {ln}")

        status = "not listed"
        for ln in out.splitlines():
            if self.device_id in ln:
                parts = ln.split(); status = parts[1] if len(parts) >= 2 else "?"
                break
        self._log(f"   Device status: {status}")

        if status == "device":
            msg = f"Connected ✓  [{self.device_id}]  mode={self.mode}"
            self._log(f"✅ {msg}"); return True, msg
        if status == "offline":
            msg = f"{self.device_id} OFFLINE — restart MuMu and reconnect"
            self._log(f"⚠  {msg}"); return False, msg
        if status == "unauthorized":
            msg = f"{self.device_id} UNAUTHORIZED — accept ADB prompt in MuMu"
            self._log(f"⚠  {msg}"); return False, msg

        msg = f"Sent connect to {self.device_id} (status={status!r})"
        self._log(f"⚠  {msg}"); return True, msg

    # ── screen capture ────────────────────────────────────────────────────────

    def screencap(self) -> Optional[np.ndarray]:
        """
        Capture MuMu screen via ADB exec-out screencap -p.
        Returns BGR ndarray, or None on failure.
        Called every ~20 ms by bot_loop — must never raise.
        """
        try:
            r = subprocess.run(
                ["adb", "-s", self.device_id, "exec-out", "screencap", "-p"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=6)
            if not r.stdout:
                err = r.stderr.decode("utf-8", errors="replace").strip()
                self._consec_screencap_fails += 1
                if self._consec_screencap_fails % 10 == 1:
                    self._log(f"screencap empty (fail #{self._consec_screencap_fails}): {err}")
                return None
            arr = np.frombuffer(r.stdout, np.uint8)
            if arr.size == 0: return None
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                self._consec_screencap_fails = 0
            return frame
        except subprocess.TimeoutExpired:
            self._log("screencap timed out"); return None
        except Exception as e:
            self._log(f"screencap: {e}"); return None

    # ── input ─────────────────────────────────────────────────────────────────

    def click(self, x, y):
        """Tap at (x, y). Used by skill screen handler."""
        ok = self._run(["adb", "-s", self.device_id, "shell", "input", "tap",
                        str(int(x)), str(int(y))])
        return ok

    def tap(self, x, y):
        """Alias of click() for backward compatibility."""
        return self.click(x, y)

    def swipe(self, x1, y1, x2, y2, duration=250):
        """
        ADB swipe from (x1,y1) to (x2,y2) over duration ms.
        Called by _execute_movement() every ~0.55 s.
        """
        ok = self._run(["adb", "-s", self.device_id, "shell", "input", "swipe",
                        str(int(x1)), str(int(y1)), str(int(x2)), str(int(y2)),
                        str(int(duration))])
        return ok

    def shell(self, command):
        out, _, _ = self._exec(["adb", "-s", self.device_id, "shell"] +
                                command.split(), timeout=5)
        return out

    # ── movement ──────────────────────────────────────────────────────────────

    def move(self, direction, frame_shape=None):
        d = direction.lower()
        if self.mode == "mumu_key": self._key_move(d)
        else: self._adb_move(d, frame_shape)

    def stop_move(self):
        if self.mode == "mumu_key": self._release_all()

    def move_vector(self, fx, fy, frame_shape):
        if self.mode == "mumu_key":
            self._key_move(_v2d(fx, fy)); return
        h, w = frame_shape[:2]
        cx, cy = int(w * _JOY_CX), int(h * _JOY_CY)
        mag = math.hypot(fx, fy) + 1e-9
        r   = min(mag * _JOY_R, _JOY_R)
        tx  = max(0, min(w-1, int(cx + (fx/mag)*r)))
        ty  = max(0, min(h-1, int(cy + (fy/mag)*r)))
        self.swipe(cx, cy, tx, ty, duration=180)

    def _adb_move(self, direction, frame_shape):
        if direction == "idle" or not frame_shape: return
        h, w = frame_shape[:2]
        cx, cy = int(w * _JOY_CX), int(h * _JOY_CY)
        vx, vy = {"up":(0,-1),"down":(0,1),"left":(-1,0),"right":(1,0),
                  "up_left":(-0.707,-0.707),"up_right":(0.707,-0.707),
                  "down_left":(-0.707,0.707),"down_right":(0.707,0.707)}.get(direction,(0,0))
        tx = max(0, min(w-1, int(cx+vx*_JOY_R)))
        ty = max(0, min(h-1, int(cy+vy*_JOY_R)))
        self.swipe(cx, cy, tx, ty, duration=200)

    def _key_move(self, direction):
        if not self._pdi: return
        keys = _DPAD.get(direction, [])
        with self._lock:
            for k in list(self._held):
                if k not in keys:
                    try: self._pdi.keyUp(k)
                    except Exception: pass
            self._held.clear()
            for k in keys:
                try: self._pdi.keyDown(k); self._held.add(k)
                except Exception: pass

    def _release_all(self):
        if not self._pdi: return
        with self._lock:
            for k in list(self._held):
                try: self._pdi.keyUp(k)
                except Exception: pass
            self._held.clear()

    def _focus_mumu(self):
        try:
            import pygetwindow as gw
            wins = [w for w in gw.getAllWindows()
                    if self.mumu_window.lower() in w.title.lower()]
            if wins: wins[0].activate(); time.sleep(0.04)
        except Exception: pass

    def status_dict(self):
        return {"device": self.device_id, "mode": self.mode}


def _v2d(fx, fy, thr=0.25):
    if abs(fx) < thr and abs(fy) < thr: return "idle"
    a = math.degrees(math.atan2(-fy, fx))
    if   -22.5 <= a <  22.5: return "right"
    elif  22.5 <= a <  67.5: return "up_right"
    elif  67.5 <= a < 112.5: return "up"
    elif 112.5 <= a < 157.5: return "up_left"
    elif abs(a) >= 157.5:    return "left"
    elif -67.5 <= a < -22.5: return "down_right"
    elif-112.5 <= a < -67.5: return "down"
    else:                     return "down_left"
