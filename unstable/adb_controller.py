"""
adb_controller.py  —  ADB + MuMu Keyboard Controller
======================================================
Supports two movement modes:
  ADB_SWIPE (default) — sends swipe commands via ADB shell input.
                        Works regardless of which window has focus.
                        Most reliable for all bot actions.

  MUMU_KEY  (optional) — sends WASD keystrokes to MuMu emulator window.
                         Matches MuMu's virtual dpad layout:
                           W = north,  S = south
                           A = west,   D = east
                         Requires MuMu window to have focus.
                         Use only if ADB swipe timing is too slow.

Movement API (both modes share the same interface):
  adb.move(direction)   — direction: 'up','down','left','right',
                          'up_left','up_right','down_left','down_right','idle'
  adb.stop_move()       — release all movement
  adb.swipe(...)        — raw ADB swipe (always ADB regardless of mode)
  adb.click(x, y)       — ADB tap
  adb.screencap()       — ADB screenshot → BGR numpy array
"""
from __future__ import annotations

import subprocess
import time
import threading
from typing import Optional

import cv2
import numpy as np

# MuMu dpad key mapping
_DPAD_KEYS: dict[str, list[str]] = {
    'up':         ['w'],
    'down':       ['s'],
    'left':       ['a'],
    'right':      ['d'],
    'up_left':    ['w', 'a'],
    'up_right':   ['w', 'd'],
    'down_left':  ['s', 'a'],
    'down_right': ['s', 'd'],
    'idle':       [],
}

# Joystick centre and radius for ADB swipe mode
# These are fractions of frame dimensions
_JOY_CX_FRAC = 0.5    # horizontal centre
_JOY_CY_FRAC = 0.82   # bottom quarter of screen (where the virtual joystick sits)
_JOY_RADIUS   = 380   # pixel offset for full-deflection swipe


class ADBController:
    """
    Unified ADB + MuMu keyboard controller.

    Parameters
    ----------
    device_id : str
        ADB device address, e.g. "127.0.0.1:7555"
    logger : callable, optional
        Log function accepting a single string.
    mode : str
        "adb" (default) or "mumu_key".
    mumu_window : str
        Partial window title used to focus MuMu when mode="mumu_key".
    """

    def __init__(self,
                 device_id: str = "127.0.0.1:7555",
                 logger=None,
                 mode: str = "adb",
                 mumu_window: str = "MuMu"):
        self.device_id   = device_id
        self.device      = device_id          # alias kept for GUI compat
        self.logger      = logger or print
        self.mode        = mode.lower()       # "adb" | "mumu_key"
        self.mumu_window = mumu_window
        self._held_keys: set[str] = set()
        self._key_lock   = threading.Lock()

        # Lazy-import pydirectinput only if keyboard mode requested
        self._pdi = None
        if self.mode == "mumu_key":
            try:
                import pydirectinput as pdi
                self._pdi = pdi
                self._log("Mode: MuMu keyboard (WASD → dpad)")
            except ImportError:
                self._log("⚠ pydirectinput not installed — falling back to ADB mode")
                self.mode = "adb"

        if self.mode == "adb":
            self._log("Mode: ADB swipe (reliable, no window focus needed)")

    # ── Internal ───────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        try:
            self.logger(f"[ADB] {msg}")
        except Exception:
            print(f"[ADB] {msg}")

    def _run(self, cmd: str | list, timeout: int = 5) -> subprocess.CompletedProcess:
        if isinstance(cmd, str):
            return subprocess.run(cmd, shell=True, capture_output=True, timeout=timeout)
        return subprocess.run(cmd, capture_output=True, timeout=timeout)

    # ── Connection ─────────────────────────────────────────────────────────────

    def connect(self) -> tuple[bool, str]:
        """Connect to ADB device. Returns (ok, message)."""
        try:
            self._run(f"adb connect {self.device_id}")
            return True, f"Connected to {self.device_id}"
        except Exception as e:
            return False, str(e)

    # ── Screen capture ─────────────────────────────────────────────────────────

    def screencap(self) -> Optional[np.ndarray]:
        """Return current game frame as a BGR numpy array, or None on failure."""
        try:
            res = self._run(
                ["adb", "-s", self.device_id, "exec-out", "screencap", "-p"],
                timeout=5)
            if not res.stdout:
                return None
            arr = np.frombuffer(res.stdout, np.uint8)
            if arr.size == 0:
                return None
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    # ── Basic input ────────────────────────────────────────────────────────────

    def click(self, x: int, y: int):
        """Send a tap at pixel (x, y)."""
        self._run(f"adb -s {self.device_id} shell input tap {x} {y}")

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: int = 250):
        """Send a raw ADB swipe. Always uses ADB regardless of mode."""
        self._run(
            f"adb -s {self.device_id} shell input swipe "
            f"{x1} {y1} {x2} {y2} {duration}"
        )

    # ── High-level movement ────────────────────────────────────────────────────

    def move(self, direction: str, frame_shape: tuple | None = None):
        """
        Move the character in `direction`.

        In ADB mode  : sends a swipe from joystick centre toward the direction.
        In MuMu mode : presses the appropriate WASD keys.

        Parameters
        ----------
        direction : str
            One of 'up','down','left','right','up_left','up_right',
            'down_left','down_right','idle'.
        frame_shape : (h, w) tuple, optional
            Required in ADB mode to compute joystick centre.
            Ignored in MuMu keyboard mode.
        """
        direction = direction.lower()
        if self.mode == "mumu_key":
            self._key_move(direction)
        else:
            self._adb_move(direction, frame_shape)

    def stop_move(self):
        """Release all movement inputs."""
        if self.mode == "mumu_key":
            self._release_all_keys()
        # In ADB mode the swipe naturally ends — nothing to release

    def _adb_move(self, direction: str, frame_shape: tuple | None):
        """Compute swipe vector and send via ADB."""
        if direction == 'idle' or frame_shape is None:
            return
        h, w = frame_shape[:2]
        cx = int(w * _JOY_CX_FRAC)
        cy = int(h * _JOY_CY_FRAC)

        # Direction → unit vector
        vx, vy = {
            'up':         ( 0, -1),
            'down':       ( 0,  1),
            'left':       (-1,  0),
            'right':      ( 1,  0),
            'up_left':    (-0.707, -0.707),
            'up_right':   ( 0.707, -0.707),
            'down_left':  (-0.707,  0.707),
            'down_right': ( 0.707,  0.707),
        }.get(direction, (0, 0))

        tx = int(cx + vx * _JOY_RADIUS)
        ty = int(cy + vy * _JOY_RADIUS)
        tx = max(0, min(w - 1, tx))
        ty = max(0, min(h - 1, ty))
        self.swipe(cx, cy, tx, ty, duration=200)

    def move_vector(self, fx: float, fy: float, frame_shape: tuple):
        """
        Move using a continuous force vector (from evasion potential field).
        Converts (fx, fy) → ADB swipe or WASD keys.

        Parameters
        ----------
        fx, fy : float
            Force components. Magnitude controls swipe distance (ADB) or
            which diagonal keys are pressed (MuMu).
        frame_shape : (h, w, ...) tuple
        """
        if self.mode == "mumu_key":
            # Quantise vector to 8 directions
            direction = _vector_to_direction(fx, fy)
            self._key_move(direction)
        else:
            h, w = frame_shape[:2]
            cx = int(w * _JOY_CX_FRAC)
            cy = int(h * _JOY_CY_FRAC)
            mag = (fx**2 + fy**2) ** 0.5 + 1e-9
            # Scale radius by magnitude, cap at _JOY_RADIUS
            r = min(mag * _JOY_RADIUS, _JOY_RADIUS)
            tx = int(cx + (fx / mag) * r)
            ty = int(cy + (fy / mag) * r)
            tx = max(0, min(w - 1, tx))
            ty = max(0, min(h - 1, ty))
            self.swipe(cx, cy, tx, ty, duration=180)

    # ── MuMu keyboard helpers ──────────────────────────────────────────────────

    def _focus_mumu(self):
        """Try to bring MuMu window to front so key events land there."""
        try:
            import pygetwindow as gw
            wins = [w for w in gw.getAllWindows()
                    if self.mumu_window.lower() in w.title.lower()]
            if wins:
                wins[0].activate()
                time.sleep(0.05)
        except Exception:
            pass

    def _key_move(self, direction: str):
        """Press the WASD keys for the given direction, release old ones."""
        if self._pdi is None:
            return
        keys = _DPAD_KEYS.get(direction, [])
        with self._key_lock:
            # Release keys not in the new direction
            for k in list(self._held_keys):
                if k not in keys:
                    try:
                        self._pdi.keyUp(k)
                    except Exception:
                        pass
            self._held_keys = set()
            # Press new keys
            for k in keys:
                try:
                    self._pdi.keyDown(k)
                    self._held_keys.add(k)
                except Exception:
                    pass

    def _release_all_keys(self):
        """Release every held key."""
        if self._pdi is None:
            return
        with self._key_lock:
            for k in list(self._held_keys):
                try:
                    self._pdi.keyUp(k)
                except Exception:
                    pass
            self._held_keys.clear()

    # ── Status ─────────────────────────────────────────────────────────────────

    def status_dict(self) -> dict:
        return {
            "device":         self.device_id,
            "mode":           self.mode,
            "device_status":  "DEVICE",
        }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _vector_to_direction(fx: float, fy: float, threshold: float = 0.25) -> str:
    """Convert a 2D force vector to one of 9 dpad directions."""
    if abs(fx) < threshold and abs(fy) < threshold:
        return 'idle'
    angle_deg = 0.0
    import math
    angle = math.atan2(-fy, fx)   # fy inverted because y grows downward
    angle_deg = math.degrees(angle)

    # Map angle to 8 compass directions (0° = right, 90° = up)
    if   -22.5 <= angle_deg <  22.5: return 'right'
    elif  22.5 <= angle_deg <  67.5: return 'up_right'
    elif  67.5 <= angle_deg < 112.5: return 'up'
    elif 112.5 <= angle_deg < 157.5: return 'up_left'
    elif abs(angle_deg) >= 157.5:    return 'left'
    elif -67.5 <= angle_deg < -22.5: return 'down_right'
    elif-112.5 <= angle_deg < -67.5: return 'down'
    else:                             return 'down_left'
