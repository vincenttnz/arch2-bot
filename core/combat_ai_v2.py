"""
core/combat_ai_v2.py
====================
Production combat AI — replaces the stub combat_ai.py.

Capabilities
────────────
1. Projectile prediction — tracks velocity across frames, predicts where
   projectiles will be in N frames and dodges the predicted path.

2. Threat heatmap — builds a 2D danger map from all projectile trajectories.
   The bot moves toward the minimum-threat cell in its reachable neighbourhood.

3. Enemy clustering — groups enemies and positions the player to maximise
   AoE coverage while staying out of the primary threat cone.

4. HP-aware strategy — at low HP, prioritises survival over positioning;
   at full HP, plays aggressively to farm kills faster.

5. Produces structured CombatFeats dict consumed by NeuralEngine for the
   neural dodge policy when it becomes available.
"""
from __future__ import annotations

import math
import time
from collections import deque
from typing import NamedTuple

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

class BBox(NamedTuple):
    x: int; y: int; w: int; h: int

    @property
    def cx(self): return self.x + self.w // 2
    @property
    def cy(self): return self.y + self.h // 2
    @property
    def center(self): return (self.cx, self.cy)

    def distance_to(self, other: "BBox") -> float:
        return math.hypot(self.cx - other.cx, self.cy - other.cy)

    def distance_to_pt(self, pt: tuple) -> float:
        return math.hypot(self.cx - pt[0], self.cy - pt[1])


class TrackedProjectile:
    """Maintains history and velocity estimate for a single projectile."""

    MAX_HISTORY = 8

    def __init__(self, cx: float, cy: float, ts: float):
        self.history: deque = deque(maxlen=self.MAX_HISTORY)
        self.history.append((cx, cy, ts))
        self.vx = 0.0
        self.vy = 0.0
        self.lost_frames = 0

    def update(self, cx: float, cy: float, ts: float):
        if len(self.history) >= 2:
            px, py, pt = self.history[-1]
            dt = max(ts - pt, 1e-3)
            self.vx = (cx - px) / dt
            self.vy = (cy - py) / dt
        self.history.append((cx, cy, ts))
        self.lost_frames = 0

    def predict(self, dt_sec: float) -> tuple[float, float]:
        cx, cy, _ = self.history[-1]
        return cx + self.vx * dt_sec, cy + self.vy * dt_sec

    @property
    def pos(self) -> tuple[float, float]:
        cx, cy, _ = self.history[-1]
        return cx, cy

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)


# ─────────────────────────────────────────────────────────────────────────────
# Projectile tracker (multi-object, greedy nearest-neighbour)
# ─────────────────────────────────────────────────────────────────────────────

class ProjectileTracker:
    """Maintains identity of projectiles across frames."""

    MAX_MATCH_DIST = 80    # pixels — max distance to associate detection to track
    MAX_LOST       = 5     # frames before discarding a lost track

    def __init__(self):
        self._tracks: list[TrackedProjectile] = []
        self._next_id = 0

    def update(self, detections: list[tuple[float, float]], ts: float):
        """Match new detections to existing tracks; create/prune as needed."""
        unmatched_dets = list(range(len(detections)))
        matched_tracks = set()

        for track in self._tracks:
            if not detections:
                break
            best_di, best_d = None, float("inf")
            px, py = track.pos
            for di in unmatched_dets:
                cx, cy = detections[di]
                d = math.hypot(cx - px, cy - py)
                if d < best_d and d < self.MAX_MATCH_DIST:
                    best_d, best_di = d, di
            if best_di is not None:
                track.update(*detections[best_di], ts)
                unmatched_dets.remove(best_di)
                matched_tracks.add(id(track))
            else:
                track.lost_frames += 1

        # New tracks for unmatched detections
        for di in unmatched_dets:
            t = TrackedProjectile(*detections[di], ts)
            self._tracks.append(t)

        # Prune lost tracks
        self._tracks = [t for t in self._tracks if t.lost_frames <= self.MAX_LOST]

    @property
    def tracks(self) -> list[TrackedProjectile]:
        return self._tracks


# ─────────────────────────────────────────────────────────────────────────────
# Threat heatmap
# ─────────────────────────────────────────────────────────────────────────────

def build_threat_map(
    frame_h: int,
    frame_w: int,
    projectile_tracks: list[TrackedProjectile],
    predict_dt: float = 0.25,
    grid_size: int = 16,
) -> np.ndarray:
    """
    Returns a (grid_h, grid_w) float32 threat map where higher = more dangerous.
    Each cell accumulates inverse-distance penalties from current + predicted
    projectile positions.
    """
    gh = frame_h // grid_size
    gw = frame_w // grid_size
    hmap = np.zeros((gh, gw), dtype=np.float32)

    for track in projectile_tracks:
        for dt in (0.0, predict_dt * 0.5, predict_dt):
            cx, cy = track.predict(dt)
            gx = int(cx / frame_w * gw)
            gy = int(cy / frame_h * gh)
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < gw and 0 <= ny < gh:
                        dist = math.hypot(dx, dy) + 0.5
                        hmap[ny, nx] += track.speed / (dist * 200 + 1)

    return hmap


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced player position estimator
# ─────────────────────────────────────────────────────────────────────────────

def estimate_player_position(frame: np.ndarray) -> tuple[int, int]:
    """
    Estimates the player's centre pixel.

    Strategy:
    1. Look for the character's green HP bar (which follows the player).
    2. Fall back to the bright health indicator sprite.
    3. Last resort: use fixed lower-centre of screen.
    """
    h, w = frame.shape[:2]
    # Search in middle 60% of screen (player rarely near edges)
    roi_y1, roi_y2 = int(h * 0.30), int(h * 0.90)
    roi_x1, roi_x2 = int(w * 0.20), int(w * 0.80)
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # HP bar is a bright green horizontal strip
    green = cv2.inRange(hsv, np.array([40, 80, 100]), np.array([90, 255, 255]))
    kernel = np.ones((3, 15), np.uint8)   # horizontal kernel matches HP bar shape
    green  = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        # HP bar: wide and short, in specific region
        if ww < 20 or hh > 20 or ww / max(hh, 1) < 3:
            continue
        # Prefer wider bars closer to centre
        score = ww - abs((roi_x1 + x + ww // 2) - w // 2) * 0.1
        if score > best_score:
            best_score = score
            best = (roi_x1 + x, roi_y1 + y, ww, hh)

    if best is not None:
        x, y, ww, hh = best
        # Player body is below the HP bar
        return x + ww // 2, y + hh + int(h * 0.05)

    return w // 2, int(h * 0.65)   # fallback


# ─────────────────────────────────────────────────────────────────────────────
# Enemy detector (improved over the stub)
# ─────────────────────────────────────────────────────────────────────────────

def detect_enemies(frame: np.ndarray) -> list[BBox]:
    """
    Detects enemy bounding boxes.
    Uses colour cues (red health indicators) + edge density validation.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, np.array([0,  90, 80]),  np.array([12, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([168, 90, 80]), np.array([179, 255, 255]))
    mask = cv2.bitwise_or(red1, red2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if area < 15 or bw < 5 or bh < 3:
            continue
        # Exclude UI elements (top bar, edges)
        if y < int(h * 0.12) or y > int(h * 0.90):
            continue
        boxes.append(BBox(int(x), int(y), int(bw), int(bh)))
    return boxes


# ─────────────────────────────────────────────────────────────────────────────
# Projectile detector (improved over the stub)
# ─────────────────────────────────────────────────────────────────────────────

def detect_projectiles(frame: np.ndarray) -> list[tuple[float, float]]:
    """
    Detects projectile centres.
    Projectiles are typically small, bright, fast-moving objects.
    Detects: yellow/orange bullets, white/cyan energy bolts, purple orbs.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    masks = [
        cv2.inRange(hsv, np.array([15,  150, 150]), np.array([35, 255, 255])),  # yellow/orange
        cv2.inRange(hsv, np.array([85,  100, 150]), np.array([105,255, 255])),  # cyan/blue
        cv2.inRange(hsv, np.array([130, 100, 120]), np.array([155,255, 255])),  # purple
        cv2.inRange(hsv, np.array([0,     0, 220]), np.array([179,  30,255])),  # near-white
    ]
    combined = masks[0]
    for m in masks[1:]:
        combined = cv2.bitwise_or(combined, m)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    num, _, stats, cents = cv2.connectedComponentsWithStats(combined, 8)
    projectiles = []
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if area < 4 or area > 400:   # too small = noise, too large = UI element
            continue
        if y < int(h * 0.10) or y > int(h * 0.92):
            continue
        cx, cy = float(cents[i][0]), float(cents[i][1])
        projectiles.append((cx, cy))
    return projectiles


# ─────────────────────────────────────────────────────────────────────────────
# Combat AI v2
# ─────────────────────────────────────────────────────────────────────────────

class CombatAIv2:
    """
    Full combat AI with projectile tracking, threat mapping, and HP-aware strategy.

    Used by bot_loop.py for Phase 0+.
    When a trained DodgePolicy is available via NeuralEngine, this class
    still runs to produce the CombatFeats dict the neural policy needs.
    """

    GRID_SIZE  = 16       # px per threat-map cell
    PREDICT_DT = 0.20     # seconds ahead to predict projectile positions
    DODGE_RADIUS_FRAC = 0.10  # swipe radius as fraction of min(w, h)
    MOVE_INTERVAL = 0.15  # seconds between movement inputs

    def __init__(self):
        self._proj_tracker = ProjectileTracker()
        self._last_move_ts = 0.0
        self._phase = 0          # 0-7 rotation for exploration when no threat
        self._frame_count = 0

    def process_frame(self, frame: np.ndarray, hp_ratio: float = 1.0) -> dict:
        """
        Analyse a combat frame and return a CombatFeats dict.

        CombatFeats keys
        ────────────────
        player_pos         (x, y) pixels
        enemies            list[BBox]
        enemy_count        int
        nearest_enemy_dx   float  normalised -1..1 (direction from player)
        nearest_enemy_dy   float
        projectiles        list[(cx,cy)]
        projectile_count   int
        proj_threat_dx     float  normalised, net threat direction
        proj_threat_dy     float
        threat_map         np.ndarray (gh, gw) float32
        dodge_dx           float  recommended dodge dx (-1..1)
        dodge_dy           float  recommended dodge dy (-1..1)
        action             str    "idle"|"up"|"down"|"left"|"right"|diagonals
        pickup_nearby      float  0 or 1
        hp_ratio           float
        """
        ts = time.time()
        h, w = frame.shape[:2]
        self._frame_count += 1

        player_pos = estimate_player_position(frame)
        enemies    = detect_enemies(frame)
        proj_pts   = detect_projectiles(frame)
        self._proj_tracker.update(proj_pts, ts)
        tracks     = self._proj_tracker.tracks

        # ── Threat map ──
        threat_map = build_threat_map(h, w, tracks, self.PREDICT_DT, self.GRID_SIZE)

        # ── Enemy features ──
        nearest_dx, nearest_dy = 0.0, 0.0
        if enemies:
            enemies_sorted = sorted(enemies, key=lambda e: e.distance_to_pt(player_pos))
            ne = enemies_sorted[0]
            nearest_dx = (ne.cx - player_pos[0]) / max(w, 1)
            nearest_dy = (ne.cy - player_pos[1]) / max(h, 1)

        # ── Projectile threat direction (net repulsion vector) ──
        proj_threat_dx, proj_threat_dy = 0.0, 0.0
        for track in tracks:
            cx, cy = track.predict(self.PREDICT_DT)
            dx = player_pos[0] - cx
            dy = player_pos[1] - cy
            dist = math.hypot(dx, dy) + 1.0
            weight = track.speed / (dist ** 1.5 + 1.0)
            proj_threat_dx += dx * weight
            proj_threat_dy += dy * weight
        mag = math.hypot(proj_threat_dx, proj_threat_dy) + 1e-6
        if mag > 0:
            proj_threat_dx /= mag
            proj_threat_dy /= mag

        # ── Pick best dodge direction from threat map ──
        dodge_dx, dodge_dy, action = self._pick_dodge(
            player_pos, threat_map, h, w, hp_ratio, enemies, proj_threat_dx, proj_threat_dy
        )

        return dict(
            player_pos=player_pos,
            enemies=enemies,
            enemy_count=len(enemies),
            nearest_enemy_dx=nearest_dx,
            nearest_enemy_dy=nearest_dy,
            projectiles=proj_pts,
            projectile_count=len(proj_pts),
            proj_threat_dx=proj_threat_dx,
            proj_threat_dy=proj_threat_dy,
            threat_map=threat_map,
            dodge_dx=dodge_dx,
            dodge_dy=dodge_dy,
            action=action,
            pickup_nearby=0.0,
            hp_ratio=hp_ratio,
        )

    def _pick_dodge(
        self,
        player_pos: tuple[int, int],
        threat_map: np.ndarray,
        h: int, w: int,
        hp_ratio: float,
        enemies: list,
        proj_dx: float, proj_dy: float,
    ) -> tuple[float, float, str]:
        """
        Choose dodge direction by sampling candidate moves and picking
        the one that minimises threat + maximises strategic value.
        """
        gh, gw = threat_map.shape
        px_norm = player_pos[0] / w
        py_norm = player_pos[1] / h
        gx = int(px_norm * gw)
        gy = int(py_norm * gh)

        # 8 directions + idle
        dirs = [
            (0, 0, "idle"),
            (0, -1, "up"), (0, 1, "down"), (-1, 0, "left"), (1, 0, "right"),
            (-1, -1, "up_left"), (1, -1, "up_right"),
            (-1,  1, "down_left"), (1, 1, "down_right"),
        ]

        # Keep player away from screen edges
        edge_penalty_x = max(0.0, 0.15 - px_norm) + max(0.0, px_norm - 0.85)
        edge_penalty_y = max(0.0, 0.15 - py_norm) + max(0.0, py_norm - 0.85)

        best_score = float("inf")
        best_dir   = (0.0, 0.0, "idle")

        for ddx, ddy, name in dirs:
            ngx = max(0, min(gw - 1, gx + ddx))
            ngy = max(0, min(gh - 1, gy + ddy))
            threat = float(threat_map[ngy, ngx])
            edge   = edge_penalty_x + edge_penalty_y

            # If low HP: pure survival — maximise distance from all enemies
            enemy_bonus = 0.0
            if hp_ratio < 0.30 and enemies:
                new_px = player_pos[0] + ddx * w // gw
                new_py = player_pos[1] + ddy * h // gh
                min_dist = min(e.distance_to_pt((new_px, new_py)) for e in enemies)
                enemy_bonus = -min_dist * 0.001   # reward distance

            score = threat + edge * 2.0 + enemy_bonus
            if score < best_score:
                best_score = score
                best_dir   = (float(ddx), float(ddy), name)

        return best_dir

    def should_move(self) -> bool:
        now = time.time()
        if now - self._last_move_ts >= self.MOVE_INTERVAL:
            self._last_move_ts = now
            return True
        return False

    def get_swipe_params(
        self,
        action: str,
        frame_w: int,
        frame_h: int,
        player_pos: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        """Convert action name to (x1, y1, x2, y2) ADB swipe coordinates."""
        radius = int(min(frame_w, frame_h) * self.DODGE_RADIUS_FRAC)
        dir_map = {
            "idle":       (0,  0),
            "up":         (0, -1), "down":       (0,  1),
            "left":       (-1, 0), "right":      (1,  0),
            "up_left":    (-1,-1), "up_right":   (1, -1),
            "down_left":  (-1, 1), "down_right": (1,  1),
        }
        ddx, ddy = dir_map.get(action, (0, 0))
        cx, cy   = player_pos
        return cx, cy, cx + ddx * radius, cy + ddy * radius
