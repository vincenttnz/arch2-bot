"""
auto_triage.py — SenseNova AI Automated Snapshot Filter
=======================================================
Reads raw screenshots from 'original_snapshots' and uses the 
latest trained YOLO weights to categorize them into Combat or Skill/Menu scenes.
"""
import os
import shutil
import time
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ CRITICAL: Ultralytics YOLO is not installed.")
    exit(1)

# --- Configuration & Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"

INPUT_DIR    = DATA_DIR / "original_snapshots"
COMBAT_DIR   = DATA_DIR / "high_priority_projectiles" # Feeds into Tab 2 Labeler
SKILL_DIR    = DATA_DIR / "skill_screens_inbox"       # Isolated for Skill OCR/Training

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
COMBAT_DIR.mkdir(parents=True, exist_ok=True)
SKILL_DIR.mkdir(parents=True, exist_ok=True)

def find_latest_weights() -> str:
    """Scans the runs/detect/ folder for the newest trained model."""
    runs_dir = PROJECT_ROOT / "runs" / "detect"
    if not runs_dir.exists():
        return None
    
    candidates = sorted(
        runs_dir.rglob("best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    return str(candidates[0]) if candidates else None

def triage_snapshots():
    print("🚀 SenseNova AI Triage Engine Initialized...")
    
    files = sorted([
        f for f in os.listdir(INPUT_DIR) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    total = len(files)
    if total == 0:
        print("ℹ️ No snapshots found in 'data/original_snapshots/'.")
        return

    print(f"📂 Found {total} raw snapshots. Loading AI Brain...")
    
    weights = find_latest_weights()
    if not weights:
        print("❌ No trained YOLO weights found. Please train the model first.")
        return
        
    print(f"🧠 Engaging Neural Net: {Path(weights).parent.parent.name}")
    model = YOLO(weights).to('cuda') # Utilizes your RTX 3060

    combat_count = 0
    skill_count = 0

    for idx, fname in enumerate(files, 1):
        src_path = INPUT_DIR / fname
        
        # Run fast inference on the RTX 3060
        results = model.predict(str(src_path), conf=0.30, verbose=False)
        
        # If the AI finds ANY combat bounding boxes, it's a combat scene
        if len(results[0].boxes) > 0:
            dest_path = COMBAT_DIR / fname
            combat_count += 1
            status = "⚔️ COMBAT"
        else:
            # If the screen lacks combat entities, it's a Skill/Menu screen
            dest_path = SKILL_DIR / fname
            skill_count += 1
            status = "🃏 SKILL "

        try:
            shutil.move(str(src_path), str(dest_path))
            print(f"[{idx}/{total}] Routed -> {status} | {fname}")
        except Exception as e:
            print(f"⚠️ Failed to move {fname}: {e}")

    print("\n" + "="*40)
    print("🏁 TRIAGE COMPLETE")
    print(f"⚔️ Combat Frames Moved: {combat_count} (Ready for Tab 2 Labeling)")
    print(f"🃏 Skill Frames Moved:  {skill_count} (Isolated for UI Training)")
    print("="*40)

if __name__ == "__main__":
    triage_snapshots()