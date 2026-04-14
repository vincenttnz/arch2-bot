import os
import json
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
LABELED_DIR  = PROJECT_ROOT / "data" / "labeled_frames"
SKILLS_DIR   = PROJECT_ROOT / "core" / "templates" / "skills"
JUNK_DIR     = PROJECT_ROOT / "debug" / "junk"
MODELS_DIR   = PROJECT_ROOT / "models"

def scan_dataset():
    print("="*50)
    print(" ARCHERO 2 NEURAL DATASET SCANNER ")
    print("="*50)

    # 1. Combat Data (YOLO / RL)
    json_count = len(list(LABELED_DIR.glob("*.json")))
    img_count  = len(list(LABELED_DIR.glob("*.png"))) + len(list(LABELED_DIR.glob("*.jpg")))
    print(f"[COMBAT DETECTOR]")
    print(f"  - Frames Labeled (JSON): {json_count}")
    print(f"  - Matching Images:       {img_count}")
    print(f"  - Readiness:             {'READY' if json_count > 500 else 'COLLECTING'}")
    print("-" * 30)

    # 2. Skill Classification Data
    skill_folders = [d for d in SKILLS_DIR.iterdir() if d.is_dir()]
    total_skill_imgs = sum(len(list(d.glob("*.png"))) for d in skill_folders)
    print(f"[SKILL CLASSIFIER]")
    print(f"  - Unique Skill Classes:  {len(skill_folders)}")
    print(f"  - Total Skill Templates: {total_skill_imgs}")
    print(f"  - Avg Imgs per Skill:    {total_skill_imgs / max(1, len(skill_folders)):.1f}")
    print(f"  - Readiness:             {'READY' if total_skill_imgs > 200 else 'COLLECTING'}")
    print("-" * 30)

    # 3. Screen Classification Data
    junk_count = len(list(JUNK_DIR.glob("*.png")))
    total_screen_data = img_count + total_skill_imgs + junk_count
    print(f"[SCREEN CLASSIFIER]")
    print(f"  - Combat Examples:       {img_count}")
    print(f"  - Menu/Junk Examples:    {junk_count}")
    print(f"  - Readiness:             {'READY' if total_screen_data > 1000 else 'COLLECTING'}")
    print("-" * 30)

    # 4. Model Status
    trained_models = [f.name for f in MODELS_DIR.glob("*.pt")]
    print(f"[TRAINED MODELS]")
    print(f"  - Found: {', '.join(trained_models) if trained_models else 'None'}")
    print("="*50)

if __name__ == "__main__":
    scan_dataset()