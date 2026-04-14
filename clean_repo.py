import os
import shutil
from pathlib import Path
from collections import defaultdict
import imagehash
from PIL import Image

def remove_empty_dirs(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
                    print(f"Removed empty dir: {dir_path}")
            except:
                pass

def delete_failed_training_runs(runs_dir):
    for run_dir in Path(runs_dir).iterdir():
        if run_dir.is_dir() and run_dir.name.startswith("SenseNova_Survival_ft_"):
            weights_dir = run_dir / "weights"
            if not (weights_dir / "best.pt").exists():
                shutil.rmtree(run_dir)
                print(f"Deleted failed run: {run_dir}")

def deduplicate_skill_images(skills_root):
    hash_map = defaultdict(list)
    for skill_folder in Path(skills_root).iterdir():
        if not skill_folder.is_dir():
            continue
        for img_path in skill_folder.glob("*.[jp][pn]g"):
            try:
                img = Image.open(img_path).convert("RGB")
                phash = str(imagehash.phash(img.resize((128,128))))
                hash_map[phash].append(img_path)
            except:
                pass
    for phash, paths in hash_map.items():
        if len(paths) > 1:
            # Keep the first, delete the rest
            for p in paths[1:]:
                p.unlink()
                print(f"Deleted duplicate: {p}")

def main():
    base = Path(".")
    print("🧹 Cleaning repository...")
    
    # 1. Remove empty directories
    remove_empty_dirs(base)
    
    # 2. Delete failed training runs
    delete_failed_training_runs(base / "runs" / "detect")
    
    # 3. Deduplicate skill card images
    skills_path = base / "core" / "templates" / "skills"
    if skills_path.exists():
        deduplicate_skill_images(skills_path)
    
    # 4. Remove duplicate virtual environments (keep only root .venv)
    for venv in base.glob("**/.venv"):
        if venv != base / ".venv":
            shutil.rmtree(venv)
            print(f"Removed duplicate venv: {venv}")
    
    # 5. Delete old log files (older than 7 days)
    # (optional)
    print("✅ Cleanup complete.")

if __name__ == "__main__":
    main()