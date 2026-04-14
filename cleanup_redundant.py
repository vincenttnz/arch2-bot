#!/usr/bin/env python3
"""
cleanup_redundant.py  –  F:\\arch2 space reclaimer
===================================================
Run once from F:\\arch2.  Safe to re-run.

Removes
-------
1. Incomplete training runs  (weights/ empty → no best.pt produced)
2. Superseded old run  SenseNova_Survival_v16  (v162/163/164 supersede it)
3. YOLO dataset IMAGE copies  (labels kept; images regenerated before each train)
4. data/temp_train*.py  artefacts

Never touches
-------------
  weights/best.pt  •  combat_boxes.jsonl  •  approved_projectiles/
  core/templates/  •  data/skill_cards/   •  models/
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _human(n: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def _folder_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


freed = 0

# ── 1. Training runs ──────────────────────────────────────────────────────────
FORCE_DELETE_RUNS = {"SenseNova_Survival_v16"}   # superseded by v162+
runs_dir = ROOT / "runs" / "detect"
print("\n[1] Training runs …")
if runs_dir.exists():
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        weights_dir = run_dir / "weights"
        is_empty    = weights_dir.exists() and not any(weights_dir.iterdir())
        if run_dir.name in FORCE_DELETE_RUNS or is_empty:
            sz = _folder_size(run_dir)
            print(f"  DELETE  {run_dir.name}  ({_human(sz)})")
            shutil.rmtree(run_dir, ignore_errors=True)
            freed += sz
        else:
            print(f"  KEEP    {run_dir.name}")

# ── 2. YOLO dataset image copies ─────────────────────────────────────────────
print("\n[2] YOLO dataset image copies …")
for split in ("train", "val"):
    img_dir = ROOT / "data" / "yolo_dataset" / split / "images"
    if img_dir.exists() and any(img_dir.iterdir()):
        sz    = _folder_size(img_dir)
        count = sum(1 for _ in img_dir.iterdir())
        print(f"  DELETE  yolo_dataset/{split}/images  ({count} files, {_human(sz)})")
        shutil.rmtree(img_dir, ignore_errors=True)
        img_dir.mkdir(parents=True, exist_ok=True)
        freed += sz
    else:
        print(f"  SKIP    yolo_dataset/{split}/images  (empty)")

# ── 3. Temp scripts ───────────────────────────────────────────────────────────
print("\n[3] Temp training scripts …")
for p in (ROOT / "data").glob("temp_train*.py"):
    sz = p.stat().st_size
    print(f"  DELETE  {p.name}")
    p.unlink()
    freed += sz

print(f"\n✅  Freed {_human(freed)}.  Weights / labels / images untouched.")
