"""
skill_card_scanner.py
======================
Automatically scans F:\\arch2\\data\\skill_cards\\ (78 k+ images),
extracts skill names from each image using a multi-method pipeline,
and outputs:

  models/skill_hash_db.pkl          — pHash lookup table (instant, no GPU)
  data/skill_card_labels.json       — {filename: skill_name} for every image
  data/skill_train_manifest.csv     — path,label CSV for YOLO classifier
  core/templates/skills/<name>/     — best representative crop per skill

Recognition pipeline (best result wins)
----------------------------------------
  Method 1 — Folder name (authoritative when skill_cards/ uses subfolders)
  Method 2 — Pytesseract OCR on the card title strip (top ~25% of card)
  Method 3 — pHash distance to existing core/templates/skills/ DB
  Method 4 — Filename parsing (skill_TIMESTAMP_n3_s0.jpg → parent-folder name)

After scanning:
  • Calls SkillRecogniser.build_hash_db() to rebuild the pHash DB
  • Writes a training manifest CSV ready for YOLO classifier training
  • Optionally trains the YOLO classifier via the GUI Skills tab
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import re
import shutil
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import pytesseract
    _OCR_OK = True
except ImportError:
    _OCR_OK = False

# ── skill name normalisation ───────────────────────────────────────────────────
_NOISE_WORDS = {"the","of","a","an","and","or","for","in","on","at","to"}

def _normalise(text: str) -> str:
    """
    Convert any skill name variant to the canonical snake_case key.
    'Front Arrow' → 'front_arrow'
    'beam-strike' → 'beam_strike'
    'FAIRY OF THE WING' → 'fairy_of_the_wing'
    """
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9 _-]", "", t)
    t = re.sub(r"[\s\-]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t or "unknown"


def _text_to_skill(text: str, alias_map: dict[str, str]) -> str:
    """Map raw OCR text to a canonical skill key using the alias map."""
    key = _normalise(text)
    if key in alias_map:
        return alias_map[key]
    # Try partial match (first two significant words)
    words = [w for w in key.split("_") if w not in _NOISE_WORDS]
    if len(words) >= 2:
        partial = "_".join(words[:2])
        for alias, canon in alias_map.items():
            if partial in alias:
                return canon
    return key


# ── pHash ─────────────────────────────────────────────────────────────────────

def _dhash(img: np.ndarray, size: int = 16) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    r    = cv2.resize(gray, (size+1, size), interpolation=cv2.INTER_AREA)
    return (r[:, 1:] > r[:, :-1]).astype(np.uint8).flatten()

def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


# ── OCR title-strip reader ────────────────────────────────────────────────────

def _ocr_title(img: np.ndarray) -> str:
    """
    Extract the skill name text from the top strip of a skill card image.
    Returns empty string if OCR unavailable or no text found.
    """
    if not _OCR_OK:
        return ""
    try:
        h, w = img.shape[:2]
        # Title text lives roughly in the top 28% of the card
        strip = img[0:int(h*0.28), :]
        # Upscale for better OCR accuracy
        strip = cv2.resize(strip, (strip.shape[1]*3, strip.shape[0]*3),
                           interpolation=cv2.INTER_CUBIC)
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        # Try both light-on-dark and dark-on-light
        _, t1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        _, t2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        best = ""
        for thresh in (t1, t2):
            cfg = "--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -"
            try:
                txt = pytesseract.image_to_string(thresh, config=cfg).strip()
                if len(txt) > len(best):
                    best = txt
            except Exception:
                pass
        return best
    except Exception:
        return ""


# ── SkillCardScanner ──────────────────────────────────────────────────────────

class SkillCardScanner:
    """
    Scans data/skill_cards/ and builds a complete skill card database.

    Usage
    -----
        scanner = SkillCardScanner(project_root="F:/arch2")
        results = scanner.scan(progress_callback=print)
        # results = {"labels": {...}, "manifest": [...], "stats": {...}}
    """

    def __init__(self, project_root):
        self.root         = Path(project_root)
        self.skill_dir    = self.root / "data" / "skill_cards"
        self.template_dir = self.root / "core" / "templates" / "skills"
        self.models_dir   = self.root / "models"
        self.data_dir     = self.root / "data"
        self.catalog_path = self.root / "skill_catalog.json"

        # Output paths
        self.labels_path   = self.data_dir / "skill_card_labels.json"
        self.manifest_path = self.data_dir / "skill_train_manifest.csv"
        self.db_path       = self.models_dir / "skill_hash_db.pkl"

        # Load alias map from skill_catalog.json
        self.alias_map: dict[str, str] = {}
        self.priority_map: dict[str, int] = {}
        self._load_catalog()

        # Load existing template hashes
        self._template_db: dict[str, list[np.ndarray]] = {}
        self._load_template_db()

    # ── setup ────────────────────────────────────────────────────────────────

    def _load_catalog(self):
        """Load skill_catalog.json for alias matching and priority lookup."""
        if self.catalog_path.exists():
            try:
                d = json.loads(self.catalog_path.read_text())
                self.priority_map = d.get("priorities", {})
                for canon, aliases in d.get("aliases", {}).items():
                    for alias in aliases:
                        self.alias_map[_normalise(alias)] = canon
                    self.alias_map[canon] = canon
                print(f"[Scanner] Loaded catalog: {len(self.priority_map)} skills")
            except Exception as e:
                print(f"[Scanner] Catalog load error: {e}")

        # Always include the names we know from templates
        if self.template_dir.exists():
            for d in self.template_dir.iterdir():
                if d.is_dir():
                    name = d.name
                    self.alias_map[name] = name
                    self.alias_map[name.replace("_", " ")] = name
                    self.alias_map[name.replace("_", "-")] = name

    def _load_template_db(self):
        """Load existing pHash template DB for Method 3 matching."""
        if self.db_path.exists():
            try:
                with open(self.db_path, "rb") as f:
                    self._template_db = pickle.load(f)
                n = sum(len(v) for v in self._template_db.values())
                print(f"[Scanner] Template DB: {len(self._template_db)} skills, {n} hashes")
            except Exception:
                pass

    # ── recognition methods ───────────────────────────────────────────────────

    def _method1_folder(self, img_path: Path) -> tuple[str, float]:
        """If the image lives in a named subfolder, that IS the skill name."""
        parent = img_path.parent.name
        if parent not in {"skill_cards", "skills", ""}:
            name = _normalise(parent)
            mapped = self.alias_map.get(name, name)
            return mapped, 1.0   # authoritative
        return "unknown", 0.0

    def _method2_ocr(self, img: np.ndarray) -> tuple[str, float]:
        """Pytesseract on title strip."""
        txt = _ocr_title(img)
        if len(txt) < 3:
            return "unknown", 0.0
        mapped = _text_to_skill(txt, self.alias_map)
        # Confidence based on text length and alias match quality
        conf = 0.75 if mapped in self.priority_map else 0.45
        return mapped, conf

    def _method3_phash(self, img: np.ndarray) -> tuple[str, float]:
        """pHash distance to template DB."""
        if not self._template_db:
            return "unknown", 0.0
        q = _dhash(img)
        best_name, best_d = "unknown", 9999
        for name, hashes in self._template_db.items():
            for h in hashes:
                d = _hamming(q, h)
                if d < best_d:
                    best_d, best_name = d, name
        conf = max(0.0, 1.0 - best_d / 256)
        return (best_name, conf) if conf > 0.55 else ("unknown", 0.0)

    def _method4_filename(self, img_path: Path) -> tuple[str, float]:
        """
        Parse skill name from filename patterns like:
          skill_1234567890_n3_s0.jpg  →  looks at folder
          ricochet_001.jpg            →  'ricochet'
          combat_1234567890.jpg       →  'combat' (not a skill, ignore)
        """
        stem = img_path.stem.lower()
        # Skip obvious non-skill names
        if stem.startswith(("combat_","menu_","orig_","frame_","aug")):
            return "unknown", 0.0
        # Try: name_digits.jpg  or  name_digits_extra.jpg
        m = re.match(r"^([a-z][a-z_]+?)(?:_\d+.*)?$", stem)
        if m:
            candidate = m.group(1).strip("_")
            if len(candidate) >= 3:
                mapped = self.alias_map.get(candidate, candidate)
                if mapped in self.priority_map:
                    return mapped, 0.60
        return "unknown", 0.0

    def _identify(self, img_path: Path, img: np.ndarray) -> tuple[str, float]:
        """Run all four methods, return highest-confidence result."""
        results = [
            self._method1_folder(img_path),
            self._method3_phash(img),
            self._method2_ocr(img),
            self._method4_filename(img_path),
        ]
        # Sort by confidence descending
        results.sort(key=lambda x: x[1], reverse=True)
        name, conf = results[0]
        return name, conf

    # ── template builder ──────────────────────────────────────────────────────

    def _update_template(self, skill_name: str, img: np.ndarray):
        """
        Keep up to 10 representative images per skill in core/templates/skills/<n>/.
        Uses perceptual diversity: only add if pHash distance > 15 from existing.
        """
        if skill_name in {"unknown", "error", "junk", "none", "start"}:
            return
        tmpl_dir = self.template_dir / skill_name
        tmpl_dir.mkdir(parents=True, exist_ok=True)

        existing = list(tmpl_dir.glob("*.jpg"))
        MAX_TEMPLATES = 12

        if len(existing) >= MAX_TEMPLATES:
            return

        # Load hashes of existing templates
        ex_hashes = []
        for ep in existing:
            ei = cv2.imread(str(ep))
            if ei is not None:
                ex_hashes.append(_dhash(ei))

        q = _dhash(img)
        # Only add if sufficiently different from all existing
        if ex_hashes:
            if min(_hamming(q, h) for h in ex_hashes) < 15:
                return   # too similar, skip

        idx = len(existing)
        cv2.imwrite(str(tmpl_dir / f"{skill_name}_{idx:04d}.jpg"), img)

    # ── main scan ─────────────────────────────────────────────────────────────

    def scan(self, progress_callback=None, max_images: int = 0) -> dict:
        """
        Scan data/skill_cards/, identify every image, write outputs.

        Parameters
        ----------
        progress_callback : callable(str) — receives status lines
        max_images        : int — stop after N images (0 = all)

        Returns
        -------
        dict with keys: labels, manifest, stats
        """
        log = progress_callback or print

        if not self.skill_dir.exists():
            log(f"[Scanner] ❌ skill_cards/ not found: {self.skill_dir}")
            return {}

        # Collect all images
        all_imgs = []
        for ext in ("*.jpg","*.jpeg","*.png"):
            all_imgs.extend(self.skill_dir.rglob(ext))
        all_imgs = sorted(all_imgs)

        total = len(all_imgs)
        if max_images > 0:
            all_imgs = all_imgs[:max_images]

        log(f"[Scanner] Found {total} images in skill_cards/")
        log(f"[Scanner] Processing {len(all_imgs)} images …")

        labels: dict[str, str]       = {}   # rel_path → skill_name
        manifest: list[tuple]        = []   # (abs_path, skill_name)
        conf_map: dict[str, float]   = {}   # rel_path → confidence
        skill_counts: Counter        = Counter()
        low_conf: list               = []   # paths with conf < 0.5

        t0 = time.time()
        for idx, img_path in enumerate(all_imgs):
            if not img_path.is_file():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            name, conf = self._identify(img_path, img)
            rel = str(img_path.relative_to(self.root))

            labels[rel]    = name
            conf_map[rel]  = conf
            skill_counts[name] += 1
            manifest.append((str(img_path), name))

            if conf < 0.5:
                low_conf.append(rel)

            # Build templates for recognised skills
            if conf >= 0.55:
                self._update_template(name, img)

            if (idx+1) % 5000 == 0 or (idx+1) == len(all_imgs):
                elapsed = time.time() - t0
                rate    = (idx+1) / max(elapsed, 0.1)
                log(f"[Scanner]   {idx+1}/{len(all_imgs)} images  "
                    f"({rate:.0f}/s)  "
                    f"skills identified: {len(skill_counts)}")

        # ── write outputs ─────────────────────────────────────────────────────

        # 1. Labels JSON
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.labels_path.write_text(json.dumps({
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_images": len(labels),
            "skills_found": len(skill_counts),
            "low_confidence": len(low_conf),
            "skill_counts": dict(skill_counts.most_common()),
            "labels": labels,
        }, indent=2))
        log(f"[Scanner] ✅ Labels written → {self.labels_path.name}")

        # 2. Training manifest CSV
        with open(self.manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label"])
            for path, label in manifest:
                if label not in {"unknown"}:
                    writer.writerow([path, label])
        log(f"[Scanner] ✅ Manifest written → {self.manifest_path.name}  "
            f"({sum(1 for _,l in manifest if l!='unknown')} labelled rows)")

        # 3. Rebuild pHash DB from templates
        log("[Scanner] Rebuilding pHash hash DB from templates …")
        n_skills = self._rebuild_hash_db()
        log(f"[Scanner] ✅ Hash DB rebuilt → {n_skills} skills")

        # Stats
        stats = {
            "total_scanned":     len(labels),
            "skills_identified": len(skill_counts),
            "low_confidence":    len(low_conf),
            "top_skills":        dict(skill_counts.most_common(15)),
            "unknown_count":     skill_counts.get("unknown", 0),
        }

        log(f"[Scanner] ─── Scan complete ───")
        log(f"[Scanner]   Images scanned  : {len(labels)}")
        log(f"[Scanner]   Unique skills   : {len(skill_counts)}")
        log(f"[Scanner]   Unknown / low-conf: {skill_counts.get('unknown',0)} / {len(low_conf)}")
        log(f"[Scanner]   Top skills      : " +
            "  ".join(f"{k}={v}" for k,v in skill_counts.most_common(5)))

        return {"labels": labels, "manifest": manifest, "stats": stats}

    def _rebuild_hash_db(self) -> int:
        """Scan core/templates/skills/ and write models/skill_hash_db.pkl."""
        if not self.template_dir.exists():
            return 0
        db: dict[str, list[np.ndarray]] = {}
        skip = {"_anchors"}
        for skill_dir in sorted(self.template_dir.iterdir()):
            if not skill_dir.is_dir() or skill_dir.name in skip:
                continue
            hashes = []
            for p in skill_dir.iterdir():
                if p.suffix.lower() in {".jpg",".jpeg",".png"}:
                    img = cv2.imread(str(p))
                    if img is not None:
                        hashes.append(_dhash(img))
            if hashes:
                db[skill_dir.name] = hashes
        self.models_dir.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump(db, f)
        self._template_db = db
        return len(db)

    def train_classifier(self, project_root, epochs: int = 20, batch: int = 32,
                         progress_callback=None) -> bool:
        """
        Train a YOLOv8n-cls classifier on data/skill_cards/ using the manifest.

        The training script is written to data/temp_skill_train.py and executed
        in a subprocess so the GUI log stream works correctly.

        Returns True on success.
        """
        import subprocess, sys as _sys, json as _json

        log = progress_callback or print
        root = Path(project_root)

        skill_data_fwd = str(root/"data"/"skill_cards").replace("\\","/")
        clf_out_fwd    = str(root/"models"/"skill_classifier.pt").replace("\\","/")

        # Count classes
        classes = [d.name for d in (root/"data"/"skill_cards").iterdir()
                   if d.is_dir()] if (root/"data"/"skill_cards").exists() else []
        if not classes:
            log("[Scanner] ❌ No class subdirs in data/skill_cards/ — cannot train")
            return False

        log(f"[Scanner] Training YOLO classifier: {len(classes)} classes, "
            f"{epochs} epochs, batch {batch}")

        script_lines = [
            "import shutil, os, glob",
            "from ultralytics import YOLO",
            "model = YOLO('yolov8n-cls.pt')",
            "model.train(",
            f"    data=r'{skill_data_fwd}',",
            "    task='classify',",
            f"    epochs={epochs},",
            "    imgsz=128,",
            f"    batch={batch},",
            "    device=0,",
            "    workers=4,",
            "    patience=10,",
            "    name='skill_classifier',",
            "    exist_ok=True,",
            ")",
            "pts = sorted(glob.glob('runs/classify/skill_classifier*/weights/best.pt'))",
            "if pts:",
            f"    os.makedirs(os.path.dirname(r'{clf_out_fwd}'), exist_ok=True)",
            f"    shutil.copy(pts[-1], r'{clf_out_fwd}')",
            f"    print('Saved to: ' + r'{clf_out_fwd}')",
            "print('Training complete.')",
        ]

        script_path = str(root/"data"/"temp_skill_train.py")
        with open(script_path,"w",encoding="utf-8") as sf:
            sf.write("\n".join(script_lines)+"\n")

        proc = subprocess.Popen(
            [_sys.executable, script_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, universal_newlines=True,
        )
        for line in iter(proc.stdout.readline, ""):
            stripped = line.rstrip()
            if stripped: log(f"[YOLO] {stripped}")
        proc.stdout.close()
        proc.wait()

        try: os.remove(script_path)
        except: pass

        if proc.returncode == 0:
            log("[Scanner] ✅ Classifier training complete")
            return True
        log(f"[Scanner] ❌ Training failed (exit {proc.returncode})")
        return False
