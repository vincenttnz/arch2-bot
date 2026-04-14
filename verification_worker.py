import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
VERIFY_DIR = PROJECT_ROOT / "debug" / "verification"
JUNK_DIR = PROJECT_ROOT / "debug" / "junk"
BRIDGE_URL = "http://localhost:5050"

def get_label_from_bridge(img_path: Path):
    # This prompt tells SenseNova to handle BOTH Skill Crops and Full Combat screens
    prompt = """Analyze this image. 
    1. If it's a skill card, return {'skill_name': 'Name'}. 
    2. If it's combat, return {'skill_name': 'Combat Data', 'detections': [...]}.
    3. If junk, return {'skill_name': 'Junk'}. Return JSON ONLY."""
    
    data = json.dumps({"image_path": str(img_path.resolve()), "prompt": prompt}).encode("utf-8")
    req = urllib.request.Request(BRIDGE_URL, data=data, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError:
        print("!!! BRIDGE OFFLINE !!! Start 'sensenova_bridge.py' to resume.")
        return "OFFLINE"
    except Exception: return None

def watch_directory():
    print(f"Watching {VERIFY_DIR}...")
    while True:
        # Sort to ensure consistent processing
        files = sorted(list(VERIFY_DIR.glob("*.png")) + list(VERIFY_DIR.glob("*.jpg")))
        
        for img_path in files:
            if not img_path.exists(): continue # Race condition safety

            # 1. Context: Split full screenshots into 3 slots
            if "_slot" not in img_path.name:
                try:
                    img = Image.open(img_path)
                    w, h = img.size
                    for i in range(3):
                        crop = img.crop((i * (w//3), 0, (i+1) * (w//3), h))
                        crop.save(VERIFY_DIR / f"{img_path.stem}_slot_{i}.png")
                    img_path.replace(JUNK_DIR / img_path.name)
                except Exception: continue
                continue

            # 2. Labeling slots or combat frames
            json_path = img_path.with_suffix(".json")
            if json_path.exists(): continue

            result = get_label_from_bridge(img_path)
            if result == "OFFLINE":
                time.sleep(5); break
            
            if result:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
                print(f"✓ Processed: {img_path.name} -> {result.get('skill_name')}")

        time.sleep(2)

if __name__ == "__main__":
    watch_directory()