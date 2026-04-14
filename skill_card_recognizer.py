import os
import json
from PIL import Image
import imagehash
from pathlib import Path

class SkillCardRecognizer:
    def __init__(self, db_path="data/skill_card_hashes.json"):
        self.db_path = db_path
        self.hash_to_skill = {}
        self.load()

    def build_database(self, skills_root="core/templates/skills"):
        """Scan skill folders, compute average hash for each image, store in JSON."""
        skills_root = Path(skills_root)
        if not skills_root.exists():
            print(f"❌ Skill root not found: {skills_root}")
            return
        hash_map = {}
        for skill_folder in skills_root.iterdir():
            if not skill_folder.is_dir():
                continue
            skill_name = skill_folder.name
            for img_path in skill_folder.glob("*.[jp][pn]g"):
                try:
                    img = Image.open(img_path).convert("RGB")
                    # Resize to a fixed size for consistent hashing
                    img = img.resize((128, 128))
                    phash = str(imagehash.phash(img))
                    # Store multiple hashes per skill (all variations)
                    if phash not in hash_map:
                        hash_map[phash] = skill_name
                    else:
                        # If conflict, keep the first; could log warning
                        pass
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        with open(self.db_path, "w") as f:
            json.dump(hash_map, f)
        print(f"✅ Built skill hash database: {len(hash_map)} entries -> {self.db_path}")

    def load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "r") as f:
                self.hash_to_skill = json.load(f)
            print(f"Loaded {len(self.hash_to_skill)} skill hashes.")
        else:
            print("No skill hash database found. Run build_database() first.")

    def recognize(self, card_image):
        """card_image: PIL Image or numpy array (BGR). Returns skill name or None."""
        try:
            if isinstance(card_image, np.ndarray):
                card_image = Image.fromarray(cv2.cvtColor(card_image, cv2.COLOR_BGR2RGB))
            card_image = card_image.resize((128, 128))
            phash = str(imagehash.phash(card_image))
            return self.hash_to_skill.get(phash)
        except:
            return None