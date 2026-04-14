
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any


class TemplateSkillImporter:
    def __init__(self, project_root: str | Path, db_path: str | Path | None = None,
                 skills_dir: str | Path | None = None, catalog_path: str | Path | None = None) -> None:
        self.project_root = Path(project_root).resolve()
        self.skills_dir = Path(skills_dir) if skills_dir else self.project_root / "core" / "templates" / "skills"
        self.db_path = Path(db_path) if db_path else self.project_root / "archero2_skill_database.json"
        self.catalog_path = Path(catalog_path) if catalog_path else self.skills_dir / "skill_catalog.json"

    def import_flat_pngs(self) -> dict[str, Any]:
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        flat_pngs = [p for p in sorted(self.skills_dir.glob("*.png")) if p.is_file()]
        db = self._load_json(self.db_path, {"source": "local", "skills": []})
        catalog = self._load_json(self.catalog_path, {"priorities": {}, "aliases": {}})
        imported = []
        skipped = []
        for png_path in flat_pngs:
            skill_key = self._normalize_skill_key(png_path.stem)
            if not skill_key or skill_key.isdigit():
                skipped.append(png_path.name)
                continue
            dest_dir = self.skills_dir / skill_key
            dest_dir.mkdir(parents=True, exist_ok=True)
            next_idx = self._next_index(dest_dir)
            dest_path = dest_dir / f"{next_idx:02d}.png"
            shutil.copy2(png_path, dest_path)
            pretty_name = self._pretty_name(skill_key)
            self._upsert_db_skill(db, pretty_name)
            self._upsert_catalog_skill(catalog, skill_key, pretty_name)
            imported.append({"source_png": png_path.name, "skill_key": skill_key, "dest_png": str(dest_path)})
        self.db_path.write_text(json.dumps(db, indent=2, ensure_ascii=False), encoding="utf-8")
        self.catalog_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"imported_count": len(imported), "imported": imported, "skipped": skipped}

    @staticmethod
    def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
        if not path.exists():
            return dict(default)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return dict(default)

    @staticmethod
    def _normalize_skill_key(stem: str) -> str:
        text = stem.strip().lower()
        text = re.sub(r"^(skill_|icon_|template_)+", "", text)
        text = re.sub(r"[^a-z0-9]+", "_", text)
        return re.sub(r"_+", "_", text).strip("_")

    @staticmethod
    def _pretty_name(skill_key: str) -> str:
        return " ".join(part.capitalize() for part in skill_key.split("_") if part)

    @staticmethod
    def _next_index(dest_dir: Path) -> int:
        nums = []
        for p in dest_dir.glob("*.png"):
            try:
                nums.append(int(p.stem))
            except Exception:
                continue
        return max(nums) + 1 if nums else 1

    @staticmethod
    def _upsert_db_skill(db: dict[str, Any], pretty_name: str) -> None:
        skills = db.setdefault("skills", [])
        if not any(str(s.get("name", "")).strip().lower() == pretty_name.lower() for s in skills):
            skills.append({
                "name": pretty_name,
                "tier": "Template",
                "rarity": "Template",
                "category": "imported_template",
                "priority": 7,
                "enabled": True,
                "notes": "Imported from PNG template crop",
            })

    @staticmethod
    def _upsert_catalog_skill(catalog: dict[str, Any], skill_key: str, pretty_name: str) -> None:
        priorities = catalog.setdefault("priorities", {})
        aliases = catalog.setdefault("aliases", {})
        priorities.setdefault(skill_key, 7)
        alias_vals = aliases.get(skill_key, [])
        if not isinstance(alias_vals, list):
            alias_vals = [str(alias_vals)]
        alias_vals.extend([pretty_name.lower(), pretty_name.lower().replace(" ", "_"), pretty_name.lower().replace(" ", "-")])
        aliases[skill_key] = sorted(set(alias_vals))
