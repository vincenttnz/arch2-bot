"""
validate_structure.py
=====================
Generates a clean visual tree of your repository structure, displaying 
file counts per folder and total disk usage to help validate your 
SenseNova bot architecture and dataset integrity.

Exports a text log in the target directory.
"""
import os
import sys
from pathlib import Path
from datetime import datetime

# --- Configuration ---
TARGET_DIR = r"F:\arch2"

# Folders to completely ignore so they don't clutter the output
IGNORE_DIRS = {'.git', '__pycache__', '.vs', '.idea', 'venv', 'env'}

# Global file handle for export
_export_file = None

def log_print(*args, **kwargs):
    """Print to console and also write to export file."""
    print(*args, **kwargs)
    if _export_file:
        # Convert arguments to string and write
        msg = " ".join(str(arg) for arg in args)
        _export_file.write(msg + "\n")
        _export_file.flush()

def get_dir_size(path: Path) -> int:
    """Calculates the total size of a directory in bytes."""
    total_size = 0
    try:
        for p in path.rglob('*'):
            if p.is_file():
                total_size += p.stat().st_size
    except Exception:
        pass
    return total_size

def format_size(size_in_bytes: int) -> str:
    """Converts bytes to a human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.1f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.1f} PB"

def print_tree(dir_path: Path, prefix: str = ""):
    """Recursively prints the directory tree with file counts."""
    try:
        # Sort items: Directories first, then alphabetically
        items = sorted(dir_path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
    except PermissionError:
        return

    dirs = [item for item in items if item.is_dir() and item.name not in IGNORE_DIRS]
    files = [item for item in items if item.is_file()]
    
    for i, d in enumerate(dirs):
        is_last = (i == len(dirs) - 1)
        connector = "└── " if is_last else "├── "
        
        # Count direct files in this subdirectory
        try:
            sub_files = [f for f in d.iterdir() if f.is_file()]
            sub_count = len(sub_files)
        except Exception:
            sub_count = 0
            
        log_print(f"{prefix}{connector}📂 {d.name}/ ({sub_count} files)")
        
        new_prefix = prefix + ("    " if is_last else "│   ")
        print_tree(d, new_prefix)

def main():
    global _export_file
    root_path = Path(TARGET_DIR)
    
    if not root_path.exists() or not root_path.is_dir():
        log_print(f"❌ CRITICAL: The directory {TARGET_DIR} does not exist.")
        return

    # Create export file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_filename = root_path / f"repo_structure_export_{timestamp}.txt"
    _export_file = open(export_filename, "w", encoding="utf-8")

    log_print("=" * 60)
    log_print(f"🔍 SENSENOVA REPOSITORY VALIDATION")
    log_print(f"📂 Target: {TARGET_DIR}")
    log_print(f"📄 Export file: {export_filename}")
    log_print("=" * 60)
    
    # Print root files count
    root_files = [f for f in root_path.iterdir() if f.is_file()]
    log_print(f"\n📂 {root_path.name}/ ({len(root_files)} files at root level)")
    
    # Generate Tree
    print_tree(root_path)
    
    log_print("\n" + "=" * 60)
    
    # Calculate grand totals
    total_files = sum(1 for _ in root_path.rglob('*') if _.is_file() and not any(part in IGNORE_DIRS for part in _.parts))
    total_dirs = sum(1 for _ in root_path.rglob('*') if _.is_dir() and not any(part in IGNORE_DIRS for part in _.parts))
    total_size = get_dir_size(root_path)
    
    log_print(f"📊 SYSTEM SUMMARY")
    log_print(f"   Total Directories : {total_dirs}")
    log_print(f"   Total Files       : {total_files}")
    log_print(f"   Total Disk Usage  : {format_size(total_size)}")
    log_print("=" * 60)
    log_print(f"✅ Export complete: {export_filename}")

    # Close export file
    _export_file.close()
    _export_file = None

if __name__ == "__main__":
    main()