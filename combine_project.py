import os
import re
from pathlib import Path

def combine_files():
    project_root = Path(__file__).resolve().parent
    
    file_order = [
        "adb_controller.py",
        "ppo_model.py",
        "rl_agent.py",
        "vision_skill.py",
        "game_state.py",
        "environment.py",
        "template_skill_importer.py",
        "skill_database_gui.py",
        "debug_ui.py",
        "debug_overlay.py",
        "bot_loop.py",
        "gui.py",
        "main.py"
    ]

    # Notice the future import is now at the absolute top!
    combined_code = [
        "from __future__ import annotations",
        "\"\"\"",
        "Archero 2 Bot - All-In-One (AIO) Compiled Script",
        "\"\"\"",
        "import os",
        "import sys",
        "import time",
        "import json",
        "import queue",
        "import threading",
        "import tkinter as tk",
        "from tkinter import ttk",
        "from pathlib import Path",
        "from typing import List, Dict, Any, Optional, Tuple",
        "from dataclasses import dataclass, asdict",
        "import cv2",
        "import numpy as np",
        "import torch",
        "import torch.nn as nn",
        "import torch.nn.functional as F",
        "from torch.distributions import Categorical",
        "\n# --- COMPILED CODE BEGINS --- \n"
    ]

    local_modules = [f.replace(".py", "") for f in file_order]

    for filename in file_order:
        filepath = project_root / filename
        if not filepath.exists():
            print(f"Skipping {filename} (Not found)")
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        combined_code.append(f"\n\n# {'='*60}\n# FILE: {filename}\n# {'='*60}\n")
        
        for line in lines:
            # THIS FIXES THE CRASH: Strip out all future imports from sub-files!
            if "from __future__" in line:
                continue

            if line.startswith("import ") or line.startswith("from "):
                is_local_import = any(mod in line for mod in local_modules)
                if is_local_import or "cv2" in line or "numpy" in line or "torch" in line or "tkinter" in line:
                    continue 
                    
            combined_code.append(line.rstrip())

    output_file = project_root / "archero_bot_aio.py"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(combined_code))

    print(f"✅ Success! Combined {len(file_order)} files into: {output_file.name}")

if __name__ == "__main__":
    combine_files()