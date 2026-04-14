import tkinter as tk
from pathlib import Path

class Overlay:
    def __init__(self, project_root):
        self.root = tk.Toplevel()
        self.root.title("Bot POV Overlay")
        self.root.geometry("300x200")
        self.root.attributes("-topmost", True, "-alpha", 0.7)
        self.root.configure(bg='black')
        
        tk.Label(self.root, text="ARCHERO RADAR ACTIVE", fg="green", bg="black", font=("Arial", 10, "bold")).pack(pady=10)
        self.status = tk.Label(self.root, text="Waiting for coordinates...", fg="white", bg="black")
        self.status.pack()

    def update_info(self, x, y, state):
        self.status.config(text=f"State: {state}\nPos: {x}, {y}")