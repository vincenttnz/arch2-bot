import tkinter as tk

class Overlay:
    def __init__(self, parent):
        self.win = tk.Toplevel(parent)
        self.win.title("Archero Radar")
        self.win.geometry("300x300+100+100")
        self.win.attributes("-topmost", True)
        self.win.attributes("-alpha", 0.7)
        self.win.configure(bg="black")
        
        self.label = tk.Label(self.win, text="RADAR ACTIVE", fg="green", bg="black", font=("Arial", 14, "bold"))
        self.label.pack(expand=True)
        
        # Draw a crosshair
        self.canvas = tk.Canvas(self.win, width=300, height=300, bg="black", highlightthickness=0)
        self.canvas.pack()
        self.canvas.create_line(150, 0, 150, 300, fill="green", width=2)
        self.canvas.create_line(0, 150, 300, 150, fill="green", width=2)

    def destroy(self):
        self.win.destroy()

def launch_overlay(parent):
    return Overlay(parent)