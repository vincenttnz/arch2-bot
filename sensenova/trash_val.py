import os
import random
import cv2
import shutil
import matplotlib.pyplot as plt

def validate_and_rescue(trash_path, rescue_path, num_samples=10):
    if not os.path.exists(trash_path): return
    files = [f for f in os.listdir(trash_path) if f.lower().endswith(('.jpg', '.png'))]
    if not files:
        print("🎉 Trash is empty!")
        return

    samples = random.sample(files, min(len(files), num_samples))
    plt.figure(figsize=(15, 8))
    
    for i, filename in enumerate(samples):
        img = cv2.imread(os.path.join(trash_path, filename))
        if img is not None:
            plt.subplot(2, 5, i + 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f"ID: {i}\n{filename[:15]}", fontsize=9)
            plt.axis('off')

    plt.tight_layout()
    print("🧐 Look at the grid. Are there any good images?")
    print("Type the IDs you want to RESCUE (e.g., 0 3 7) or press ENTER to skip:")
    
    # Show plot non-blockingly so we can take input
    plt.show(block=False)
    user_input = input(">> ").split()
    plt.close()

    if user_input:
        os.makedirs(rescue_path, exist_ok=True)
        for idx_str in user_input:
            try:
                idx = int(idx_str)
                filename = samples[idx]
                shutil.move(os.path.join(trash_path, filename), os.path.join(rescue_path, filename))
                print(f"✅ Rescued {filename} to {rescue_path}")
            except:
                print(f"⚠️ Invalid ID: {idx_str}")

if __name__ == "__main__":
    BASE = r"C:\Users\Vince\Desktop\arch2\data"
    TRASH = os.path.join(BASE, "labeled_data", "trash")
    RAW = os.path.join(BASE, "raw_frames")
    
    validate_and_rescue(TRASH, RAW)