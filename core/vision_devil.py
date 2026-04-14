import cv2
import numpy as np


class DevilDetector:
    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower1 = np.array([0, 120, 120])
        upper1 = np.array([10, 255, 255])

        lower2 = np.array([170, 120, 120])
        upper2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = mask1 + mask2

        red_pixels = np.sum(mask > 0)
        print(f"[DEVIL] red_pixels: {red_pixels}")

        if red_pixels > 8000:
            print("[DEVIL] DETECTED")
            return True
        return False