import cv2
import numpy as np


class AngelDetector:
    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.array([40, 100, 100])
        upper = np.array([80, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)
        green_pixels = np.sum(mask > 0)

        print(f"[ANGEL] green_pixels: {green_pixels}")

        if green_pixels > 15000:
            print("[ANGEL] DETECTED")
            return True
        return False