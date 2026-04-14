import cv2
import numpy as np

class SenseNovaDetector:
    def __init__(self):
        # We target complex entity classes for better training data
        self.classes = ["enemy", "projectile", "devil", "angel", "valkyrie", "boss"]

    def detect_entities(self, frame):
        """
        Currently acts as a bridge. In the next step, we'll hook 
        the API endpoint here to auto-label your training data.
        """
        # Placeholder for API response
        return []