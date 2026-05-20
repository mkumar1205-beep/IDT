import unittest

try:
    import cv2
    import numpy as np

    from detector import detect_damage
except ModuleNotFoundError:
    cv2 = None
    np = None
    detect_damage = None


@unittest.skipIf(detect_damage is None, "OpenCV is not installed")
class DetectorTests(unittest.TestCase):
    def test_dark_water_filled_pothole_is_detected(self):
        image = np.full((220, 340, 3), 155, dtype=np.uint8)

        cv2.ellipse(image, (165, 135), (70, 30), -8, 0, 360, (35, 35, 35), -1)
        cv2.ellipse(image, (168, 138), (42, 18), -8, 0, 360, (215, 215, 210), -1)
        cv2.line(image, (0, 155), (340, 145), (95, 95, 95), 2)

        _, _, stats = detect_damage(image)

        self.assertGreaterEqual(stats["pothole_count"], 1)
        self.assertGreater(stats["damage_area_pct"], 0)


if __name__ == "__main__":
    unittest.main()
