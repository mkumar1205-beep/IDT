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
    def test_smooth_clean_road_has_no_damage(self):
        image = np.full((220, 340, 3), 140, dtype=np.uint8)
        image[:70, :] = (190, 190, 190)
        cv2.rectangle(image, (0, 80), (340, 220), (130, 130, 130), -1)

        _, _, stats = detect_damage(image)

        self.assertEqual(stats["pothole_count"], 0)
        self.assertEqual(stats["damage_area_pct"], 0.0)

    def test_dark_water_filled_pothole_is_detected(self):
        image = np.full((220, 340, 3), 155, dtype=np.uint8)

        cv2.ellipse(image, (165, 135), (70, 30), -8, 0, 360, (35, 35, 35), -1)
        cv2.ellipse(image, (168, 138), (42, 18), -8, 0, 360, (215, 215, 210), -1)
        cv2.line(image, (0, 155), (340, 145), (95, 95, 95), 2)

        _, _, stats = detect_damage(image)

        self.assertGreaterEqual(stats["pothole_count"], 1)
        self.assertGreater(stats["damage_area_pct"], 0)

    def test_patchy_broken_road_is_detected(self):
        image = np.full((240, 360, 3), 135, dtype=np.uint8)
        image[:80, :] = (190, 190, 190)

        cv2.rectangle(image, (0, 95), (360, 240), (120, 120, 115), -1)
        for x, y, rw, rh in [
            (45, 150, 56, 18),
            (82, 178, 42, 20),
            (145, 158, 68, 22),
            (205, 188, 60, 18),
            (275, 170, 46, 16),
        ]:
            cv2.ellipse(image, (x, y), (rw, rh), -10, 0, 360, (220, 220, 215), -1)
            cv2.ellipse(image, (x - 3, y + 1), (rw + 10, rh + 5), -10, 0, 360, (60, 60, 58), 3)

        _, _, stats = detect_damage(image)

        self.assertGreaterEqual(stats["pothole_count"], 2)
        self.assertGreater(stats["damage_area_pct"], 0)


if __name__ == "__main__":
    unittest.main()
