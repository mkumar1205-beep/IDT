import unittest

from scorer import quality_score


class QualityScoreTests(unittest.TestCase):
    def test_clean_road_is_good(self):
        score, label, _ = quality_score(0, 0.0, {"small": 0, "medium": 0, "large": 0})
        self.assertEqual(label, "Good")
        self.assertGreaterEqual(score, 75)

    def test_any_detected_pothole_is_not_good(self):
        score, label, _ = quality_score(1, 0.4, {"small": 1, "medium": 0, "large": 0})
        self.assertEqual(label, "Moderate")
        self.assertLess(score, 75)

    def test_multiple_potholes_are_poor(self):
        score, label, _ = quality_score(4, 2.0, {"small": 4, "medium": 0, "large": 0})
        self.assertEqual(label, "Poor")
        self.assertLess(score, 45)

    def test_large_pothole_is_poor(self):
        score, label, _ = quality_score(1, 1.0, {"small": 0, "medium": 0, "large": 1})
        self.assertEqual(label, "Poor")
        self.assertLess(score, 45)


if __name__ == "__main__":
    unittest.main()
