import unittest
import numpy as np
from src.model_components import NumericStrategy, compute_mild_class_weights

class TestModelComponents(unittest.TestCase):
    def test_raw_strategy_replaces_non_finite_values(self):
        X = np.array([[1.0, np.nan], [3.0, 2.0], [5.0, np.inf]])
        transformed = NumericStrategy("S0_Raw").fit_transform(X)
        self.assertTrue(np.isfinite(transformed).all())

    def test_clipping_strategy_bounds_extreme_values(self):
        X = np.arange(100, dtype=float).reshape(-1, 1)
        transformer = NumericStrategy("S1_Clipping").fit(X)
        result = transformer.transform(np.array([[-1000.0], [1000.0]]))
        self.assertGreaterEqual(result.min(), transformer.lower_[0])
        self.assertLessEqual(result.max(), transformer.upper_[0])

    def test_signed_log_preserves_sign(self):
        X = np.array([[-10.0], [0.0], [10.0]])
        result = NumericStrategy("S2_SignedLog", zero_threshold=0.1).fit_transform(X)
        self.assertLess(result[0, 0], 0)
        self.assertEqual(result[1, 0], 0)
        self.assertGreater(result[2, 0], 0)

    def test_unknown_strategy_is_rejected(self):
        transformer = NumericStrategy("UNKNOWN").fit(np.array([[1.0], [2.0]]))
        with self.assertRaises(ValueError):
            transformer.transform(np.array([[1.5]]))

    def test_mild_weights_are_finite_positive_and_mean_one(self):
        y = np.array(["Normal"] * 90 + ["DoS"] * 9 + ["Worms"])
        weights = compute_mild_class_weights(y)
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue((weights > 0).all())
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertGreater(weights[-1], weights[0])

if __name__ == "__main__":
    unittest.main()
