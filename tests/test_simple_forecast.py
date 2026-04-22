import os
import unittest

import numpy as np
import tensorflow as tf

from src.simple_rnn_forecast import (
    mae,
    rmse,
    make_windows,
    time_split,
    build_model,
    train_model,
    evaluate_model,
)


class TestMetrics(unittest.TestCase):
    def test_mae(self):
        y_true = np.array([0, 1, 2], dtype=np.float32)
        y_pred = np.array([0, 2, 1], dtype=np.float32)
        self.assertAlmostEqual(mae(y_true, y_pred), 2 / 3, places=6)

    def test_rmse(self):
        y_true = np.array([0, 0], dtype=np.float32)
        y_pred = np.array([3, 4], dtype=np.float32)
        self.assertAlmostEqual(rmse(y_true, y_pred), np.sqrt(12.5), places=6)


class TestWindowingAndSplit(unittest.TestCase):
    def test_make_windows_shapes_and_values(self):
        series = np.arange(10, dtype=np.float32)  # 0..9
        X, y = make_windows(series, window=3)

        self.assertEqual(X.shape, (7, 3, 1))
        self.assertEqual(y.shape, (7, 1))

        np.testing.assert_allclose(X[0, :, 0], [0, 1, 2])
        np.testing.assert_allclose(y[0, 0], 3)

        np.testing.assert_allclose(X[-1, :, 0], [6, 7, 8])
        np.testing.assert_allclose(y[-1, 0], 9)

    def test_time_split_no_shuffle_boundary(self):
        series = np.arange(30, dtype=np.float32)
        X, y = make_windows(series, window=5)  # N=25

        (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = time_split(X, y, 0.6, 0.2)

        self.assertEqual(len(X_tr) + len(X_val) + len(X_te), len(X))
        # Ensure chronological order preserved
        self.assertAlmostEqual(float(y_tr[-1, 0] + 1.0), float(y_val[0, 0]), places=6)
        self.assertAlmostEqual(float(y_val[-1, 0] + 1.0), float(y_te[0, 0]), places=6)


class TestModelAndTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tf.keras.utils.set_random_seed(123)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

    def test_build_model_shapes(self):
        model = build_model(window=20, n_units=8, dense_units=8, dropout=0.0)
        self.assertEqual(model.input_shape, (None, 20, 1))
        self.assertEqual(model.output_shape, (None, 1))
        self.assertIsNotNone(model.optimizer)

    def test_train_model_smoke(self):
        # Different data than demo (different period & seed) to avoid overfitting to demo.
        rng = np.random.default_rng(7)
        t = np.arange(700, dtype=np.float32)
        series = (
            0.001 * t
            + 1.7 * np.sin(2 * np.pi * t / 47.0)
            + 0.6 * np.sin(2 * np.pi * t / 19.0)
            + rng.normal(0, 0.15, size=len(t)).astype(np.float32)
        )

        model, X_test, y_test, history = train_model(
            series, window=30, epochs=2, batch_size=32, seed=123, verbose=0
        )

        self.assertTrue(hasattr(history, "history"))
        self.assertIn("loss", history.history)

        y_pred = model.predict(X_test, verbose=0)
        self.assertEqual(y_pred.shape, y_test.shape)
        self.assertTrue(np.isfinite(y_pred).all())

        metrics = evaluate_model(model, X_test, y_test)
        self.assertIn("mae", metrics)
        self.assertIn("rmse", metrics)
        self.assertTrue(np.isfinite(metrics["mae"]))
        self.assertTrue(np.isfinite(metrics["rmse"]))


class TestOptionalQuality(unittest.TestCase):
    @unittest.skipUnless(os.getenv("RUN_SLOW_TESTS") == "1", "Set RUN_SLOW_TESTS=1 to run")
    def test_model_not_worse_than_baseline(self):
        tf.keras.utils.set_random_seed(7)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

        # Easier series (low noise) for stable learning
        rng = np.random.default_rng(7)
        t = np.arange(1400, dtype=np.float32)
        series = (
            0.001 * t
            + 2.0 * np.sin(2 * np.pi * t / 50.0)
            + 0.8 * np.sin(2 * np.pi * t / 16.0)
            + rng.normal(0, 0.05, size=len(t)).astype(np.float32)
        )

        window = 40
        model, X_test, y_test, _ = train_model(series, window=window, epochs=18, batch_size=64, seed=7, verbose=0)

        # Baseline: last value in the window
        y_base = X_test[:, -1, 0:1]
        rmse_base = rmse(y_test, y_base)

        y_pred = model.predict(X_test, verbose=0)
        rmse_model = rmse(y_test, y_pred)

        self.assertLessEqual(rmse_model, rmse_base)


if __name__ == "__main__":
    unittest.main()
