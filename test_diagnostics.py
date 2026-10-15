"""Focused guardrail and normalization tests; no Galaxy10 download required."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import diagnose_models as d


class DiagnosticTests(unittest.TestCase):
    def test_ten_class_metrics_include_unpredicted_classes(self):
        y = np.arange(10)
        p = np.zeros((10, 10))
        p[:, 0] = 1
        measured = d.measure(y, p)
        self.assertAlmostEqual(measured["accuracy"], .1)
        self.assertAlmostEqual(measured["macro_f1"], (2/11)/10)
        self.assertEqual(np.asarray(measured["confusion_matrix"]).sum(), 10)
        self.assertTrue(np.isfinite(measured["cross_entropy"]))

    def test_invalid_probability_rejected(self):
        with self.assertRaises(ValueError):
            d.measure([0], np.zeros((1, 10)))

    def test_smoke_run_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "RUN_COMPLETE.json").write_text('{"smoke": true}')
            (root / "locked_protocol.json").write_text('{"smoke": true}')
            with self.assertRaisesRegex(ValueError, "Smoke"):
                d.resolve_run(root, root)

    def test_trajectory_rejects_validation_or_duplicate_queries(self):
        for selected in ([2], [0, 0]):
            with self.subTest(selected=selected), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / "trajectory_complete.json").write_text('{}')
                (root / "round_000.json").write_text(json.dumps({
                    "seed": 42, "strategy": "random", "round": 0,
                    "queried_positions": selected}))
                with self.assertRaisesRegex(ValueError, "Non-training"):
                    d.final_record(root, 42, "random", np.asarray([0, 1]),
                                   np.asarray([10, 11, 12]), np.asarray([0, 1, 2]), 1)

    def test_checkpoint_tamper_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "model.pt"
            file.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "Checkpoint changed"):
                d.mlp_predict(file, hashlib.sha256(b"original").hexdigest(), np.zeros((1, 512)))

    def test_linear_export_matches_predictions_and_train_scaler(self):
        from threadpoolctl import threadpool_limits
        rng = np.random.default_rng(4)
        y = np.tile(np.arange(10), 5)
        x = rng.normal(size=(50, 12)) + np.eye(10, 12)[y]*2
        vx = rng.normal(size=(20, 12)) + 20
        vy = np.tile(np.arange(10), 2)
        with tempfile.TemporaryDirectory() as folder, threadpool_limits(limits=1):
            out = Path(folder)
            pt, pv, chosen = d.fit_linear(x, y, vx, vy, out)
            model = np.load(out / "linear_model.npz")
            np.testing.assert_allclose(model["mean"], x.mean(axis=0))
            logits = ((vx - model["mean"]) / model["scale"]) @ model["coefficients"].T + model["intercept"]
            logits -= logits.max(axis=1, keepdims=True)
            expected = np.exp(logits)
            expected /= expected.sum(axis=1, keepdims=True)
            np.testing.assert_allclose(expected, pv, atol=1e-12)
            self.assertEqual(pt.shape, (50, 10))
            self.assertIn(chosen["C"], d.C_GRID)
            self.assertFalse(d.read_json(out / "linear_selection.json")["test_used"])


if __name__ == "__main__":
    unittest.main()
