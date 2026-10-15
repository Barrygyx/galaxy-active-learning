"""Meaningful early-stopping, checkpoint-restoration and label isolation tests."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import run_head_ablation as a


class AblationTests(unittest.TestCase):
    def test_loss_guard_stops_despite_increasing_f1(self):
        stopper = a.EarlyStop(patience=3)
        reasons = [stopper.update(f, loss) for f, loss in [(0.5, 1), (.51, 1.01), (.52, 1.02), (.53, 1.03)]]
        self.assertEqual(reasons[:3], [None]*3)
        self.assertEqual(reasons[3], "validation_loss_no_meaningful_improvement")

    def test_f1_stagnation_stops_despite_decreasing_loss(self):
        stopper = a.EarlyStop(patience=3)
        reasons = [stopper.update(.5, loss) for loss in [1, .9, .8, .7]]
        self.assertEqual(reasons[-1], "validation_f1_no_meaningful_improvement")

    def test_small_improvements_can_accumulate(self):
        stopper = a.EarlyStop(patience=3)
        stopper.update(.5, 1)
        stopper.update(.5006, .9)
        stopper.update(.5012, .8)
        self.assertEqual(stopper.f1_stale, 0)

    def test_best_checkpoint_restored_and_train_only_normalization(self):
        import torch
        from threadpoolctl import threadpool_limits
        torch.set_num_threads(1)
        rng = np.random.default_rng(23)
        x = rng.normal(size=(80, 16)).astype(np.float32)
        y = np.tile(np.arange(10), 8)
        vx = (rng.normal(size=(30, 16)) + 3).astype(np.float32)
        vy = np.tile(np.arange(10), 3)
        settings = {"hidden_dim": 8, "dropout": .1, "weight_decay": .001,
                    "learning_rate": .01, "batch_size": 20, "max_epochs": 12,
                    "patience": 2, "f1_delta": .001, "loss_delta": .0001}
        with threadpool_limits(limits=1):
            cp, pt, pv, history, status = a.train_mlp(x, y, vx, vy, settings, 100)
        best = max(history, key=lambda r: (r["validation_macro_f1"], -r["validation_loss"]))
        self.assertEqual(cp["best_epoch"], best["epoch"])
        self.assertAlmostEqual(a.io.measure(vy, pv)["macro_f1"], best["validation_macro_f1"])
        np.testing.assert_allclose(cp["mean"].numpy(), x.astype(float).mean(axis=0), atol=1e-7)
        self.assertLessEqual(status["epochs_run"], settings["max_epochs"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            torch.save(cp, path)
            reproduced = a.io.mlp_predict(path, a.io.sha256(path), vx)
            np.testing.assert_allclose(reproduced, pv, atol=1e-7)

    def test_linear_missing_class_probabilities_are_aligned(self):
        from threadpoolctl import threadpool_limits
        rng = np.random.default_rng(10)
        y = np.tile(np.asarray([1, 4, 7]), 10)
        x = rng.normal(size=(30, 8))
        with threadpool_limits(limits=1):
            model, pt, pv, status = a.train_linear(x, y, x[:5], .01)
        self.assertEqual(pv.shape, (5, 10))
        np.testing.assert_allclose(pv.sum(1), 1)
        np.testing.assert_array_equal(pv[:, [0, 2, 3, 5, 6, 8, 9]], 0)
        np.testing.assert_allclose(model["mean"], x.mean(0))
        self.assertTrue(status["converged"])

    def test_regularization_variants_change_one_head_parameter(self):
        self.assertEqual(a.MLP_VARIANTS["mlp_early"], {})
        for name in ("mlp_small32", "mlp_dropout03", "mlp_wd1e3"):
            self.assertEqual(len(a.MLP_VARIANTS[name]), 1)

    def test_metric_uses_ten_classes(self):
        y = np.arange(10)
        p = np.zeros((10, 10))
        p[:, 0] = 1
        self.assertAlmostEqual(a.io.measure(y, p)["macro_f1"], (2/11)/10)

    def test_resume_rejects_corrupted_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            item = {"seed": 42, "strategy": "random", "ids_sha256": "subset"}
            identity = {"protocol_sha256": "protocol", "seed": 42, "strategy": "random",
                        "variant": "mlp_early", "labeled_ids_sha256": "subset", "labels": 2}
            (root / "predictions.npz").write_bytes(b"changed")
            a.io.write_json(root / "result.json", {"identity": identity,
                "artifact_sha256": {"predictions.npz": "wrong_hash"}})
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                a.fit_or_resume(root, item, "mlp_early", None, np.asarray([0, 1]), None,
                                None, None, None, {}, "protocol")


if __name__ == "__main__":
    unittest.main()
