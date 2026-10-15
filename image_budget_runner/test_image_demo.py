"""Gradient, augmentation, label isolation and exact epoch-resume tests."""
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import h5py
import numpy as np
import torch
from galaxy_image_demo.common import classification_metrics, read_json, sha256_file
from galaxy_image_demo.data import select_subsets, cache_images
from galaxy_image_demo.models import build_model, configure_trainable, training_mode, augment_d4
from galaxy_image_demo.training import fit_model, class_weights


class ImageDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_train_selection_does_not_use_labels_and_budgets_are_nested(self):
        roles = np.asarray(["train_pool"]*100 + ["validation"]*50 + ["test"]*20)
        data = SimpleNamespace(labels=np.arange(170) % 10, positions=lambda r: np.flatnonzero(roles == r))
        a = select_subsets(data, 20, 30, 42)
        b = select_subsets(data, 50, 30, 42)
        np.testing.assert_array_equal(a[0], b[0][:20])
        np.testing.assert_array_equal(a[1], b[1])
        self.assertTrue(set(a[0]) <= set(range(100)))
        self.assertTrue(set(a[1]) <= set(range(100, 150)))
        self.assertEqual(set(data.labels[a[1]]), set(range(10)))
        data.labels[:100] = 9  # Hidden train labels cannot change which images are queried.
        c = select_subsets(data, 20, 30, 42)
        np.testing.assert_array_equal(a[0], c[0])

    def test_augmentation_preserves_pixels_input_and_reproducibility(self):
        x = torch.arange(4*3*8*8).reshape(4, 3, 8, 8).float()
        before = x.clone()
        a = augment_d4(x, torch.Generator().manual_seed(19))
        b = augment_d4(x, torch.Generator().manual_seed(19))
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(x, before, rtol=0, atol=0)
        for i in range(len(x)):
            torch.testing.assert_close(a[i].flatten().sort().values, x[i].flatten().sort().values, rtol=0, atol=0)

    def test_partial_cnn_updates_selected_convolution_only(self):
        model, _ = build_model("resnet18_partial_aug", seed=11, download=False)
        configure_trainable(model)
        for name, p in model.backbone.named_parameters():
            self.assertEqual(p.requires_grad, name.startswith(("layer3.", "layer4.")))
        training_mode(model)
        self.assertFalse(model.backbone.bn1.training)
        self.assertFalse(model.backbone.layer4[0].bn1.training)
        frozen = model.backbone.conv1.weight.detach().clone()
        tuned = model.backbone.layer4[0].conv1.weight.detach().clone()
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
        loss = torch.nn.functional.cross_entropy(model(torch.rand(2, 3, 64, 64)), torch.tensor([0, 1]))
        loss.backward()
        self.assertIsNone(model.backbone.conv1.weight.grad)
        self.assertIsNotNone(model.backbone.layer4[0].conv1.weight.grad)
        optimizer.step()
        torch.testing.assert_close(frozen, model.backbone.conv1.weight, rtol=0, atol=0)
        self.assertFalse(torch.equal(tuned, model.backbone.layer4[0].conv1.weight))

    def test_full_finetuning_and_warmup_trainable_flags(self):
        model, _ = build_model("resnet18_full_aug", seed=11, download=False)
        configure_trainable(model, warmup=False)
        self.assertTrue(all(p.requires_grad for p in model.backbone.parameters()))
        configure_trainable(model, warmup=True)
        self.assertFalse(any(p.requires_grad for p in model.backbone.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.head.parameters()))

    def test_warmup_then_resume_updates_convolution_and_keeps_bn_stats(self):
        rng = np.random.default_rng(23)
        images = rng.random((18, 3, 64, 64), dtype=np.float32)
        initial = rng.normal(size=(18, 512)).astype(np.float32)
        y = np.arange(18) % 10
        subset = {"train": np.arange(8), "train_monitor": np.arange(8),
                  "monitor": np.arange(8, 18), "validation": np.arange(8, 18)}
        settings = {"seed": 7, "epochs": 3, "min_epochs": 1, "warmup_epochs": 1,
                    "patience": 99, "batch_size": 4, "min_delta": .0001,
                    "head_lr": .001, "backbone_lr": .0001, "scratch_lr": .001, "weight_decay": .0001}
        model, _ = build_model("resnet18_partial_aug", seed=7, download=False)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        identity = {"model_name": "resnet18_partial_aug"}
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, "Simulated"):
                fit_model(model, images, initial, subset, y, settings, Path(folder), identity,
                          torch.device("cpu"), interrupt_after_epoch=1)
            first = torch.load(Path(folder)/"progress.pt", map_location="cpu", weights_only=True)
            key = "backbone.layer4.0.conv1.weight"
            torch.testing.assert_close(first["model"][key], before[key], rtol=0, atol=0)
            model, _ = build_model("resnet18_partial_aug", seed=7, download=False)
            with self.assertRaisesRegex(RuntimeError, "Simulated"):
                fit_model(model, images, initial, subset, y, settings, Path(folder), identity,
                          torch.device("cpu"), interrupt_after_epoch=2)
            second = torch.load(Path(folder)/"progress.pt", map_location="cpu", weights_only=True)
            self.assertFalse(torch.equal(second["model"][key], before[key]))
            for k in before:
                if "running_mean" in k or "running_var" in k:
                    torch.testing.assert_close(second["model"][k], before[k], rtol=0, atol=0)

    def test_epoch_resume_restores_exact_model_and_best_checkpoint(self):
        rng = np.random.default_rng(23)
        images = rng.random((26, 3, 64, 64), dtype=np.float32)
        y = np.arange(26) % 10
        subset = {"train": np.arange(16), "train_monitor": np.arange(16),
                  "monitor": np.arange(16, 26), "validation": np.arange(16, 26)}
        settings = {"seed": 7, "epochs": 3, "min_epochs": 1, "warmup_epochs": 0,
                    "patience": 99, "batch_size": 8, "min_delta": .0001,
                    "head_lr": .001, "backbone_lr": .0001, "scratch_lr": .001, "weight_decay": .0001}
        identity = {"model_name": "tiny_cnn_aug", "unit_test": True}
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder)/"uninterrupted", Path(folder)/"interrupted"
            model, _ = build_model("tiny_cnn_aug", seed=7, download=False)
            reference = fit_model(model, images, None, subset, y, settings, a, identity, torch.device("cpu"))
            model, _ = build_model("tiny_cnn_aug", seed=7, download=False)
            with self.assertRaisesRegex(RuntimeError, "Simulated"):
                fit_model(model, images, None, subset, y, settings, b, identity, torch.device("cpu"), interrupt_after_epoch=1)
            model, _ = build_model("tiny_cnn_aug", seed=7, download=False)
            resumed = fit_model(model, images, None, subset, y, settings, b, identity, torch.device("cpu"))
            self.assertEqual(reference["validation"], resumed["validation"])
            self.assertEqual(reference["best_epoch"], resumed["best_epoch"])
            cp1 = torch.load(a/"best_model.pt", map_location="cpu", weights_only=True)
            cp2 = torch.load(b/"best_model.pt", map_location="cpu", weights_only=True)
            for key in cp1["state_dict"]:
                torch.testing.assert_close(cp1["state_dict"][key], cp2["state_dict"][key], rtol=0, atol=0)
            history = read_json(b/"history.json")
            expected = max(history, key=lambda r: (r["monitor_macro_f1"], -r["monitor_ce"]))
            self.assertEqual(expected["epoch"], resumed["best_epoch"])
            # Finished fits must validate cached artifact bytes before reuse.
            (b/"predictions.npz").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                fit_model(model, images, None, subset, y, settings, b, identity, torch.device("cpu"))

    def test_frozen_feature_normalization_uses_only_train(self):
        rng = np.random.default_rng(9)
        initial = rng.normal(size=(26, 512)).astype(np.float32)
        initial[16:] += 50
        y = np.arange(26) % 10
        subset = {"train": np.arange(16), "train_monitor": np.arange(16),
                  "monitor": np.arange(16, 26), "validation": np.arange(16, 26)}
        settings = {"seed": 7, "epochs": 1, "min_epochs": 1, "warmup_epochs": 0,
                    "patience": 99, "batch_size": 8, "min_delta": .0001,
                    "head_lr": .001, "backbone_lr": .0001, "scratch_lr": .001, "weight_decay": .0001}
        model, _ = build_model("resnet18_frozen", seed=7, download=False)
        with tempfile.TemporaryDirectory() as folder:
            fit_model(model, None, initial, subset, y, settings, Path(folder),
                      {"model_name": "resnet18_frozen"}, torch.device("cpu"))
            expected = initial[:16].astype(float).mean(0).astype(np.float32)
            np.testing.assert_array_equal(model.feature_mean.numpy(), expected)

    def test_raw_pixel_hash_mismatch_rejected_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "raw.h5"
            with h5py.File(source, "w") as h:
                h.create_dataset("images", data=np.zeros((1, 8, 8, 3), dtype=np.uint8))
            before = sha256_file(source)
            data = SimpleNamespace(source=source, ids=np.asarray([0]), split_sha256="split", image_hashes=["wrong"])
            with self.assertRaisesRegex(ValueError, "Raw image changed"):
                cache_images(root, data, np.asarray([0]), 64, torch)
            self.assertEqual(before, sha256_file(source))

    def test_class_weights_derive_from_observed_counts(self):
        weights = class_weights(np.asarray([0]*9 + [4]))
        self.assertGreater(weights[4], weights[0])
        self.assertEqual(weights[1], 0)
        self.assertAlmostEqual(float(weights[[0, 4]].mean()), 1)

    def test_macro_metric_includes_all_ten_classes(self):
        p = np.zeros((10, 10))
        p[:, 0] = 1
        self.assertAlmostEqual(classification_metrics(np.arange(10), p)["macro_f1"], (2/11)/10)


if __name__ == "__main__":
    unittest.main()
