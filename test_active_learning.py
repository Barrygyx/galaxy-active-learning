"""Meaningful offline algorithm checks: python -m unittest test_active_learning -v."""
import unittest
import numpy as np
from galaxy_al.acquisition import entropy, select_batch, unit_features
from galaxy_al.experiment import LabelOracle, make_budgets
from galaxy_al.metrics import classification_metrics, normalized_aulc


class ActiveLearningTests(unittest.TestCase):
    def test_uncertainty_prefers_uniform_probability(self):
        x = unit_features(np.eye(4, dtype=np.float32))
        p = np.full((3, 10), .001)
        p[0, 0] = .991
        p[1] = .1
        p[2, 0] = .991
        selected = select_batch("uncertainty", x, [1, 2, 3], [0], 1, 9, p)
        self.assertEqual(selected.tolist(), [2])
        self.assertGreater(entropy(p)[1], entropy(p)[0])

    def test_diversity_covers_distant_regions(self):
        x = np.array([[1., 0.], [.99, .01], [0., 1.], [-1., 0.]], dtype=np.float32)
        x = unit_features(x)
        selected = select_batch("diversity", x, [1, 2, 3], [0], 2, 1)
        self.assertEqual(selected.tolist(), [3, 2])

    def test_identical_features_never_repeat_query(self):
        x = np.ones((10, 4), dtype=np.float32)
        selected = select_batch("diversity", x, np.arange(1, 10), [0], 8, 2)
        self.assertEqual(len(set(selected.tolist())), 8)
        self.assertNotIn(0, selected)

    def test_hybrid_cannot_select_outside_uncertainty_shortlist(self):
        rng = np.random.default_rng(0)
        x = unit_features(rng.normal(size=(20, 8)))
        available = np.arange(2, 20)
        p = rng.dirichlet(np.ones(10), size=len(available))
        ranked = available[np.lexsort((available, -entropy(p)))]
        selected = select_batch("hybrid", x, available, [0, 1], 3, 1, p, multiplier=2)
        self.assertTrue(set(selected) <= set(ranked[:6]))

    def test_random_seed_reproducibility(self):
        x = np.eye(20, dtype=np.float32)
        a = select_batch("random", x, np.arange(2, 20), [0, 1], 5, 42)
        b = select_batch("random", x, np.arange(2, 20), [0, 1], 5, 42)
        np.testing.assert_array_equal(a, b)

    def test_oracle_denies_hidden_and_nontraining_labels(self):
        oracle = LabelOracle(np.array([3, 4, 5, 6]), [0, 1])
        with self.assertRaises(ValueError):
            oracle.revealed_labels([0])
        with self.assertRaises(ValueError):
            oracle.query([2])
        self.assertEqual(oracle.query([0]).tolist(), [3])
        with self.assertRaises(ValueError):
            oracle.query([0])
        self.assertEqual(oracle.revealed_labels([0]).tolist(), [3])

    def test_metrics_keep_all_ten_classes(self):
        truth = np.arange(10)
        perfect = classification_metrics(truth, np.eye(10))
        self.assertEqual(perfect["macro_f1"], 1.)
        self.assertEqual(perfect["accuracy"], 1.)
        constant = np.zeros((10, 10)); constant[:, 0] = 1
        measured = classification_metrics(truth, constant)
        self.assertAlmostEqual(measured["macro_f1"], (2 / 11) / 10)
        self.assertEqual(measured["per_class"][4]["recall"], 0.)

    def test_budget_endpoint_and_aulc(self):
        config = {"initial_labels": 200, "max_labels": 650, "query_batch_size": 200}
        self.assertEqual(make_budgets(config), [200, 400, 600, 650])
        self.assertAlmostEqual(normalized_aulc([200, 400, 600], [.2, .4, .6]), .4)


if __name__ == "__main__":
    unittest.main()
