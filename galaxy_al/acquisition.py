"""Selection functions receive features/probabilities, never ground-truth labels."""
import numpy as np


def unit_features(features):
    x = np.asarray(features, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def entropy(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 10 or not np.isfinite(p).all():
        raise ValueError("Uncertainty acquisition requires finite ten-class probabilities.")
    return -(p * np.log(np.maximum(p, 1e-12))).sum(axis=1)


def kcenter_greedy(features, candidates, labeled, count):
    """Farthest-first coverage in L2-normalized, FROZEN image feature space.

    Initialize distances against ALL labeled centers. After each choice update
    against the newly selected center. Matrix sizes are bounded by chunking.
    Equal-distance ties resolve by the stable candidate order.
    """
    candidates = np.asarray(candidates, dtype=np.int64)
    labeled = np.asarray(labeled, dtype=np.int64)
    if count > len(candidates) or count < 1 or len(labeled) == 0:
        raise ValueError("Invalid k-center selection size or empty labeled set.")
    x = features[candidates]
    distances = np.full(len(candidates), np.inf, dtype=np.float32)
    norms = np.einsum("ij,ij->i", x, x)
    for start in range(0, len(labeled), 256):
        centers = features[labeled[start:start + 256]]
        center_norms = np.einsum("ij,ij->i", centers, centers)
        for offset in range(0, len(x), 1024):
            block = x[offset:offset + 1024]
            d = norms[offset:offset + len(block), None] + center_norms[None, :] - 2 * block @ centers.T
            distances[offset:offset + len(block)] = np.minimum(
                distances[offset:offset + len(block)], np.maximum(d, 0).min(axis=1))
    selected = []
    selected_mask = np.zeros(len(candidates), dtype=bool)
    for _ in range(count):
        local = int(np.argmax(distances))
        selected.append(int(candidates[local]))
        center = x[local]
        d = np.maximum(norms + float(center @ center) - 2 * (x @ center), 0)
        distances = np.minimum(distances, d)
        # All prior picks stay unavailable, including the degenerate identical-vector case.
        selected_mask[local] = True
        distances[selected_mask] = -np.inf
    return np.array(selected, dtype=np.int64)


def select_batch(strategy, features, available, labeled, count, seed,
                 probabilities=None, multiplier=5):
    available = np.asarray(available, dtype=np.int64)
    if count > len(available) or count < 1:
        raise ValueError("Query batch exceeds available pool.")
    if np.intersect1d(available, labeled).size:
        raise ValueError("Available and labeled pools overlap.")
    if strategy == "random":
        chosen = np.random.default_rng(seed).choice(available, count, replace=False)
    elif strategy == "diversity":
        chosen = kcenter_greedy(features, available, labeled, count)
    elif strategy in ("uncertainty", "hybrid"):
        if probabilities is None or len(probabilities) != len(available):
            raise ValueError("Probability rows must match available pool positions.")
        scores = entropy(probabilities)
        ranked = available[np.lexsort((available, -scores))]
        if strategy == "uncertainty":
            chosen = ranked[:count]
        else:
            candidate_count = min(len(ranked), max(count, int(multiplier) * count))
            # Stable source-position order also makes diversity-only tie rules comparable.
            candidates = np.sort(ranked[:candidate_count])
            chosen = kcenter_greedy(features, candidates, labeled, count)
    else:
        raise ValueError(f"Unknown acquisition strategy: {strategy}")
    if len(np.unique(chosen)) != count or not np.isin(chosen, available).all():
        raise RuntimeError("Acquisition returned duplicate or unavailable indices.")
    return chosen
