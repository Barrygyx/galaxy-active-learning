"""Conservative Galaxy10 cleaning and fixed stratified splits.

Run: python prepare_data.py
Dependencies: h5py, numpy (already used by inspect_data.py).

The raw HDF5 is opened read-only. No image is overwritten or relabeled.
Exact-image matches and exact finite (RA, Dec) matches define linked groups.
Conflicting groups are quarantined; consistent groups retain the first valid
source index. Nonfinite images are excluded. Missing metadata alone does not
exclude an image because the planned model uses images only.

Outputs reference ORIGINAL HDF5 row indices, not renumbered image positions.
No perceptual duplicate search or approximate sky-coordinate matching is done.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    import h5py
    import numpy as np
except ImportError as exc:
    raise SystemExit(f"Missing {exc.name}. Run: python -m pip install h5py numpy") from exc


CLASS_NAMES = {
    0: "Disturbed Galaxies", 1: "Merging Galaxies", 2: "Round Smooth Galaxies",
    3: "In-between Round Smooth Galaxies", 4: "Cigar Shaped Smooth Galaxies",
    5: "Barred Spiral Galaxies", 6: "Unbarred Tight Spiral Galaxies",
    7: "Unbarred Loose Spiral Galaxies", 8: "Edge-on Galaxies without Bulge",
    9: "Edge-on Galaxies with Bulge",
}


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        self.parent[self.find(b)] = self.find(a)


def save_csv(path, columns, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def counts_by_class(indices, labels):
    counts = Counter(int(labels[i]) for i in indices)
    return {str(label): counts[label] for label in CLASS_NAMES}


def stratified_split(kept, labels, seed):
    """Per-class 70/15/15 allocation with largest-remainder rounding."""
    rng = np.random.default_rng(seed)
    assignments = {}
    names = ("train_pool", "validation", "test")
    for label in CLASS_NAMES:
        indices = np.array([i for i in kept if labels[i] == label], dtype=np.int64)
        if len(indices) < 7:
            raise ValueError(f"Class {label} has fewer than 7 clean images; review the split design.")
        rng.shuffle(indices)
        expected = len(indices) * np.array([0.70, 0.15, 0.15])
        sizes = np.floor(expected).astype(int)
        order = np.argsort(-(expected - sizes), kind="stable")
        for split_id in order[:len(indices) - int(sizes.sum())]:
            sizes[split_id] += 1
        start = 0
        for name, size in zip(names, sizes):
            for idx in indices[start:start + size]:
                assignments[int(idx)] = name
            start += int(size)
    return assignments


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data/raw/Galaxy10_DECals.h5")
    parser.add_argument("--output", type=Path, default=root / "data/processed")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    source = args.data.expanduser().resolve()
    if not source.is_file():
        parser.error(f'Data file not found: {source}. Use --data "FULL_PATH" if needed.')

    metadata = {}
    metadata_checks = {}
    metadata_notes = []
    with h5py.File(source, "r") as f:
        if not all(name in f for name in ("images", "ans")):
            raise ValueError("The HDF5 must contain images and ans datasets.")
        images = f["images"]
        if images.ndim != 4 or images.shape[-1] != 3 or images.dtype.kind not in "biuf":
            raise ValueError("Expected numeric images of shape (N,H,W,3).")
        n = len(images)
        if not n:
            raise ValueError("The dataset is empty.")
        raw_labels = np.asarray(f["ans"][()]).reshape(-1)
        if len(raw_labels) != n or raw_labels.dtype.kind not in "biuf":
            raise ValueError("Expected exactly one numeric label per image.")
        if not np.isfinite(raw_labels).all() or not np.equal(raw_labels, np.floor(raw_labels)).all():
            raise ValueError("Invalid/non-integer labels found. No automatic relabeling is performed.")
        labels = raw_labels.astype(np.int64)
        if not np.isin(labels, list(CLASS_NAMES)).all():
            raise ValueError("Labels outside the expected DECaLS range 0..9 were found.")

        links = UnionFind(n)
        coordinate_first = {}
        coordinate_counts = Counter()
        for name in ("ra", "dec", "redshift", "pxscale"):
            if name not in f or f[name].size != n or f[name].dtype.kind not in "biuf":
                metadata_notes.append(f"{name}: absent or not aligned numeric metadata; not used for modeling.")
                continue
            values = np.asarray(f[name][()]).reshape(-1)
            metadata[name] = values
            metadata_checks[name] = {"nonfinite_count": int((~np.isfinite(values)).sum())}
        if "ra" in metadata and "dec" in metadata:
            for idx, (ra, dec) in enumerate(zip(metadata["ra"], metadata["dec"])):
                if np.isfinite(ra) and np.isfinite(dec):
                    key = (float(ra), float(dec))
                    coordinate_counts[key] += 1
                    first = coordinate_first.setdefault(key, idx)
                    links.union(first, idx)

        digests = []
        hash_first = {}
        hash_counts = Counter()
        valid_pixels = np.ones(n, dtype=bool)
        print(f"Scanning {n:,} images; the raw file is read-only.", flush=True)
        for start in range(0, n, args.batch_size):
            batch = np.asarray(images[start:start + args.batch_size])
            for offset, img in enumerate(batch):
                idx = start + offset
                valid_pixels[idx] = bool(np.isfinite(img).all())
                digest = hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()
                digests.append(digest)
                hash_counts[digest] += 1
                first = hash_first.setdefault(digest, idx)
                links.union(first, idx)
            done = min(start + args.batch_size, n)
            if start == 0 or done % (args.batch_size * 32) == 0 or done == n:
                print(f"  {done:,}/{n:,}", flush=True)
        image_shape = list(images.shape)
        image_dtype = str(images.dtype)

    components = defaultdict(list)
    for idx in range(n):
        components[links.find(idx)].append(idx)
    reasons = [""] * n
    group_ids = np.empty(n, dtype=np.int64)
    conflict_group_count = 0
    for group in components.values():
        group_id = min(group)
        group_ids[group] = group_id
        if len({int(labels[i]) for i in group}) > 1:
            conflict_group_count += 1
            for idx in group:
                reasons[idx] = "conflicting_labels_in_linked_group"
            continue
        eligible = [idx for idx in group if valid_pixels[idx]]
        representative = min(eligible) if eligible else None
        for idx in group:
            if not valid_pixels[idx]:
                reasons[idx] = "nonfinite_image_pixels"
            elif idx != representative:
                reasons[idx] = "redundant_consistent_group_member"

    kept = [idx for idx in range(n) if not reasons[idx]]
    excluded = [idx for idx in range(n) if reasons[idx]]
    assignments = stratified_split(kept, labels, args.seed)
    split_indices = {name: [i for i in kept if assignments[i] == name]
                     for name in ("train_pool", "validation", "test")}
    assert len(kept) + len(excluded) == n
    assert len(assignments) == len(kept)
    assert len({int(group_ids[i]) for i in kept}) == len(kept)
    assert len({digests[i] for i in kept}) == len(kept)
    assert all(len({assignments[i] for i in group if i in assignments}) <= 1
               for group in components.values())

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = args.output.expanduser().resolve() / f"prepared_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    base_columns = ["source_index", "label", "group_id", "sha256"]
    def base_row(i):
        return [i, int(labels[i]), int(group_ids[i]), digests[i]]
    save_csv(out / "clean_indices.csv", base_columns, (base_row(i) for i in kept))
    save_csv(out / "splits.csv", base_columns + ["split"],
             (base_row(i) + [assignments[i]] for i in kept))
    save_csv(out / "excluded_samples.csv", base_columns + ["reason"],
             (base_row(i) + [reasons[i]] for i in excluded))
    meta_names = list(metadata)
    save_csv(out / "cleaning_audit.csv", base_columns + ["status", "reason", "split"] + meta_names,
             (base_row(i) + ["excluded" if reasons[i] else "kept", reasons[i], assignments.get(i, "")]
              + [float(metadata[name][i]) for name in meta_names] for i in range(n)))
    for name, indices in split_indices.items():
        np.save(out / f"{name}_indices.npy", np.array(indices, dtype=np.int64), allow_pickle=False)
    save_csv(out / "class_distribution.csv",
             ["label", "class_name", "original", "excluded", "clean", "train_pool", "validation", "test"],
             ([label, CLASS_NAMES[label], int((labels == label).sum()),
               sum(labels[i] == label for i in excluded), sum(labels[i] == label for i in kept)]
              + [sum(labels[i] == label for i in indices) for indices in split_indices.values()]
              for label in CLASS_NAMES))

    artifacts = {}
    for name in ("clean_indices.csv", "splits.csv", "excluded_samples.csv", "cleaning_audit.csv"):
        artifacts[name] = hashlib.sha256((out / name).read_bytes()).hexdigest()
    report = {
        "created_utc": stamp, "source_file": str(source), "source_size_bytes": source.stat().st_size,
        "source_image_shape": image_shape, "source_image_dtype": image_dtype,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "h5py": h5py.__version__},
        "seed": args.seed, "original_count": n, "clean_count": len(kept),
        "excluded_count": len(excluded), "excluded_fraction": len(excluded) / n,
        "exact_image_duplicate_groups": sum(count > 1 for count in hash_counts.values()),
        "exact_coordinate_duplicate_groups": sum(count > 1 for count in coordinate_counts.values()),
        "conflicting_linked_groups": conflict_group_count,
        "images_with_nonfinite_pixels": int((~valid_pixels).sum()),
        "exclusion_reason_counts": dict(Counter(reasons[i] for i in excluded)),
        "original_class_counts": counts_by_class(range(n), labels),
        "clean_class_counts": counts_by_class(kept, labels),
        "split_sizes": {name: len(indices) for name, indices in split_indices.items()},
        "split_class_counts": {name: counts_by_class(indices, labels) for name, indices in split_indices.items()},
        "metadata_checks": metadata_checks, "metadata_notes": metadata_notes,
        "index_semantics": "All indices are zero-based ORIGINAL rows in the source HDF5.",
        "split_policy": "Per-class 70/15/15 largest-remainder allocation, seeded numpy default_rng.",
        "cleaning_policy": {
            "grouping": "Transitive union of byte-identical images and exactly equal finite RA/Dec pairs.",
            "conflicts": "Exclude entire linked group; do not guess or vote on its label.",
            "consistent_groups": "Keep smallest original index with finite pixels.",
            "metadata_missing": "Retain image; metadata is not a classifier input.",
            "class_imbalance": "Preserve natural proportions; no oversampling or balancing here.",
        },
        "limitations": [
            "Exact-coordinate matches are used as conservative linkage, not verified catalog identities.",
            "Near-duplicate images and slightly different coordinates may remain undetected.",
            "The curated subset excludes ambiguous linked labels; performance is not for the full raw dataset.",
            "Validation/test labels are fixed evaluation resources and must be reported separately from query budgets.",
        ],
        "artifact_sha256": artifacts,
    }
    (out / "cleaning_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"\nOriginal images: {n:,}")
    print(f"Excluded: {len(excluded):,}")
    print(f"Retained: {len(kept):,}")
    for name, indices in split_indices.items():
        print(f"  {name}: {len(indices):,}")
    print(f"\nSaved preparation results: {out}")
    print("Next: retain this run's splits.csv as the common split for ALL acquisition strategies.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
