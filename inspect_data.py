"""Read-only Galaxy10 DECaLS inspection; no training or data splitting.

Install: python -m pip install h5py numpy matplotlib
Run from any directory: python path/to/inspect_data.py
Custom path: python inspect_data.py --data "D:/datasets/Galaxy10_DECals.h5"

Defaults are relative to this script, not the terminal's working directory.
Images are read in small batches. The input HDF5 is opened read-only.
Class names follow https://github.com/henrysky/Galaxy10 (DECaLS version).
An exact-image hash detects byte-identical images, not all near-duplicates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        f"Missing dependency: {exc.name}\n"
        "Install with: python -m pip install h5py numpy matplotlib"
    ) from exc


CLASS_NAMES = {
    0: "Disturbed Galaxies",
    1: "Merging Galaxies",
    2: "Round Smooth Galaxies",
    3: "In-between Round Smooth Galaxies",
    4: "Cigar Shaped Smooth Galaxies",
    5: "Barred Spiral Galaxies",
    6: "Unbarred Tight Spiral Galaxies",
    7: "Unbarred Loose Spiral Galaxies",
    8: "Edge-on Galaxies without Bulge",
    9: "Edge-on Galaxies with Bulge",
}


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def structure_summary(handle):
    result = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            result[name] = {
                "shape": list(obj.shape), "dtype": str(obj.dtype),
                "compression": obj.compression,
            }

    handle.visititems(visit)
    return result


def numeric_summary(dataset):
    """Summarize small metadata arrays; images are handled separately."""
    result = {"shape": list(dataset.shape), "dtype": str(dataset.dtype)}
    if dataset.dtype.kind not in "biuf":
        result["numeric_summary"] = "not applicable"
        return result
    count = finite_count = 0
    low, high = float("inf"), float("-inf")
    chunks = [dataset[()]] if dataset.ndim == 0 else (
        dataset[start:start + 4096] for start in range(0, len(dataset), 4096)
    )
    for chunk in chunks:
        values = np.asarray(chunk)
        finite = np.isfinite(values)
        count += values.size
        finite_count += int(finite.sum())
        if finite.any():
            low = min(low, float(values[finite].min()))
            high = max(high, float(values[finite].max()))
    result.update({
        "value_count": count, "nonfinite_count": count - finite_count,
        "min": low if finite_count else None,
        "max": high if finite_count else None,
    })
    return result


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=root / "data/raw/Galaxy10_DECals.h5")
    parser.add_argument("--output", type=Path, default=root / "outputs/data_inspection")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42,
                        help="Only controls sample-image selection; no split is made.")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    source = args.data.expanduser().resolve()
    if not source.is_file():
        parser.error(
            f"Data file not found: {source}\n"
            'Place it in data/raw/ next to this script, or use --data "FULL_PATH".'
        )

    warnings = []
    with h5py.File(source, "r") as handle:
        structure = structure_summary(handle)
        print("HDF5 datasets:", flush=True)
        for name, info in structure.items():
            print(f"  {name}: shape={info['shape']}, dtype={info['dtype']}")
        if "images" not in handle or "ans" not in handle:
            raise ValueError("Expected datasets named 'images' and 'ans' at the HDF5 root.")
        images = handle["images"]
        if images.ndim != 4 or images.shape[-1] not in (1, 3):
            raise ValueError(f"Expected images with shape (N,H,W,1 or 3); got {images.shape}.")
        n = len(images)
        if not n or images.dtype.kind not in "biuf":
            raise ValueError("Images must be nonempty numeric arrays.")
        raw_labels = np.asarray(handle["ans"][()]).reshape(-1)
        if len(raw_labels) != n or raw_labels.dtype.kind not in "biuf":
            raise ValueError("The label dataset must contain exactly one numeric label per image.")
        if not np.isfinite(raw_labels).all() or not np.equal(raw_labels, np.floor(raw_labels)).all():
            raise ValueError("Labels contain nonfinite or non-integer values.")
        labels = raw_labels.astype(np.int64)
        if not np.isin(labels, list(CLASS_NAMES)).all():
            raise ValueError(f"Unexpected DECaLS labels: {np.unique(labels).tolist()}")

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        out = args.output.expanduser().resolve() / stamp
        out.mkdir(parents=True, exist_ok=False)
        counts = np.bincount(labels, minlength=10)
        print(f"\nImages: {n:,}; image shape: {images.shape[1:]}")
        for label, count in enumerate(counts):
            print(f"  {label}: {CLASS_NAMES[label]}: {count:,}")
        if n != 17736:
            warnings.append(f"Found {n} images; the standard README describes 17736. Check the version.")
        if np.any(counts == 0):
            warnings.append("Some of the ten expected classes are absent.")

        metadata = {}
        for key in ("ra", "dec", "redshift", "pxscale"):
            if key in handle:
                metadata[key] = numeric_summary(handle[key])
                if not handle[key].shape or handle[key].shape[0] != n:
                    warnings.append(f"Metadata '{key}' is not aligned with the image count.")
                if metadata[key].get("nonfinite_count", 0):
                    warnings.append(f"Metadata '{key}' contains nonfinite values.")
            else:
                warnings.append(f"Metadata '{key}' is absent.")

        # These are coordinate matches, not a definitive astronomical object match.
        coordinate_groups = []
        if all(key in handle and handle[key].size == n for key in ("ra", "dec")):
            ra = np.asarray(handle["ra"][()]).reshape(-1)
            dec = np.asarray(handle["dec"][()]).reshape(-1)
            if ra.dtype.kind in "biuf" and dec.dtype.kind in "biuf":
                groups = defaultdict(list)
                for idx in np.flatnonzero(np.isfinite(ra) & np.isfinite(dec)):
                    groups[(float(ra[idx]), float(dec[idx]))].append(int(idx))
                coordinate_groups = [indices for indices in groups.values() if len(indices) > 1]

        first_index = {}
        duplicates = defaultdict(list)
        nonfinite_rows = []
        min_value, max_value = float("inf"), float("-inf")
        print("\nScanning ALL images for exact duplicates and invalid pixels...", flush=True)
        with (out / "image_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image_index", "label", "sha256", "first_identical_image_index"])
            for start in range(0, n, args.batch_size):
                batch = np.asarray(images[start:start + args.batch_size])
                for offset, img in enumerate(batch):
                    idx = start + offset
                    finite = np.isfinite(img)
                    if not finite.all():
                        nonfinite_rows.append(idx)
                    if finite.any():
                        min_value = min(min_value, float(img[finite].min()))
                        max_value = max(max_value, float(img[finite].max()))
                    digest = hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()
                    first = first_index.setdefault(digest, idx)
                    if first != idx:
                        if not duplicates[digest]:
                            duplicates[digest].append(first)
                        duplicates[digest].append(idx)
                    writer.writerow([idx, int(labels[idx]), digest, first])
                done = min(start + args.batch_size, n)
                if start == 0 or done % (args.batch_size * 32) == 0 or done == n:
                    print(f"  {done:,}/{n:,} scanned", flush=True)

        duplicate_groups = list(duplicates.values())
        conflicts = [group for group in duplicate_groups if len(set(labels[group].tolist())) > 1]
        if duplicate_groups:
            warnings.append("Exact duplicate images found: group/remove them BEFORE splitting.")
        if coordinate_groups:
            warnings.append("Repeated exact coordinate pairs found: review object identity before splitting.")
        if conflicts:
            warnings.append("Some byte-identical images have conflicting labels; review before training.")
        if nonfinite_rows:
            warnings.append("Some images contain NaN or infinity; review before training.")

        write_csv(out / "class_distribution.csv", ["label", "class_name", "count", "fraction"],
                  ([i, CLASS_NAMES[i], int(counts[i]), float(counts[i] / n)] for i in range(10)))
        write_csv(out / "exact_duplicate_groups.csv", ["group_id", "image_index", "label"],
                  ([g, idx, int(labels[idx])] for g, group in enumerate(duplicate_groups) for idx in group))
        write_csv(out / "coordinate_groups.csv", ["group_id", "image_index", "label"],
                  ([g, idx, int(labels[idx])] for g, group in enumerate(coordinate_groups) for idx in group))

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(range(10), counts)
        ax.bar_label(bars, padding=3, fontsize=9)
        ax.set(xticks=range(10), xlabel="Class ID (see class_distribution.csv)",
               ylabel="Image count", title="Galaxy10 DECaLS: class distribution")
        ax.margins(y=0.15)
        fig.tight_layout()
        fig.savefig(out / "class_distribution.png", dpi=160)
        plt.close(fig)

        rng = np.random.default_rng(args.seed)
        fig, axes = plt.subplots(10, 3, figsize=(10, 27), squeeze=False)
        sample_rows = []
        for label in range(10):
            candidates = np.flatnonzero(labels == label)
            chosen = rng.choice(candidates, size=min(3, len(candidates)), replace=False)
            for col in range(3):
                ax = axes[label, col]
                ax.axis("off")
                if col >= len(chosen):
                    continue
                idx = int(chosen[col])
                img = np.asarray(images[idx])
                # Display-only scaling for nonstandard numeric images; raw data is unchanged.
                if not (img.dtype == np.uint8 or (np.isfinite(img).all() and img.min() >= 0 and img.max() <= 1)):
                    img = np.nan_to_num(img.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
                    img = (img - img.min()) / max(float(np.ptp(img)), 1e-12)
                ax.imshow(img[..., 0] if img.shape[-1] == 1 else img,
                          cmap="gray" if img.shape[-1] == 1 else None)
                ax.set_title(f"Class {label} | index {idx}\n{CLASS_NAMES[label]}", fontsize=8)
                sample_rows.append([label, idx])
        fig.tight_layout()
        fig.savefig(out / "sample_images.png", dpi=120)
        plt.close(fig)
        write_csv(out / "sample_indices.csv", ["label", "image_index"], sample_rows)

        report = {
            "created_utc": stamp, "input_file": str(source),
            "input_size_bytes": source.stat().st_size,
            "input_mtime_ns": source.stat().st_mtime_ns,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "h5py": h5py.__version__, "matplotlib": matplotlib.__version__},
            "arguments": {"batch_size": args.batch_size, "sample_seed": args.seed},
            "datasets": structure, "image_count": n,
            "class_counts": {str(i): int(counts[i]) for i in range(10)},
            "observed_class_imbalance_ratio": float(counts.max() / counts[counts > 0].min()),
            "pixel_min": min_value if np.isfinite(min_value) else None,
            "pixel_max": max_value if np.isfinite(max_value) else None,
            "images_with_nonfinite_pixels": nonfinite_rows,
            "metadata": metadata,
            "exact_duplicate_group_count": len(duplicate_groups),
            "redundant_exact_image_count": sum(len(group) - 1 for group in duplicate_groups),
            "conflicting_label_groups": conflicts,
            "repeated_exact_coordinate_group_count": len(coordinate_groups),
            "warnings": warnings,
            "limitations": [
                "No perceptual near-duplicate detection or angular-distance object matching is performed.",
                "Metadata ranges do not establish validity; finite sentinel values may still exist.",
                "No samples are removed, no data split is created, and no model is trained.",
            ],
        }
        (out / "inspection_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
    print(f"\nExact duplicate groups: {len(duplicate_groups)}")
    print(f"Redundant exact images: {report['redundant_exact_image_count']}")
    print(f"Repeated exact coordinate groups: {len(coordinate_groups)}")
    for warning in warnings:
        print(f"NOTE: {warning}")
    print(f"\nInspection complete. Results: {out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
