"""Validate and reuse the existing split; never re-split or rebalance it."""
from dataclasses import dataclass
from pathlib import Path
import h5py
import numpy as np
from .common import read_csv, read_json, sha256_file


def select_subsets(data, train_size, monitor_size, seed):
    """Uniform train query; stratified validation monitor; full validation retained."""
    rng = np.random.default_rng(seed)
    pool, validation = data.positions("train_pool"), data.positions("validation")
    if not 2 <= train_size <= len(pool):
        raise ValueError(f"train-size must be between 2 and {len(pool)}")
    selected = rng.permutation(pool)[:train_size]
    # Validation monitor stays fixed if only the training budget changes.
    rng = np.random.default_rng(seed + 997)
    monitor_size = min(monitor_size, len(validation))
    if monitor_size < 10:
        raise ValueError("Validation monitor must have at least 10 images.")
    counts = np.bincount(data.labels[validation], minlength=10)
    ideal = counts * monitor_size / counts.sum()
    allocation = np.maximum(1, np.floor(ideal).astype(int))
    while allocation.sum() < monitor_size:
        choices = np.flatnonzero(allocation < counts)
        chosen = choices[np.argmax(ideal[choices] - allocation[choices])]
        allocation[chosen] += 1
    while allocation.sum() > monitor_size:
        choices = np.flatnonzero(allocation > 1)
        chosen = choices[np.argmax(allocation[choices] - ideal[choices])]
        allocation[chosen] -= 1
    monitor = np.concatenate([rng.choice(validation[data.labels[validation] == i], allocation[i], replace=False) for i in range(10)])
    train_monitor = rng.choice(selected, min(256, len(selected)), replace=False)
    return selected, np.sort(monitor), validation, np.sort(train_monitor)


def cache_images(root, data, positions, size, torch):
    import hashlib
    from .common import fingerprint, write_json
    positions = np.asarray(sorted(set(map(int, positions))), dtype=np.int64)
    ids = data.ids[positions]
    signature = {"split_sha256": data.split_sha256, "source_size": data.source.stat().st_size,
                 "source_mtime_ns": data.source.stat().st_mtime_ns,
                 "ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(), "image_size": size,
                 "transform": "full_frame_bilinear_antialias_float16_0to1", "torch": torch.__version__,
                 "data_code_sha256": sha256_file(Path(__file__))}
    folder = root / "cache/image_demo" / fingerprint(signature)[:16]
    folder.mkdir(parents=True, exist_ok=True)
    image_path, done_path, manifest_path = folder / "images.npy", folder / "done.npy", folder / "manifest.json"
    shape = (len(ids), 3, size, size)
    if image_path.exists() and done_path.exists():
        cached = np.load(image_path, mmap_mode="r+", allow_pickle=False)
        done = np.load(done_path, mmap_mode="r+", allow_pickle=False)
        if cached.shape != shape or cached.dtype != np.float16 or done.shape != (len(ids),):
            raise ValueError("Image demo cache has invalid shape/dtype.")
    else:
        cached = np.lib.format.open_memmap(image_path, mode="w+", dtype=np.float16, shape=shape)
        cached[:] = np.nan
        cached.flush()
        done = np.lib.format.open_memmap(done_path, mode="w+", dtype=bool, shape=(len(ids),))
        done[:] = False
        done.flush()
    if done.all() and manifest_path.exists():
        saved = read_json(manifest_path)
        if saved["signature"] != signature or saved["images_sha256"] != sha256_file(image_path):
            raise ValueError("Image cache signature/checksum mismatch.")
    else:
        pending = np.flatnonzero(~done)
        print(f"Read-only image preparation: {len(pending)}/{len(ids)} remaining", flush=True)
        with h5py.File(data.source, "r") as handle:
            for start in range(0, len(pending), 32):
                indices = pending[start:start + 32]
                raw = np.asarray(handle["images"][ids[indices]])
                for j, index in enumerate(indices):
                    digest = hashlib.sha256(np.ascontiguousarray(raw[j]).tobytes()).hexdigest()
                    if digest != data.image_hashes[int(positions[index])]:
                        raise ValueError(f"Raw image changed: source index {ids[index]}")
                tensor = torch.from_numpy(raw).permute(0, 3, 1, 2).float().div_(255)
                resized = torch.nn.functional.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
                cached[indices] = resized.numpy().astype(np.float16)
                cached.flush()
                done[indices] = True
                done.flush()
                print(f"  prepared images {int(done.sum())}/{len(ids)}", flush=True)
        np.save(folder / "source_indices.npy", ids, allow_pickle=False)
        write_json(manifest_path, {"signature": signature, "images_sha256": sha256_file(image_path), "shape": list(shape)})
    if not np.isfinite(cached).all() or cached.min() < 0 or cached.max() > 1:
        raise ValueError("Invalid cached image values.")
    saved_ids = np.load(folder / "source_indices.npy", allow_pickle=False)
    if not np.array_equal(saved_ids, ids):
        raise ValueError("Image cache indices changed.")
    return np.load(image_path, mmap_mode="r", allow_pickle=False), positions, folder, read_json(manifest_path)


@dataclass
class PreparedData:
    source: Path
    directory: Path
    ids: np.ndarray                 # Original HDF5 indices, sorted.
    labels: np.ndarray              # Ground truth; selection functions never receive this.
    roles: np.ndarray
    image_hashes: list
    split_sha256: str
    report: dict

    def positions(self, role):
        return np.flatnonzero(self.roles == role)


def load_prepared(root, config, smoke=False):
    source = (root / config["data_file"]).resolve()
    directory = (root / config["prepared_dir"]).resolve()
    if not source.is_file():
        raise ValueError(f"HDF5 not found: {source}. Update data_file in al_config.json.")
    if not directory.is_dir():
        raise ValueError(f"Prepared directory not found: {directory}. Update prepared_dir in al_config.json.")
    report = read_json(directory / "cleaning_report.json")
    for name, expected in report["artifact_sha256"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"Prepared file changed since cleaning: {name}. Do not mix preparation runs.")
    rows = read_csv(directory / "splits.csv")
    clean = read_csv(directory / "clean_indices.csv")
    rows.sort(key=lambda row: int(row["source_index"]))
    ids = np.array([int(row["source_index"]) for row in rows], dtype=np.int64)
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    roles = np.array([row["split"] for row in rows])
    hashes = [row["sha256"] for row in rows]
    if len(ids) != report["clean_count"] or len(np.unique(ids)) != len(ids):
        raise ValueError("Invalid retained count or duplicate source indices.")
    if len(set(hashes)) != len(rows) or len({row["group_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate images/linked groups remain in the split.")
    clean_map = {int(row["source_index"]): row for row in clean}
    if set(ids.tolist()) != set(clean_map) or not np.isin(labels, np.arange(10)).all():
        raise ValueError("Clean indices and split indices/labels do not agree.")
    if not np.isin(roles, ["train_pool", "validation", "test"]).all():
        raise ValueError("Unexpected split name.")
    for row in rows:
        if any(row[key] != clean_map[int(row["source_index"])][key]
               for key in ("label", "group_id", "sha256")):
            raise ValueError("Clean/split row mismatch.")
    for role in ("train_pool", "validation", "test"):
        actual = ids[roles == role]
        saved = np.load(directory / f"{role}_indices.npy", allow_pickle=False)
        if not np.array_equal(np.sort(saved), actual) or len(np.unique(saved)) != len(saved):
            raise ValueError(f"{role} .npy does not match splits.csv.")
        if set(labels[roles == role].tolist()) != set(range(10)):
            raise ValueError(f"{role} does not contain all ten classes.")
    with h5py.File(source, "r") as handle:
        images = handle["images"]
        if list(images.shape) != report["source_image_shape"] or images.dtype != np.uint8:
            raise ValueError("Raw image shape/dtype differs from the cleaning report.")
        if source.stat().st_size != report["source_size_bytes"]:
            raise ValueError("Raw file size changed since cleaning.")
        if not np.array_equal(np.asarray(handle["ans"][()]).reshape(-1)[ids], labels):
            raise ValueError("Raw labels no longer match the fixed split.")
    if smoke:
        # Uniform image subsampling, no class labels consulted. Separate cache/results.
        rng = np.random.default_rng(20261014)
        selected = []
        for role, limit in (("train_pool", 200), ("validation", 50), ("test", 50)):
            positions = np.flatnonzero(roles == role)
            selected.extend(rng.permutation(positions)[:limit].tolist())
        selected = np.array(sorted(selected), dtype=np.int64)
        ids, labels, roles = ids[selected], labels[selected], roles[selected]
        hashes = [hashes[i] for i in selected]
    return PreparedData(source, directory, ids, labels, roles, hashes,
                        sha256_file(directory / "splits.csv"), report)
