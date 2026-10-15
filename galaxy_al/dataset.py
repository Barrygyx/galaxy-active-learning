"""Validate and reuse the existing split; never re-split or rebalance it."""
from dataclasses import dataclass
from pathlib import Path
import h5py
import numpy as np
from .common import read_csv, read_json, sha256_file


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
