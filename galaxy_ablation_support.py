"""Input/checkpoint validators shared with the previous diagnostic package.
Vendored here to avoid changing the original AL modules or their fingerprints.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import traceback
import warnings
from pathlib import Path

CLASS_NAMES = ["Disturbed", "Merging", "Round smooth", "In-between round smooth",
               "Cigar shaped smooth", "Barred spiral", "Unbarred tight spiral",
               "Unbarred loose spiral", "Edge-on without bulge", "Edge-on with bulge"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve_run(root, requested=None):
    if requested:
        run = Path(requested)
        run = run if run.is_absolute() else root / run
    else:
        latest = root / "outputs/latest_run.json"
        value = read_json(latest) if latest.exists() else {}
        run = Path(value.get("run_directory", "__missing__"))
        if not run.is_absolute():
            run = root / run
        if not run.is_dir() and value.get("run_directory"):
            # Moving the project should not require editing old metadata.
            old_name = str(value["run_directory"]).replace("\\", "/").split("/")[-1]
            run = root / "outputs/active_learning" / old_name
        if value.get("smoke") or not run.is_dir():
            choices = sorted((root / "outputs/active_learning").glob("research_*/RUN_COMPLETE.json"))
            if len(choices) != 1:
                raise ValueError("Cannot identify one completed research run. Pass --run outputs/active_learning/research_... .")
            run = choices[0].parent
    if not (run / "RUN_COMPLETE.json").is_file():
        raise ValueError(f"Run has not completed: {run}")
    locked = read_json(run / "locked_protocol.json")
    if locked.get("smoke") or read_json(run / "RUN_COMPLETE.json").get("smoke"):
        raise ValueError("Smoke runs are not research results. Choose a completed research run.")
    if not locked["config"].get("full_label_reference"):
        raise ValueError("This diagnosis requires existing full-label reference checkpoints.")
    return run.resolve(), locked


def load_inputs(root, run, locked, prepared_override=None):
    import numpy as np
    config = locked["config"]
    prepared = Path(prepared_override or config["prepared_dir"])
    if not prepared.is_absolute():
        prepared = root / prepared
    report = read_json(prepared / "cleaning_report.json")
    for name, expected in report["artifact_sha256"].items():
        if sha256(prepared / name) != expected:
            raise ValueError(f"Prepared artifact changed: {name}")
    if sha256(prepared / "splits.csv") != locked["split_sha256"]:
        raise ValueError("Split differs from this completed experiment.")
    rows = sorted(read_csv(prepared / "splits.csv"), key=lambda r: int(r["source_index"]))
    ids = np.asarray([int(r["source_index"]) for r in rows], dtype=np.int64)
    y = np.asarray([int(r["label"]) for r in rows], dtype=np.int64)
    roles = np.asarray([r["split"] for r in rows])
    if len(ids) != report["clean_count"] or len(np.unique(ids)) != len(ids):
        raise ValueError("Invalid or duplicate split indices.")
    if not np.isin(y, np.arange(10)).all() or not np.isin(roles, ["train_pool", "validation", "test"]).all():
        raise ValueError("Invalid split labels/roles.")
    for role in ("train_pool", "validation", "test"):
        saved = np.load(prepared / f"{role}_indices.npy", allow_pickle=False)
        if not np.array_equal(np.sort(saved), ids[roles == role]):
            raise ValueError(f"Split CSV/npy mismatch: {role}")
    key = locked["feature_key"]
    cache = root / "cache/features" / key.split("_")[0]
    if not cache.is_dir():
        environment = read_json(run / "environment.json")
        cache = Path(environment["feature_cache"])
    manifest = read_json(cache / "feature_manifest.json")
    if manifest["signature"]["split_sha256"] != locked["split_sha256"] or manifest["signature"].get("smoke"):
        raise ValueError("Features belong to a different split or a smoke run.")
    feature_hash = sha256(cache / "features.npy")
    if feature_hash != manifest["features_sha256"] or key != cache.name + "_" + feature_hash[:16]:
        raise ValueError("Feature cache checksum/key mismatch.")
    cached_ids = np.load(cache / "source_indices.npy", allow_pickle=False)
    if not np.array_equal(cached_ids, ids) or hashlib.sha256(ids.tobytes()).hexdigest() != manifest["signature"]["ids_sha256"]:
        raise ValueError("Feature rows do not match the fixed split.")
    features = np.load(cache / "features.npy", mmap_mode="r", allow_pickle=False)
    done = np.load(cache / "done.npy", allow_pickle=False)
    if features.shape != (len(ids), 512) or done.shape != (len(ids),) or not done.all() or not np.isfinite(features).all():
        raise ValueError("Features are incomplete or invalid.")
    train, val = np.flatnonzero(roles == "train_pool"), np.flatnonzero(roles == "validation")
    if set(y[train]) != set(range(10)) or set(y[val]) != set(range(10)):
        raise ValueError("Full training/validation split must contain all ten classes.")
    return features, ids, y, roles, train, val, cache, prepared


def measure(y, probabilities):
    import numpy as np
    p = np.asarray(probabilities)
    y = np.asarray(y, dtype=np.int64)
    if not len(y) or p.shape != (len(y), 10) or not np.isfinite(p).all():
        raise ValueError("Expected nonempty labels and finite (N,10) probabilities.")
    if not np.isin(y, np.arange(10)).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Invalid labels/probabilities.")
    predicted = p.argmax(axis=1)
    cm = np.bincount(y * 10 + predicted, minlength=100).reshape(10, 10)
    tp, support = np.diag(cm).astype(float), cm.sum(axis=1)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros(10), where=cm.sum(axis=0) > 0)
    recall = np.divide(tp, support, out=np.zeros(10), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(10), where=precision + recall > 0)
    return {"accuracy": float((predicted == y).mean()), "macro_f1": float(f1.mean()),
            "cross_entropy": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean()),
            "balanced_accuracy": float(recall.mean()), "confusion_matrix": cm.tolist(),
            "per_class": [{"class_id": i, "precision": float(precision[i]), "recall": float(recall[i]),
                           "f1": float(f1[i]), "support": int(support[i])} for i in range(10)]}


def mlp_predict(checkpoint_path, expected_hash, features):
    import numpy as np
    import torch
    from torch import nn
    if sha256(checkpoint_path) != expected_hash:
        raise ValueError(f"Checkpoint changed: {checkpoint_path}")
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # Same 'layers' keys and eval behavior as the original Head; no encoder import.
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(nn.Linear(cp["input_dim"], cp["hidden_dim"]), nn.ReLU(),
                                        nn.Dropout(cp["dropout"]), nn.Linear(cp["hidden_dim"], 10))

        def forward(self, x):
            return self.layers(x)
    network = Head()
    network.load_state_dict(cp["state_dict"], strict=True)
    network.eval()
    mean, std = cp["mean"].numpy(), cp["std"].numpy()
    if cp["input_dim"] != features.shape[1] or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid checkpoint normalization/dimensions.")
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(features), 4096):
            x = torch.tensor((features[start:start + 4096] - mean) / std, dtype=torch.float32)
            chunks.append(network(x).softmax(1).numpy())
    return np.concatenate(chunks)


def final_record(folder, seed, strategy, train, ids, y, expected_budget):
    import numpy as np
    records = sorted(folder.glob("round_[0-9][0-9][0-9].json"))
    if not records or not (folder / "trajectory_complete.json").exists():
        raise ValueError(f"Missing completed trajectory: {folder}")
    cumulative = []
    allowed = set(train.tolist())
    for round_id, path in enumerate(records):
        record = read_json(path)
        selected = record["queried_positions"]
        if record["seed"] != seed or record["strategy"] != strategy or record["round"] != round_id:
            raise ValueError("Saved trajectory identity mismatch.")
        if len(set(selected)) != len(selected) or not set(selected) <= allowed or set(selected) & set(cumulative):
            raise ValueError("Non-training, repeated or invalid queried positions.")
        if record["queried_source_indices"] != ids[selected].tolist() or record["revealed_labels"] != y[selected].tolist():
            raise ValueError("Saved queries do not match feature rows/labels.")
        cumulative.extend(selected)
        if record["labeled_positions"] != cumulative or record["budget"] != len(cumulative):
            raise ValueError("Invalid cumulative labeled set/budget.")
    if record["budget"] != expected_budget:
        raise ValueError("Final checkpoint does not match locked final budget.")
    return record, np.asarray(cumulative, dtype=np.int64), folder / f"round_{record['round']:03d}.pt"
