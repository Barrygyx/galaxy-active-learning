"""Label-budget-matched trajectories with paired seeds and resumable rounds."""
from collections import Counter
from pathlib import Path
import time
import numpy as np
import torch
from .acquisition import select_batch, unit_features
from .common import code_hashes, fingerprint, read_json, sha256_file, write_csv, write_json
from .metrics import classification_metrics
from .model import load_checkpoint, predict, save_checkpoint, train_head


class LabelOracle:
    """Ground truth is exposed only for explicit train-pool queries."""
    def __init__(self, labels, train_positions):
        self._labels = labels
        self.allowed = set(map(int, train_positions))
        self.queried = set()

    def query(self, positions):
        positions = list(map(int, positions))
        if len(set(positions)) != len(positions) or not set(positions) <= self.allowed:
            raise ValueError("Oracle query contains duplicates or non-training samples.")
        if set(positions) & self.queried:
            raise ValueError("Oracle was asked to label a previously queried image.")
        self.queried.update(positions)
        return self._labels[positions].copy()

    def revealed_labels(self, positions):
        positions = list(map(int, positions))
        if not set(positions) <= self.queried:
            raise ValueError("Attempted access to unrevealed labels.")
        return self._labels[positions].copy()


def make_budgets(config):
    budgets = [config["initial_labels"]]
    while budgets[-1] < config["max_labels"]:
        budgets.append(min(config["max_labels"], budgets[-1] + config["query_batch_size"]))
    return budgets


def run_trajectory(directory, strategy, seed, features, normalized, data, config, device):
    directory.mkdir(parents=True, exist_ok=True)
    train_positions = data.positions("train_pool")
    validation_positions = data.positions("validation")
    oracle = LabelOracle(data.labels, train_positions)
    initial = np.random.default_rng(seed).choice(train_positions, config["initial_labels"], replace=False)
    labeled = np.empty(0, dtype=np.int64)
    records, queries = [], []
    previous_checkpoint = None
    for round_id, budget in enumerate(make_budgets(config)):
        record_path = directory / f"round_{round_id:03d}.json"
        checkpoint_path = directory / f"round_{round_id:03d}.pt"
        if record_path.exists():
            record = read_json(record_path)
            selected = np.array(record["queried_positions"], dtype=np.int64)
            if record["budget"] != budget or record["seed"] != seed or record["strategy"] != strategy:
                raise ValueError("Saved round does not match this experiment.")
            if not checkpoint_path.exists() or sha256_file(checkpoint_path) != record["checkpoint_sha256"]:
                raise ValueError(f"Missing or changed checkpoint: {checkpoint_path}")
            if round_id == 0 and not np.array_equal(selected, initial):
                raise ValueError("Initial labeled set changed.")
            if len(selected) != budget - len(labeled):
                raise ValueError("Saved query count does not match annotation budget.")
            query_labels = oracle.query(selected)
            if not np.array_equal(data.ids[selected], np.array(record["queried_source_indices"])):
                raise ValueError("Saved positions no longer match original source indices.")
            if not np.array_equal(query_labels, np.array(record["revealed_labels"])):
                raise ValueError("Saved revealed labels changed.")
            labeled = np.concatenate([labeled, selected])
            if record["labeled_positions"] != labeled.tolist():
                raise ValueError("Saved cumulative labeled set changed.")
            previous_checkpoint = load_checkpoint(checkpoint_path)
            print(f"  resume {strategy} seed={seed} round={round_id} labels={budget}", flush=True)
        else:
            started = time.monotonic()
            available = np.setdiff1d(train_positions, labeled, assume_unique=True)
            probability = None
            if round_id == 0:
                selected = initial.copy()
            else:
                if strategy in ("uncertainty", "hybrid"):
                    probability = predict(previous_checkpoint, features[available], device)
                selected = select_batch(
                    strategy, normalized, available, labeled, budget - len(labeled),
                    seed=seed * 100003 + round_id, probabilities=probability,
                    multiplier=config["hybrid_candidate_multiplier"],
                )
            acquisition_seconds = time.monotonic() - started
            query_labels = oracle.query(selected)
            labeled = np.concatenate([labeled, selected])
            if len(labeled) != budget or len(oracle.queried) != budget:
                raise RuntimeError("Cumulative label budget was violated.")
            # Same initialization/shuffle seed at each round across all strategies.
            training_seed = seed * 1000 + round_id
            started = time.monotonic()
            checkpoint, history = train_head(
                features[labeled], oracle.revealed_labels(labeled),
                features[validation_positions], data.labels[validation_positions],
                config, training_seed, device,
            )
            training_seconds = time.monotonic() - started
            validation = classification_metrics(
                data.labels[validation_positions], predict(checkpoint, features[validation_positions], device))
            known_counts = Counter(map(int, oracle.revealed_labels(labeled)))
            record = {
                "round": round_id, "budget": budget, "seed": seed, "strategy": strategy,
                "queried_positions": selected.tolist(), "queried_source_indices": data.ids[selected].tolist(),
                "revealed_labels": query_labels.tolist(), "labeled_positions": labeled.tolist(),
                "labeled_class_counts": {str(i): known_counts[i] for i in range(10)},
                "training_seed": training_seed, "best_epoch": checkpoint["best_epoch"],
                "validation": validation, "history": history,
                "acquisition_seconds": acquisition_seconds, "training_seconds": training_seconds,
            }
            save_checkpoint(checkpoint_path, checkpoint)
            record["checkpoint_sha256"] = sha256_file(checkpoint_path)
            # This JSON is the round commit: interruptions before it simply retrain the round.
            write_json(record_path, record)
            previous_checkpoint = checkpoint
            print(f"  {strategy:11s} seed={seed} round={round_id:02d} labels={budget:4d} "
                  f"val macro-F1={validation['macro_f1']:.4f} "
                  f"val rare recall={validation['per_class'][config['rare_class']]['recall']:.3f} "
                  f"time={acquisition_seconds + training_seconds:.1f}s", flush=True)
        records.append(record)
        for position, label in zip(selected, query_labels):
            queries.append({"round": round_id, "budget": budget, "source_index": int(data.ids[position]),
                            "revealed_label": int(label), "initial_sample": round_id == 0})
    write_csv(directory / "queries.csv", queries)
    write_json(directory / "trajectory_complete.json", {
        "seed": seed, "strategy": strategy, "budgets": make_budgets(config),
        "initial_source_indices": data.ids[initial].tolist(), "total_queried": len(oracle.queried),
    })
    return records


def run_reference(directory, seed, features, data, config, device):
    directory.mkdir(parents=True, exist_ok=True)
    record_path, checkpoint_path = directory / "reference.json", directory / "reference.pt"
    if record_path.exists():
        record = read_json(record_path)
        if not checkpoint_path.exists() or sha256_file(checkpoint_path) != record["checkpoint_sha256"]:
            raise ValueError("Full-label reference checkpoint was changed.")
        return record
    train, validation = data.positions("train_pool"), data.positions("validation")
    checkpoint, history = train_head(features[train], data.labels[train], features[validation],
                                     data.labels[validation], config, seed * 1000 + 999, device)
    val = classification_metrics(data.labels[validation], predict(checkpoint, features[validation], device))
    save_checkpoint(checkpoint_path, checkpoint)
    record = {"seed": seed, "strategy": "full_reference", "budget": len(train), "round": -1,
              "validation": val, "history": history, "best_epoch": checkpoint["best_epoch"],
              "checkpoint_sha256": sha256_file(checkpoint_path)}
    write_json(record_path, record)
    print(f"  full-label reference seed={seed} validation macro-F1={val['macro_f1']:.4f}", flush=True)
    return record


def run_experiments(root, features, feature_key, data, config, device, smoke=False):
    identity = {"config": config, "feature_key": feature_key, "split_sha256": data.split_sha256,
                "code": code_hashes(["acquisition.py", "experiment.py", "model.py", "metrics.py"]),
                "smoke": smoke}
    run_key = fingerprint(identity)[:16]
    directory = root / "outputs/active_learning" / (("smoke_" if smoke else "research_") + run_key)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "locked_protocol.json", identity)
    write_json(root / "outputs/latest_run.json", {"run_directory": str(directory), "smoke": smoke})
    normalized = unit_features(features)
    all_records = {}
    print(f"\nExperiment directory: {directory}", flush=True)
    print("Acquisition/training phase: test labels are not evaluated in this phase.", flush=True)
    for seed in config["seeds"]:
        for strategy in config["strategies"]:
            folder = directory / f"seed_{seed}" / strategy
            records = run_trajectory(folder, strategy, seed, features, normalized, data, config, device)
            all_records[(seed, strategy)] = records
    references = {}
    if config["full_label_reference"]:
        for seed in config["seeds"]:
            references[seed] = run_reference(directory / f"seed_{seed}/full_reference",
                                              seed, features, data, config, device)
    # Test evaluation is intentionally below the ENTIRE acquisition phase.
    for seed in config["seeds"]:
        initial_sets = []
        for strategy in config["strategies"]:
            folder = directory / f"seed_{seed}" / strategy
            if not (folder / "trajectory_complete.json").exists():
                raise RuntimeError("Refusing test evaluation before ALL trajectories finish.")
            initial_sets.append(all_records[(seed, strategy)][0]["queried_source_indices"])
        if any(value != initial_sets[0] for value in initial_sets[1:]):
            raise RuntimeError("Strategies did not share the initial labeled set.")
    write_json(directory / "acquisition_phase_complete.json", {
        "all_trajectories_complete": True, "budgets": make_budgets(config),
        "test_usage": "Post-hoc evaluation only; no acquisition or parameter changes based on test results.",
    })
    print("\nAll trajectories are locked. Evaluating held-out test checkpoints now.", flush=True)
    test_positions = data.positions("test")
    metrics_rows, per_class_rows = [], []
    for seed in config["seeds"]:
        for strategy in config["strategies"] + (["full_reference"] if references else []):
            records = [references[seed]] if strategy == "full_reference" else all_records[(seed, strategy)]
            folder = directory / f"seed_{seed}" / strategy
            for record in records:
                name = "reference.pt" if strategy == "full_reference" else f"round_{record['round']:03d}.pt"
                checkpoint = load_checkpoint(folder / name)
                probabilities = predict(checkpoint, features[test_positions], device)
                test = classification_metrics(data.labels[test_positions], probabilities)
                write_json(folder / (name.replace(".pt", "_test.json")), test)
                for role, measured in (("validation", record["validation"]), ("test", test)):
                    rare = measured["per_class"][config["rare_class"]]
                    metrics_rows.append({
                        "seed": seed, "strategy": strategy, "round": record["round"], "labels": record["budget"],
                        "split": role, "accuracy": measured["accuracy"], "macro_f1": measured["macro_f1"],
                        "balanced_accuracy": measured["balanced_accuracy"], "rare_recall": rare["recall"],
                        "rare_precision": rare["precision"], "rare_f1": rare["f1"],
                    })
                    for item in measured["per_class"]:
                        per_class_rows.append({"seed": seed, "strategy": strategy, "round": record["round"],
                                               "labels": record["budget"], "split": role, **item})
                if record is records[-1]:
                    np.savez_compressed(folder / "final_test_predictions.npz", source_indices=data.ids[test_positions],
                                        truth=data.labels[test_positions], probabilities=probabilities)
    write_csv(directory / "metrics.csv", metrics_rows)
    write_csv(directory / "per_class_metrics.csv", per_class_rows)
    labeled_rows = []
    for (seed, strategy), records in all_records.items():
        for record in records:
            for label, count in record["labeled_class_counts"].items():
                labeled_rows.append({"seed": seed, "strategy": strategy, "labels": record["budget"],
                                     "class_id": int(label), "queried_class_count": count})
    write_csv(directory / "labeled_class_counts.csv", labeled_rows)
    return directory, metrics_rows
