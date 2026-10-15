"""Read-only diagnosis of an existing Galaxy10 active-learning run.

Uses cached features, saved MLP checkpoints and the original train/validation
split. Never extracts CNN features, changes acquisition, or evaluates test data.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path


CLASS_NAMES = ["Disturbed", "Merging", "Round smooth", "In-between round smooth",
               "Cigar shaped smooth", "Barred spiral", "Unbarred tight spiral",
               "Unbarred loose spiral", "Edge-on without bulge", "Edge-on with bulge"]
C_GRID = [0.01, 0.1, 1.0, 10.0]


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


def fit_linear(train_x, train_y, validation_x, validation_y, out):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.exceptions import ConvergenceWarning
    scaler = StandardScaler().fit(train_x)  # Full training pool ONLY.
    tx, vx = scaler.transform(train_x), scaler.transform(validation_x)
    candidates, best = [], None
    for c in C_GRID:
        print(f"  Linear classifier C={c}: training on {len(tx):,} labels", flush=True)
        estimator = LogisticRegression(C=c, solver="lbfgs", max_iter=2000, tol=1e-4)
        warning_messages, converged = [], False
        for maximum in (2000, 4000):
            estimator.set_params(max_iter=maximum)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                estimator.fit(tx, train_y)
            warning_messages.extend(str(w.message) for w in caught)
            converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
            if converged:
                break
        if not np.array_equal(estimator.classes_, np.arange(10)):
            raise ValueError("Linear classifier did not fit all ten classes.")
        metrics = measure(validation_y, estimator.predict_proba(vx))
        row = {"C": c, "validation_macro_f1": metrics["macro_f1"],
               "validation_cross_entropy": metrics["cross_entropy"],
               "converged": converged, "iterations": int(estimator.n_iter_.max()),
               "max_iter": maximum, "warnings": " | ".join(warning_messages)}
        candidates.append(row)
        # Only converged models eligible; ties prefer lower validation CE then smaller C.
        rank = (metrics["macro_f1"], -metrics["cross_entropy"], -c)
        if converged and (best is None or rank > best[0]):
            best = (rank, estimator, row)
        print(f"    val macro-F1={metrics['macro_f1']:.4f}; converged={converged}", flush=True)
    write_csv(out / "linear_candidates.csv", candidates)
    if best is None:
        raise RuntimeError("No converged linear candidate. See linear_candidates.csv; no comparison was declared.")
    estimator, chosen = best[1], best[2]
    np.savez_compressed(out / "linear_model.npz", coefficients=estimator.coef_, intercept=estimator.intercept_,
                        classes=estimator.classes_, mean=scaler.mean_, scale=scaler.scale_, C=chosen["C"])
    write_json(out / "linear_selection.json", {"C_grid": C_GRID, "chosen": chosen,
               "selection": "validation macro-F1; ties validation CE then smaller C",
               "test_used": False, "class_weight": None, "training_labels": len(tx)})
    return estimator.predict_proba(tx), estimator.predict_proba(vx), chosen


def evaluate_and_export(out, name, seed, budget, positions, val, ids, y, p_train, p_val,
                        metric_rows, class_rows, errors, matrices):
    import numpy as np
    for split, pos, probability in (("labeled_train", positions, p_train), ("validation", val, p_val)):
        result = measure(y[pos], probability)
        metric_rows.append({"model": name, "seed": seed, "labels": budget, "split": split,
                            **{k: result[k] for k in ("accuracy", "macro_f1", "balanced_accuracy", "cross_entropy")}})
        for item in result["per_class"]:
            class_rows.append({"model": name, "seed": seed, "labels": budget, "split": split,
                               "class_name": CLASS_NAMES[item["class_id"]], **item})
        if split == "validation":
            matrix = np.asarray(result["confusion_matrix"])
            matrices.setdefault(name, []).append(matrix)
            tag = f"{name}_seed_{seed}"
            np.savez_compressed(out / "predictions" / f"{tag}.npz", source_indices=ids[pos],
                                truth=y[pos], probabilities=probability)
            np.savetxt(out / "confusion_matrices" / f"{tag}_counts.csv", matrix, fmt="%d", delimiter=",")
            normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
            np.savetxt(out / "confusion_matrices" / f"{tag}_row_normalized.csv", normalized, fmt="%.8f", delimiter=",")
            predicted = probability.argmax(axis=1)
            for local in np.flatnonzero(predicted != y[pos]):
                errors.append({"model": name, "seed": seed, "source_index": int(ids[pos[local]]),
                               "true_class": int(y[pos[local]]), "predicted_class": int(predicted[local]),
                               "predicted_confidence": float(probability[local, predicted[local]]),
                               "true_class_probability": float(probability[local, y[pos[local]]])})
    return measure(y[val], p_val)


def plot_and_report(out, rows, per_class, matrices, curves, chosen, config, run):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = ["full_reference", "linear_full", *config["strategies"]]
    plt.rcParams.update({"font.size": 10})

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / "figures" / (name + ".png"), dpi=160)
        fig.savefig(out / "figures" / (name + ".pdf"))
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    summary = []
    for model in models:
        item = {"model": model}
        for split in ("labeled_train", "validation"):
            values = [r for r in rows if r["model"] == model and r["split"] == split]
            item[split] = {key: float(np.mean([r[key] for r in values])) for key in ("macro_f1", "accuracy", "cross_entropy")}
            item[split]["macro_f1_std"] = float(np.std([r["macro_f1"] for r in values], ddof=1)) if len(values) > 1 else 0.0
        summary.append(item)
    x = np.arange(len(models))
    for split, offset, label in (("labeled_train", -.18, "Own labeled training set"), ("validation", .18, "Common validation set")):
        axes[0].bar(x + offset, [s[split]["macro_f1"] for s in summary], .36,
                    yerr=[s[split]["macro_f1_std"] for s in summary], capsize=3, label=label)
        axes[1].bar(x + offset, [s[split]["cross_entropy"] for s in summary], .36, label=label)
    for ax in axes:
        ax.set_xticks(x, models, rotation=25, ha="right")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Macro-F1 (10 classes)")
    axes[0].set_ylim(0, 1)
    axes[1].set_ylabel("Unweighted cross-entropy (eval mode)")
    fig.suptitle("Training vs validation: AL training sets differ; full models share all training labels")
    save(fig, "train_validation_comparison")

    fig, axes = plt.subplots(len(curves), 2, figsize=(12, 3.5 * len(curves)), squeeze=False)
    for r, (seed, record) in enumerate(curves):
        history = record["history"]
        epochs = [h["epoch"] for h in history]
        axes[r, 0].plot(epochs, [h["training_loss"] for h in history], label="Online train loss (dropout on)")
        axes[r, 0].plot(epochs, [h["validation_loss"] for h in history], label="Validation loss (eval)")
        axes[r, 1].plot(epochs, [h["validation_macro_f1"] for h in history], label="Validation macro-F1")
        for ax in axes[r]:
            ax.axvline(record["best_epoch"], color="gray", linestyle="--", label="Selected checkpoint")
            ax.set_xlabel("Epoch")
            ax.set_title(f"Full-reference MLP | seed {seed}")
            ax.legend(fontsize=8)
        axes[r, 0].set_ylabel("Cross-entropy")
        axes[r, 1].set_ylabel("Macro-F1")
    save(fig, "full_reference_training_curves")

    fig, ax = plt.subplots(figsize=(12, 6))
    class_mean = {}
    for model in models:
        class_mean[model] = np.asarray([np.mean([r["f1"] for r in per_class if r["model"] == model and r["split"] == "validation" and r["class_id"] == i]) for i in range(10)])
        ax.plot(range(10), class_mean[model], marker="o", label=model)
    ax.set_xticks(range(10), [f"{i}: {s}" for i, s in enumerate(CLASS_NAMES)], rotation=35, ha="right")
    ax.set_ylabel("Validation class F1, mean over MLP seeds")
    ax.set_ylim(0, 1)
    ax.legend(ncol=3)
    save(fig, "validation_per_class_f1")

    pairs = []
    for model in models:
        # Sum counts across seeds (repeated evaluation of SAME validation examples).
        aggregate = np.sum(matrices[model], axis=0)
        norm = aggregate / np.maximum(aggregate.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
        for i in range(10):
            for j in range(10):
                ax.text(j, i, f"{100 * norm[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > .5 else "black")
                if i != j:
                    pairs.append({"model": model, "true_class": i, "true_name": CLASS_NAMES[i],
                                  "predicted_class": j, "predicted_name": CLASS_NAMES[j],
                                  "row_fraction": float(norm[i, j]), "count_across_models": int(aggregate[i, j]),
                                  "evaluated_models": len(matrices[model])})
        ax.set_xticks(range(10))
        ax.set_yticks(range(10), [f"{i}: {s}" for i, s in enumerate(CLASS_NAMES)])
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(f"{model}: validation row percentages\nSeed counts pooled; not additional independent images")
        fig.colorbar(im, ax=ax, label="Fraction within true class")
        save(fig, f"confusion_{model}")
    pairs.sort(key=lambda r: (r["model"], -r["row_fraction"]))
    write_csv(out / "confusion_pairs.csv", pairs)
    write_json(out / "summary.json", {"models": summary, "linear_C": chosen["C"], "test_evaluated": False})

    lines = ["# Galaxy10 模型诊断报告", "", f"原实验：`{run.name}`", "",
             "仅使用原实验的训练集和验证集；未读取测试预测、未评估测试指标。未改动原实验文件。",
             "", "## 1. 同一验证集上的比较", "",
             "| Model | Training labels | Train macro-F1 | Validation macro-F1 ± seed SD | Train CE | Validation CE |",
             "|---|---:|---:|---:|---:|---:|"]
    for s in summary:
        labels = next(r["labels"] for r in rows if r["model"] == s["model"])
        t, v = s["labeled_train"], s["validation"]
        spread = f" ± {v['macro_f1_std']:.4f}" if s["model"] != "linear_full" else " (one deterministic fit)"
        lines.append(f"| {s['model']} | {labels} | {t['macro_f1']:.4f} | {v['macro_f1']:.4f}{spread} | {t['cross_entropy']:.4f} | {v['cross_entropy']:.4f} |")
    full, linear = summary[0], summary[1]
    gap = full["labeled_train"]["macro_f1"] - full["validation"]["macro_f1"]
    difference = linear["validation"]["macro_f1"] - full["validation"]["macro_f1"]
    lines.extend(["", f"Full-reference 的平均 train−validation macro-F1 差距为 **{gap:+.4f}**。",
                  f"线性分类器−full-reference MLP 的 validation macro-F1 差为 **{difference:+.4f}**；所选 C={chosen['C']}。", "",
                  "训练指标是在已选 checkpoint、eval 模式下重新计算的；历史训练 loss 是训练过程中、dropout 开启时的 batch 平均，两者不可直接当成同一条曲线。",
                  "AL 的 train 指标各自在被选中的训练样本上计算，样本难度及分布不同，不能据此直接排名选样策略。", "",
                  "## 2. 怎样读这些诊断", "",
                  "- 若 full-reference 训练表现明显高于验证表现，且验证 loss 后期反弹，应优先检查泛化问题、正则化和早停。单个差距不证明过拟合原因。",
                  "- 若训练和验证表现都有限，可能与特征可分性、分类器容量、优化或标签歧义有关。冻结特征上的线性/MLP 比较不能单独证明 ResNet 是瓶颈。",
                  "- 若线性模型接近或超过 MLP，可优先复核 MLP 的训练设置；若明显落后，说明当前非线性分类器有收益，但不代表它已达到最佳表现。",
                  "- 线性分类器使用全部 train_pool 标签，必须与 full-reference 比较。它不是 2,000-label 主动学习方法，不能据此声称节省标注。",
                  "- 线性 C 通过验证集挑选，原 MLP 也通过验证集早停；这是开发诊断，验证成绩存在选择偏差，不是新独立测试。", "",
                  "## 3. 各类别的验证 F1", "", "| Class | Random | Hybrid | Hybrid−Random | Full MLP | Linear |",
                  "|---|---:|---:|---:|---:|---:|"])
    for i in range(10):
        a, b = class_mean["random"][i], class_mean["hybrid"][i]
        lines.append(f"| {i}: {CLASS_NAMES[i]} | {a:.4f} | {b:.4f} | {b-a:+.4f} | {class_mean['full_reference'][i]:.4f} | {class_mean['linear_full'][i]:.4f} |")
    lines.extend(["", "重点检查 class 0（Disturbed）、class 4（Cigar shaped smooth）和 class 5（Barred spiral）的错误去向：", ""])
    for model in ("random", "hybrid", "full_reference", "linear_full"):
        for i in (0, 4, 5):
            top = sorted([p for p in pairs if p["model"] == model and p["true_class"] == i and p["row_fraction"] > 0], key=lambda p: -p["row_fraction"])[:2]
            detail = "; ".join(f"predicted {p['predicted_class']} ({p['predicted_name']}): {p['row_fraction']:.1%}" for p in top) or "No errors for this true class."
            lines.append(f"- {model}, true class {i}: {detail}")
    lines.extend(["", "## 4. 文件与实验边界", "",
                  "- `figures/full_reference_training_curves.png`：原训练过程及选中的 epoch。",
                  "- `figures/train_validation_comparison.png`：最终 checkpoint 的训练/验证指标。",
                  "- `figures/validation_per_class_f1.png`：各类别对照。",
                  "- `figures/confusion_*.png`：行归一化混淆矩阵，行是真实类别、列是预测类别。",
                  "- `diagnostic_metrics.csv` / `diagnostic_per_class.csv`：每个 seed 的明细。",
                  "- `validation_errors.csv`：所有验证错误的原始 source_index 与置信度，可回查原图；置信度未校准。",
                  "- `linear_candidates.csv`：固定 C 网格及收敛状态；不收敛的候选不参与选择。", "",
                  "MLP 误差条为已有 seed 的样本标准差，不是置信区间。线性模型只训练一份，不复制三份伪造重复实验。",
                  "同一验证样本在不同 seed 下重复评估，混淆矩阵汇总计数不代表独立样本数增加。",
                  "本项目原测试集已经查看过；后续改进仍在验证集上开发，原测试集不能重新称为未见过的独立确认集。",
                  "这一步没有训练新的 MLP 或新 CNN，因此只能缩小原因范围，不能给出因果证明。", ""])
    (out / "diagnostic_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--run", help="Existing completed research run, relative to project root or absolute.")
    parser.add_argument("--prepared-dir", help="Override moved prepared directory; checksum still must match.")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    import numpy as np
    import torch
    import sklearn
    import matplotlib
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    root = args.project_dir.resolve()
    run, locked = resolve_run(root, args.run)
    config = locked["config"]
    print(f"Read-only source run: {run}", flush=True)
    x, ids, y, roles, train, val, cache, prepared = load_inputs(root, run, locked, args.prepared_dir)
    # New folder only. No original files or previous diagnoses are overwritten.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = root / "outputs/model_diagnostics" / f"{run.name}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    for name in ("figures", "predictions", "confusion_matrices"):
        (out / name).mkdir()
    print(f"Verified features: {x.shape}; train={len(train):,}; validation={len(val):,}", flush=True)
    print(f"Output directory: {out}", flush=True)
    write_json(out / "diagnostic_protocol.json", {
        "source_run": str(run), "locked_protocol_sha256": sha256(run / "locked_protocol.json"),
        "feature_cache": str(cache), "split_sha256": locked["split_sha256"],
        "script_sha256": sha256(Path(__file__)), "test_evaluated": False,
        "linear": {"C_grid": C_GRID, "solver": "lbfgs", "tol": 1e-4, "max_iter": [2000, 4000],
                   "standardization_fit": "full train_pool only", "class_weight": None},
        "seeds": config["seeds"], "cpu_threads": args.threads,
        "versions": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
                     "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__}})
    metric_rows, class_rows, errors, matrices, curves, history_rows, gaps = [], [], [], {}, [], [], []
    with threadpool_limits(limits=args.threads):
        for seed in config["seeds"]:
            for name in ["full_reference", *config["strategies"]]:
                folder = run / f"seed_{seed}" / name
                if name == "full_reference":
                    record = read_json(folder / "reference.json")
                    if record["seed"] != seed or record["strategy"] != name or record["budget"] != len(train):
                        raise ValueError("Full-reference identity/budget mismatch.")
                    positions, checkpoint = train, folder / "reference.pt"
                    curves.append((seed, record))
                else:
                    record, positions, checkpoint = final_record(folder, seed, name, train, ids, y, config["max_labels"])
                print(f"  Diagnose {name}, seed={seed}, labels={len(positions):,}", flush=True)
                pt = mlp_predict(checkpoint, record["checkpoint_sha256"], x[positions])
                pv = mlp_predict(checkpoint, record["checkpoint_sha256"], x[val])
                result = evaluate_and_export(out, name, seed, len(positions), positions, val, ids, y, pt, pv,
                                             metric_rows, class_rows, errors, matrices)
                old = record["validation"]
                for key in ("macro_f1", "accuracy"):
                    if abs(result[key] - old[key]) > 1e-5:
                        raise ValueError(f"Recomputed validation {key} differs from original checkpoint for {name}, seed {seed}.")
                a, b = metric_rows[-2:]
                gaps.append({"model": name, "seed": seed, "labels": len(positions),
                             "train_minus_validation_macro_f1": a["macro_f1"] - b["macro_f1"],
                             "validation_minus_train_cross_entropy": b["cross_entropy"] - a["cross_entropy"]})
                for item in record["history"]:
                    history_rows.append({"model": name, "seed": seed, "labels": len(positions),
                                         "selected_epoch": record["best_epoch"], **item})
        pt, pv, chosen = fit_linear(x[train], y[train], x[val], y[val], out)
        evaluate_and_export(out, "linear_full", "single_fit", len(train), train, val, ids, y, pt, pv,
                            metric_rows, class_rows, errors, matrices)
    write_csv(out / "diagnostic_metrics.csv", metric_rows)
    write_csv(out / "diagnostic_per_class.csv", class_rows)
    write_csv(out / "training_history.csv", history_rows)
    write_csv(out / "train_validation_gaps.csv", gaps)
    errors.sort(key=lambda r: (r["model"], str(r["seed"]), -r["predicted_confidence"]))
    write_csv(out / "validation_errors.csv", errors, ["model", "seed", "source_index", "true_class",
              "predicted_class", "predicted_confidence", "true_class_probability"])
    plot_and_report(out, metric_rows, class_rows, matrices, curves, chosen, config, run)
    write_json(out / "RUN_COMPLETE.json", {"completed": True, "test_evaluated": False,
                                          "source_run": str(run), "report": "diagnostic_report.md"})
    write_json(root / "outputs/latest_diagnostic.json", {"diagnostic_directory": str(out), "source_run": str(run)})
    print(f"\nDONE: {out / 'diagnostic_report.md'}", flush=True)
    print("Original AL results and feature caches were not modified. Test was not evaluated.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(f"\nDIAGNOSIS STOPPED: {exc}", file=sys.stderr)
        if not isinstance(exc, KeyboardInterrupt):
            traceback.print_exc()
        print("Original experiment remains unchanged. A failed diagnostic folder without RUN_COMPLETE is incomplete.", file=sys.stderr)
        sys.exit(1)
