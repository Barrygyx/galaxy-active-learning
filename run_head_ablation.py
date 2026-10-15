"""Fixed-label classifier ablation: original subsets, cached features, validation only."""
import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import sys
import traceback
import warnings
from pathlib import Path
import galaxy_ablation_support as io


MLP_VARIANTS = {
    "mlp_early": {},
    "mlp_small32": {"hidden_dim": 32},
    "mlp_dropout03": {"dropout": .3},
    "mlp_wd1e3": {"weight_decay": .001},
}
LINEAR_VARIANTS = {"linear_C0p001": .001, "linear_C0p01": .01, "linear_C0p1": .1}
VARIANTS = ["original_saved", *MLP_VARIANTS, *LINEAR_VARIANTS]


def atomic_json(path, value):
    temp = Path(path).with_name(Path(path).name + ".tmp")
    io.write_json(temp, value)
    os.replace(temp, path)


def digest_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class EarlyStop:
    """Separate progress thresholds from exact checkpoint ranking.

    Stop when either validation F1 or CE has not meaningfully improved for the
    specified patience. A CE tie is not an improvement. Never inspect test data.
    """
    def __init__(self, patience=3, f1_delta=.001, loss_delta=.0001):
        self.patience, self.f1_delta, self.loss_delta = patience, f1_delta, loss_delta
        self.f1_anchor, self.loss_anchor = -float("inf"), float("inf")
        self.f1_stale = self.loss_stale = 0

    def update(self, f1, loss):
        if f1 > self.f1_anchor + self.f1_delta:
            self.f1_anchor, self.f1_stale = f1, 0
        else:
            self.f1_stale += 1
        if loss < self.loss_anchor - self.loss_delta:
            self.loss_anchor, self.loss_stale = loss, 0
        else:
            self.loss_stale += 1
        if self.loss_stale >= self.patience:
            return "validation_loss_no_meaningful_improvement"
        if self.f1_stale >= self.patience:
            return "validation_f1_no_meaningful_improvement"
        return None


def train_mlp(tx, ty, vx, vy, settings, seed):
    import numpy as np
    import torch
    from torch import nn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    mean = np.asarray(tx, dtype=np.float64).mean(0).astype(np.float32)
    std = np.asarray(tx, dtype=np.float64).std(0).astype(np.float32)
    std[std < 1e-6] = 1
    train = torch.tensor((tx - mean) / std, dtype=torch.float32)
    val = torch.tensor((vx - mean) / std, dtype=torch.float32)
    targets = torch.tensor(ty, dtype=torch.long)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(nn.Linear(tx.shape[1], settings["hidden_dim"]), nn.ReLU(),
                                        nn.Dropout(settings["dropout"]), nn.Linear(settings["hidden_dim"], 10))

        def forward(self, x):
            return self.layers(x)

    model = Head()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    stopper = EarlyStop(settings["patience"], settings["f1_delta"], settings["loss_delta"])
    history, best_state = [], None
    best_f1, best_ce, best_epoch = -1, float("inf"), 0
    stop_reason = "max_epochs_reached"
    for epoch in range(1, settings["max_epochs"] + 1):
        model.train()
        order = torch.randperm(len(train))
        online = 0.0
        for start in range(0, len(train), settings["batch_size"]):
            batch = order[start:start + settings["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(train[batch]), targets[batch])
            loss.backward()
            optimizer.step()
            online += loss.item() * len(batch)
        model.eval()
        with torch.inference_mode():
            train_probability = model(train).softmax(1).numpy()
            val_probability = model(val).softmax(1).numpy()
        mt, mv = io.measure(ty, train_probability), io.measure(vy, val_probability)
        f1, ce = mv["macro_f1"], mv["cross_entropy"]
        # Checkpoint ranking uses exact observed F1, CE breaks ties. Stop thresholds
        # affect training duration, not which observed checkpoint is retained.
        if f1 > best_f1 + 1e-12 or (abs(f1 - best_f1) <= 1e-12 and ce < best_ce):
            best_state = copy.deepcopy(model.state_dict())
            best_f1, best_ce, best_epoch = f1, ce, epoch
        history.append({"epoch": epoch, "online_training_loss": online / len(train),
                        "train_eval_macro_f1": mt["macro_f1"], "train_eval_loss": mt["cross_entropy"],
                        "validation_macro_f1": f1, "validation_loss": ce,
                        "validation_class4_f1": mv["per_class"][4]["f1"],
                        "validation_class5_f1": mv["per_class"][5]["f1"]})
        reason = stopper.update(f1, ce)
        print(f"    epoch {epoch:02d}: train F1={mt['macro_f1']:.4f}, val F1={f1:.4f}, val CE={ce:.4f}", flush=True)
        if reason:
            stop_reason = reason
            break
    checkpoint = {"state_dict": best_state, "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
                  "input_dim": int(tx.shape[1]), "hidden_dim": int(settings["hidden_dim"]),
                  "dropout": float(settings["dropout"]), "best_epoch": best_epoch, "training_seed": seed}
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        pt, pv = model(train).softmax(1).numpy(), model(val).softmax(1).numpy()
    return checkpoint, pt, pv, history, {"best_epoch": best_epoch, "epochs_run": len(history), "stop_reason": stop_reason}


def train_linear(tx, ty, vx, c):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    scaler = StandardScaler().fit(tx)
    train, val = scaler.transform(tx), scaler.transform(vx)
    model = LogisticRegression(C=c, solver="lbfgs", max_iter=2000, tol=1e-4)
    messages, converged = [], False
    for maximum in (2000, 4000):
        model.set_params(max_iter=maximum)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(train, ty)
        messages.extend(str(w.message) for w in caught)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        if converged:
            break
    # Fixed ten outputs, even if an original random initial subset missed a class.
    def expand(values):
        result = np.zeros((len(values), 10), dtype=np.float64)
        result[:, model.classes_.astype(int)] = values
        return result
    if len(model.classes_) < 2:
        raise ValueError("Linear classifier needs at least two observed training classes.")
    export = {"coefficients": model.coef_, "intercept": model.intercept_, "mean": scaler.mean_,
              "scale": scaler.scale_, "classes": model.classes_, "C": np.asarray(c)}
    return export, expand(model.predict_proba(train)), expand(model.predict_proba(val)), {
        "converged": converged, "eligible": converged, "warnings": messages, "iterations": int(model.n_iter_.max()),
        "epochs_run": None, "best_epoch": None, "stop_reason": "lbfgs_converged" if converged else "lbfgs_not_converged"}


def select_records(run, locked, train, ids, y, budget):
    """Take a real budget point from the original trajectory, never resample it."""
    import numpy as np
    selected = []
    for seed in locked["config"]["seeds"]:
        for strategy in locked["config"]["strategies"]:
            folder = run / f"seed_{seed}" / strategy
            io.final_record(folder, seed, strategy, train, ids, y, locked["config"]["max_labels"])
            choices = [(p, io.read_json(p)) for p in sorted(folder.glob("round_[0-9][0-9][0-9].json"))]
            matching = [(p, r) for p, r in choices if r["budget"] == budget]
            if len(matching) != 1:
                raise ValueError(f"Budget {budget} is not a saved round for {strategy}, seed {seed}. Available: {[r['budget'] for _, r in choices]}")
            record_path, record = matching[0]
            checkpoint = folder / f"round_{record['round']:03d}.pt"
            if io.sha256(checkpoint) != record["checkpoint_sha256"]:
                raise ValueError(f"Changed original checkpoint: {checkpoint}")
            pos = np.asarray(record["labeled_positions"], dtype=np.int64)
            selected.append({"seed": seed, "strategy": strategy, "record": record, "positions": pos,
                             "checkpoint": checkpoint, "record_path": record_path,
                             "ids_sha256": hashlib.sha256(ids[pos].tobytes()).hexdigest()})
    return selected


def save_npz(path, **arrays):
    import numpy as np
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def fit_or_resume(folder, item, variant, tx, ty, vx, vy, ids_train, ids_val, settings, identity):
    import numpy as np
    import torch
    folder.mkdir(parents=True, exist_ok=True)
    commit = folder / "result.json"
    expected = {"protocol_sha256": identity, "seed": item["seed"], "strategy": item["strategy"],
                "variant": variant, "labeled_ids_sha256": item["ids_sha256"], "labels": len(ty)}
    if commit.exists():
        result = io.read_json(commit)
        if result["identity"] != expected:
            raise ValueError("Committed fit identity changed; refusing to mix results.")
        for name, digest in result["artifact_sha256"].items():
            if io.sha256(folder / name) != digest:
                raise ValueError(f"Committed ablation artifact changed: {folder / name}")
        print(f"  resume {item['strategy']} seed={item['seed']} {variant}", flush=True)
        return result
    print(f"  train {item['strategy']} seed={item['seed']} {variant}; labels={len(ty)}", flush=True)
    history = []
    if variant == "original_saved":
        pt = io.mlp_predict(item["checkpoint"], item["record"]["checkpoint_sha256"], tx)
        pv = io.mlp_predict(item["checkpoint"], item["record"]["checkpoint_sha256"], vx)
        mv = io.measure(vy, pv)
        for key in ("macro_f1", "accuracy"):
            if abs(mv[key] - item["record"]["validation"][key]) > 1e-5:
                raise ValueError("Original validation results did not reproduce.")
        shutil.copyfile(item["checkpoint"], folder / "model.pt")
        status = {"best_epoch": item["record"]["best_epoch"], "epochs_run": len(item["record"]["history"]),
                  "stop_reason": "reused_original_checkpoint", "converged": None, "eligible": True}
        history = item["record"]["history"]
    elif variant in MLP_VARIANTS:
        params = {**settings, **MLP_VARIANTS[variant]}
        cp, pt, pv, history, status = train_mlp(tx, ty, vx, vy, params, item["record"]["training_seed"])
        temp = folder / "model.pt.tmp"
        torch.save(cp, temp)
        os.replace(temp, folder / "model.pt")
        status.update({"eligible": True, "converged": None})
    else:
        model, pt, pv, status = train_linear(tx, ty, vx, LINEAR_VARIANTS[variant])
        save_npz(folder / "linear_model.npz", **model)
    save_npz(folder / "predictions.npz", train_source_indices=ids_train, validation_source_indices=ids_val,
             train_probabilities=pt, validation_probabilities=pv)
    atomic_json(folder / "history.json", history)
    artifacts = ["predictions.npz", "history.json", "linear_model.npz" if variant in LINEAR_VARIANTS else "model.pt"]
    result = {"identity": expected, "train": io.measure(ty, pt), "validation": io.measure(vy, pv),
              "status": status, "settings": ({**settings, **MLP_VARIANTS.get(variant, {})} if variant in MLP_VARIANTS
                  else {"C": LINEAR_VARIANTS[variant]} if variant in LINEAR_VARIANTS else {"source": str(item["checkpoint"])}),
              "artifact_sha256": {name: io.sha256(folder / name) for name in artifacts}}
    atomic_json(commit, result)  # Last write commits this fit; incomplete fits retrain on next run.
    return result


def export_results(out, results, strategies, seeds, budget, protocol):
    import numpy as np
    rows, classes, pairs, histories, counts = [], [], [], [], []
    for (strategy, seed, variant), result in results.items():
        for split in ("train", "validation"):
            m = result[split]
            rows.append({"strategy": strategy, "seed": seed, "variant": variant, "labels": budget,
                         "split": split, **{k: m[k] for k in ("macro_f1", "accuracy", "balanced_accuracy", "cross_entropy")},
                         "rare_f1": m["per_class"][4]["f1"], "rare_recall": m["per_class"][4]["recall"],
                         "class5_f1": m["per_class"][5]["f1"], **result["status"]})
            for c in m["per_class"]:
                classes.append({"strategy": strategy, "seed": seed, "variant": variant, "labels": budget,
                                "split": split, "class_name": io.CLASS_NAMES[c["class_id"]], **c})
        if variant == "original_saved":
            counts.extend({"strategy": strategy, "seed": seed, "class_id": c["class_id"], "labels": c["support"]}
                          for c in result["train"]["per_class"])
        else:
            base = results[(strategy, seed, "original_saved")]
            pairs.append({"strategy": strategy, "seed": seed, "variant": variant,
                          "validation_macro_f1_difference": result["validation"]["macro_f1"] - base["validation"]["macro_f1"],
                          "validation_ce_difference": result["validation"]["cross_entropy"] - base["validation"]["cross_entropy"],
                          "class4_f1_difference": result["validation"]["per_class"][4]["f1"] - base["validation"]["per_class"][4]["f1"],
                          "class5_f1_difference": result["validation"]["per_class"][5]["f1"] - base["validation"]["per_class"][5]["f1"]})
            if variant in MLP_VARIANTS:
                path = out / f"seed_{seed}" / strategy / variant / "history.json"
                histories.extend({"strategy": strategy, "seed": seed, "variant": variant, **r} for r in io.read_json(path))
    for name, values in (("metrics.csv", rows), ("per_class_metrics.csv", classes),
                         ("paired_vs_original.csv", pairs), ("training_history.csv", histories), ("labeled_class_counts.csv", counts)):
        # Status fields differ between linear and MLP: use an explicit union of columns.
        fields = list(dict.fromkeys(k for row in values for k in row))
        io.write_csv(out / name, values, fields)
    summary = []
    for strategy in strategies:
        for variant in VARIANTS:
            group = [r for r in rows if r["strategy"] == strategy and r["variant"] == variant and r["split"] == "validation"]
            item = {"strategy": strategy, "variant": variant, "seeds": len(seeds),
                    "eligible": all(bool(r["eligible"]) for r in group)}
            for key in ("macro_f1", "cross_entropy", "rare_f1", "rare_recall", "class5_f1"):
                item[key + "_mean"] = float(np.mean([r[key] for r in group]))
                item[key + "_std"] = float(np.std([r[key] for r in group], ddof=1)) if len(group) > 1 else 0.0
            summary.append(item)
    io.write_csv(out / "metrics_mean_std.csv", summary)
    ranking = []
    for variant in VARIANTS:
        group = [r for r in summary if r["variant"] == variant]
        ranking.append({"variant": variant, "eligible": all(r["eligible"] for r in group),
                        "mean_validation_macro_f1": float(np.mean([r["macro_f1_mean"] for r in group])),
                        "mean_validation_ce": float(np.mean([r["cross_entropy_mean"] for r in group]))})
    ranking.sort(key=lambda r: (not r["eligible"], -r["mean_validation_macro_f1"], r["mean_validation_ce"], VARIANTS.index(r["variant"])))
    # One shared recommendation; never optimize a different head for each acquisition strategy.
    chosen = next(r for r in ranking if r["eligible"])
    atomic_json(out / "shared_validation_selection.json", {"selected_variant": chosen["variant"], "ranking": ranking,
                "rule": "mean validation macro-F1 across strategies/seeds, then mean CE; one shared variant",
                "test_used": False, "confirmatory": False})
    make_plots(out, summary, rows, strategies, seeds)
    make_report(out, summary, ranking, pairs, rows, chosen, budget, protocol)


def make_plots(out, summary, rows, strategies, seeds):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = out / "figures"
    figures.mkdir(exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figures / (name + ".png"), dpi=150)
        fig.savefig(figures / (name + ".pdf"))
        plt.close(fig)

    def bars(keys, labels, name):
        fig, axes = plt.subplots(1, len(keys), figsize=(9 * len(keys), 6), squeeze=False)
        x, width = np.arange(len(VARIANTS)), .8 / len(strategies)
        for ax, key, ylabel in zip(axes[0], keys, labels):
            for j, strategy in enumerate(strategies):
                group = [next(r for r in summary if r["strategy"] == strategy and r["variant"] == v) for v in VARIANTS]
                ax.bar(x + (j - (len(strategies) - 1) / 2) * width, [r[key + "_mean"] for r in group], width,
                       yerr=[r[key + "_std"] for r in group], capsize=2, label=strategy)
            ax.set_xticks(x, VARIANTS, rotation=35, ha="right")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=9)
        fig.suptitle("Fixed original labeled subsets; validation only; error bars = seed sample SD")
        save(fig, name)
    bars(["macro_f1"], ["Validation macro-F1"], "validation_model_comparison")
    bars(["rare_f1", "class5_f1"], ["Class 4 validation F1", "Class 5 validation F1"], "class4_class5_comparison")
    bars(["cross_entropy"], ["Validation cross-entropy (lower is better)"], "validation_probability_loss")
    fig, ax = plt.subplots(figsize=(12, 5))
    for j, strategy in enumerate(strategies):
        means = []
        for variant in VARIANTS:
            gaps = [results_gap(rows, strategy, seed, variant) for seed in seeds]
            means.append(float(np.mean(gaps)))
        width = .8 / len(strategies)
        ax.bar(np.arange(len(VARIANTS)) + (j - (len(strategies)-1)/2)*width, means, width, label=strategy)
    ax.set_xticks(np.arange(len(VARIANTS)), VARIANTS, rotation=35, ha="right")
    ax.set_ylabel("Train minus validation macro-F1")
    ax.set_title("Own labeled train sets; a smaller gap alone does not imply a better model")
    ax.legend()
    save(fig, "train_validation_gaps")
    for strategy in strategies:
        for seed in seeds:
            fig, axes = plt.subplots(2, 2, figsize=(13, 8))
            for index, variant in enumerate(MLP_VARIANTS):
                h = io.read_json(out / f"seed_{seed}" / strategy / variant / "history.json")
                result = io.read_json(out / f"seed_{seed}" / strategy / variant / "result.json")
                epochs = [r["epoch"] for r in h]
                color = f"C{index}"
                axes[0, 0].plot(epochs, [r["train_eval_loss"] for r in h], "--", color=color, alpha=.55)
                axes[0, 0].plot(epochs, [r["validation_loss"] for r in h], color=color, label=variant)
                for ax, key in ((axes[0, 1], "validation_macro_f1"), (axes[1, 0], "validation_class4_f1"), (axes[1, 1], "validation_class5_f1")):
                    ax.plot(epochs, [r[key] for r in h], color=color, label=variant)
                    best_epoch = result["status"]["best_epoch"]
                    ax.scatter([best_epoch], [h[best_epoch-1][key]], color=color, marker="x", s=70)
            for ax, title in zip(axes.flat, ["CE: validation solid / train eval dashed", "Validation macro-F1 (x = selected epoch)", "Class 4 validation F1", "Class 5 validation F1"]):
                ax.set_title(title)
                ax.set_xlabel("Epoch")
                ax.legend(fontsize=8)
            fig.suptitle(f"{strategy}, seed {seed}: same labeled subset, new heads only")
            save(fig, f"training_curves_{strategy}_seed_{seed}")


def results_gap(rows, strategy, seed, variant):
    values = {r["split"]: r for r in rows if r["strategy"] == strategy and r["seed"] == seed and r["variant"] == variant}
    return values["train"]["macro_f1"] - values["validation"]["macro_f1"]


def make_report(out, summary, ranking, pairs, rows, chosen, budget, protocol):
    import numpy as np
    lines = ["# 固定标注子集：分类器与早停对照", "", f"原实验：`{Path(protocol['source_run']).name}`；每个 seed/策略使用其原轨迹中的 **{budget} 个标签**。", "",
             "训练和评估已完成；只使用原验证集，没有计算新的测试指标。", "",
             "## 1. 所有策略共用的候选排名", "", "| Variant | Mean validation macro-F1 | Mean validation CE | Eligible |",
             "|---|---:|---:|---|"]
    for row in ranking:
        lines.append(f"| {row['variant']} | {row['mean_validation_macro_f1']:.4f} | {row['mean_validation_ce']:.4f} | {row['eligible']} |")
    baseline = next(r for r in ranking if r["variant"] == "original_saved")
    delta = chosen["mean_validation_macro_f1"] - baseline["mean_validation_macro_f1"]
    lines.extend(["", f"按预设的跨策略/seed 平均验证 macro-F1 规则，候选为 **{chosen['variant']}**，相对原模型平均差 **{delta:+.4f}**。",
                  "这只是开发阶段的候选选择，不是独立检验，也不是强制替换原模型。若原模型排第一，代码会如实保留它。", "",
                  "## 2. 各策略的类别取舍", "", "| Strategy | Variant | Validation macro-F1 ± SD | Class 4 F1 | Class 5 F1 | Validation CE |",
                  "|---|---|---:|---:|---:|---:|"])
    for r in summary:
        lines.append(f"| {r['strategy']} | {r['variant']} | {r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f} | {r['rare_f1_mean']:.4f} | {r['class5_f1_mean']:.4f} | {r['cross_entropy_mean']:.4f} |")
    lines.extend(["", "## 3. 同 seed 对原 checkpoint 的变化", "", "| Strategy | Selected head ΔF1 | Δclass4 F1 | Δclass5 F1 | Seeds improved in macro-F1 |", "|---|---:|---:|---:|---:|"])
    for strategy in protocol["strategies"]:
        group = [r for r in pairs if r["strategy"] == strategy and r["variant"] == chosen["variant"]]
        if group:
            values = [float(np.mean([r[k] for r in group])) for k in ("validation_macro_f1_difference", "class4_f1_difference", "class5_f1_difference")]
            wins = sum(r["validation_macro_f1_difference"] > 0 for r in group)
            lines.append(f"| {strategy} | {values[0]:+.4f} | {values[1]:+.4f} | {values[2]:+.4f} | {wins}/{len(group)} |")
        else:
            lines.append(f"| {strategy} | +0.0000 | +0.0000 | +0.0000 | 0/{len(protocol['seeds'])} (original retained) |")
    lines.extend(["", "## 4. 早停如何执行", "",
                  f"新 MLP 的最大 epoch={protocol['settings']['max_epochs']}，patience={protocol['settings']['patience']}；每 epoch 用 eval 模式计算训练/验证 loss 和 F1。",
                  "验证 F1 连续 patience 个 epoch 没有累计超过 0.001 的改善，或验证 CE 连续 patience 个 epoch 没有超过 0.0001 的下降，就停止。",
                  "停止后恢复所有已观察 epoch 中验证 F1 最高的 checkpoint，F1 相同时优先 CE 更低者；不会使用最后一轮替代最佳模型。",
                  "CE 保护可能在 F1 尚有上升空间时提前停止，这是待检验的取舍。早停和正则化不保证泛化改善。",
                  "original_saved 保留原早停结果；mlp_early 只改变早停规则，其余 MLP 在同一新规则下分别改变容量、dropout 或 weight decay。", "",
                  "## 5. 数据和结论边界", "",
                  "- 每种方法在自己的原标注子集上训练；同一策略/seed 的所有候选严格共享同一组标签。标准化只 fit 这组标签对应的特征。",
                  "- 此次线性分类器也只使用本次预算内的标签，可以与同预算 MLP 比较；每个 C 单独报告，没有偷偷为每个策略挑不同的 C。",
                  "- 验证集保持原有完整样本，尤其保留稀有类别支持数。没有重划 split、额外平衡类别或查看未查询训练标签来选样。",
                  "- 本实验使用旧 MLP 产生的查询轨迹，是固定数据上的分类器消融；还不是新分类器参与每轮选样的完整主动学习实验。",
                  "- 类别 F1 取舍、概率 CE 和训练/验证差距应一起看；训练分数降低或差距变小本身不代表更好。",
                  "- seed SD 不是置信区间；同一个验证集被重复使用，网格排名有选择偏差。之前已经查看的测试集没有重新变成独立确认集。",
                  "- 不收敛的线性候选保留诊断，但不参与共享候选选择。MLP 的 eligible 表示训练及 checkpoint 有效，不表示数学意义上的优化收敛。", "",
                  "## 6. 建议回传的文件", "", "1. `ablation_report.md`", "2. `metrics_mean_std.csv`", "3. `paired_vs_original.csv`",
                  "4. `figures/validation_model_comparison.png`", "5. `figures/class4_class5_comparison.png`",
                  "6. `figures/training_curves_hybrid_seed_42.png`（或可用 seed）", ""])
    (out / "ablation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--run", help="Completed original research run")
    parser.add_argument("--labels", type=int, help="A saved original budget; default original final budget")
    parser.add_argument("--prepared-dir", help="Moved prepared directory; checksum must still match")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.patience, args.max_epochs, args.threads) < 1 or (args.labels is not None and args.labels < 2):
        raise ValueError("Positive epochs/patience/threads and labels>=2 required.")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    import numpy as np
    import torch
    import sklearn
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    root = args.project_dir.resolve()
    run, locked = io.resolve_run(root, args.run)
    x, ids, y, roles, train, val, cache, prepared = io.load_inputs(root, run, locked, args.prepared_dir)
    config = locked["config"]
    if set(config["strategies"]) != {"random", "uncertainty", "diversity", "hybrid"}:
        raise ValueError("This comparison requires the four original strategies.")
    budget = args.labels if args.labels is not None else config["max_labels"]
    items = select_records(run, locked, train, ids, y, budget)
    settings = {"hidden_dim": config["head_hidden_dim"], "dropout": config["head_dropout"],
                "weight_decay": config["head_weight_decay"], "learning_rate": config["head_learning_rate"],
                "batch_size": config["head_batch_size"], "max_epochs": args.max_epochs,
                "patience": args.patience, "f1_delta": .001, "loss_delta": .0001}
    code_dir = Path(__file__).resolve().parent
    protocol = {"source_run": str(run), "locked_sha256": io.sha256(run / "locked_protocol.json"),
                "feature_key": locked["feature_key"], "split_sha256": locked["split_sha256"],
                "labels": budget, "validation_size": len(val), "test_evaluated": False,
                "strategies": config["strategies"], "seeds": config["seeds"], "settings": settings,
                "mlp_variants": MLP_VARIANTS, "linear_variants": LINEAR_VARIANTS,
                "cpu_threads": args.threads,
                "versions": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__},
                "code": {name: io.sha256(code_dir / name) for name in ("run_head_ablation.py", "galaxy_ablation_support.py")},
                "source_subsets": [{"seed": item["seed"], "strategy": item["strategy"],
                   "labeled_ids_sha256": item["ids_sha256"], "record_sha256": io.sha256(item["record_path"]),
                   "checkpoint_sha256": item["record"]["checkpoint_sha256"]} for item in items]}
    identity = digest_value(protocol)
    out = root / "outputs/head_ablation" / f"fixed_{budget}_{identity[:16]}"
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = out / "locked_ablation_protocol.json"
    if protocol_path.exists() and io.read_json(protocol_path) != protocol:
        raise ValueError("Ablation protocol changed in an existing output folder.")
    atomic_json(protocol_path, protocol)
    print(f"Original run: {run}\nAblation output: {out}\nBudget per strategy/seed: {budget}; full validation: {len(val)}", flush=True)
    print("No CNN extraction, no fresh acquisition, no test evaluation.", flush=True)
    results = {}
    with threadpool_limits(limits=args.threads):
        for item in items:
            pos = item["positions"]
            for variant in VARIANTS:
                folder = out / f"seed_{item['seed']}" / item["strategy"] / variant
                result = fit_or_resume(folder, item, variant, x[pos], y[pos], x[val], y[val],
                                       ids[pos], ids[val], settings, identity)
                results[(item["strategy"], item["seed"], variant)] = result
    export_results(out, results, config["strategies"], config["seeds"], budget, protocol)
    atomic_json(out / "RUN_COMPLETE.json", {"complete": True, "test_evaluated": False,
                                           "protocol_sha256": identity, "fit_records": len(results)})
    atomic_json(root / "outputs/latest_head_ablation.json", {"ablation_directory": str(out), "source_run": str(run)})
    print(f"\nDONE: {out / 'ablation_report.md'}", flush=True)
    print("Use validation results to assess improvement; smaller train/val gap alone is insufficient.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(f"\nABLATION STOPPED: {exc}", file=sys.stderr)
        if not isinstance(exc, KeyboardInterrupt):
            traceback.print_exc()
        print("Original experiment is unchanged. Rerun the same command to reuse committed fits.", file=sys.stderr)
        sys.exit(1)
