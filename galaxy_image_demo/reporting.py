"""Report image-pipeline pilot results without using the test split."""
from pathlib import Path
import numpy as np
from .common import CLASS_NAMES, read_json, write_csv, write_json


def report(directory, results, subset, y, data, protocol, smoke, suite):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = directory / "reports" / suite
    folder.mkdir(parents=True, exist_ok=True)
    figures = folder / "figures"
    figures.mkdir(exist_ok=True)
    rows, class_rows, full = [], [], []
    for name, result in results.items():
        for split in ("train", "monitor", "validation"):
            m = result[split]
            row = {"model": name, "split": split, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                   "cross_entropy": m["cross_entropy"], "class4_f1": m["per_class"][4]["f1"],
                   "class4_recall": m["per_class"][4]["recall"], "class5_f1": m["per_class"][5]["f1"],
                   "best_epoch": result["best_epoch"], "epochs_run": result["epochs_run"],
                   "training_seconds": result["training_seconds"], "stop_reason": result["stop_reason"]}
            rows.append(row)
            if split == "validation":
                full.append(row)
            class_rows.extend({"model": name, "split": split, "class_name": CLASS_NAMES[c["class_id"]], **c} for c in m["per_class"])
    write_csv(folder / "metrics.csv", rows)
    write_csv(folder / "per_class_metrics.csv", class_rows)
    ranked = sorted(full, key=lambda r: (-r["macro_f1"], r["cross_entropy"]))
    write_csv(folder / "model_comparison.csv", ranked)
    names = list(results)
    index = np.arange(len(names))
    full_map = {r["model"]: r for r in full}

    def save(fig, filename):
        fig.tight_layout()
        fig.savefig(figures / (filename + ".png"), dpi=150)
        fig.savefig(figures / (filename + ".pdf"))
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, key in zip(axes, ("accuracy", "macro_f1")):
        ax.bar(index, [full_map[n][key] for n in names], color=[f"C{i%10}" for i in index])
        ax.set_xticks(index, names, rotation=35, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Full validation " + key)
    fig.suptitle("One-seed IMAGE MODEL PILOT; same train subset; test not evaluated")
    save(fig, "image_model_comparison")
    fig, ax = plt.subplots(figsize=(13, 6))
    for j, key in enumerate(("class4_f1", "class5_f1")):
        ax.bar(index + (j-.5)*.35, [full_map[n][key] for n in names], .35, label=key)
    ax.set_xticks(index, names, rotation=35, ha="right")
    ax.set_ylabel("Full validation class F1")
    ax.set_ylim(0, 1)
    ax.legend()
    save(fig, "image_class4_class5")
    fig, axes = plt.subplots(len(names), 2, figsize=(12, 3*len(names)), squeeze=False)
    for i, name in enumerate(names):
        result = results[name]
        h = read_json(directory / "models" / name / "history.json")
        epochs = [r["epoch"] for r in h]
        axes[i, 0].plot(epochs, [r["train_monitor_ce"] for r in h], label="Train monitor CE (eval)")
        axes[i, 0].plot(epochs, [r["monitor_ce"] for r in h], label="Validation monitor CE")
        axes[i, 1].plot(epochs, [r["monitor_macro_f1"] for r in h], label="Validation monitor macro-F1")
        for ax in axes[i]:
            ax.axvline(result["best_epoch"], color="gray", linestyle="--", label="Selected epoch")
            ax.set_title(name)
            ax.set_xlabel("Epoch")
            ax.legend(fontsize=8)
    save(fig, "image_training_curves")
    best = ranked[0]["model"]
    cm = np.asarray(results[best]["validation"]["confusion_matrix"])
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{norm[i,j]*100:.0f}", ha="center", va="center", color="white" if norm[i,j]>.5 else "black", fontsize=8)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10), [f"{i}: {n}" for i, n in enumerate(CLASS_NAMES)])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(best + " | full validation row percentages")
    fig.colorbar(im, ax=ax)
    save(fig, "best_validation_confusion")
    train_counts = np.bincount(y[subset["train"]], minlength=10)
    val_counts = np.bincount(y[subset["validation"]], minlength=10)
    monitor_counts = np.bincount(y[subset["monitor"]], minlength=10)
    lines = ["# Galaxy10 图像模型 Demo", "", "**这是工程 smoke 测试，分数没有科研含义。**" if smoke else "**这是单 seed、小预算开发实验；不是完整主动学习结果。**", "",
             f"Train labels={len(subset['train'])}；early-stop validation monitor={len(subset['monitor'])}；最终完整 validation={len(subset['validation'])}；image size={protocol['image_size']}。", "",
             "所有方法共享相同训练图和验证划分。训练子集均匀随机抽取，没有用真实类别平衡初始标注。测试集未评估。", "",
             "## 完整验证集结果", "", "| Model | Accuracy | Macro-F1 | Class 4 F1 | Class 5 F1 | CE | Best epoch / trained | Training min |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in ranked:
        lines.append(f"| {r['model']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['class4_f1']:.4f} | {r['class5_f1']:.4f} | {r['cross_entropy']:.4f} | {r['best_epoch']} / {r['epochs_run']} | {r['training_seconds']/60:.1f} |")
    lines.extend(["", f"本次完整验证集 macro-F1 排名第一：**{best}**。单 seed 排名可能随样本和初始化变化，不代表稳定优势。", "",
                  "## 类别支持数", "", "| Class | Train | Monitor | Full validation |", "|---|---:|---:|---:|"])
    for i in range(10):
        lines.append(f"| {i}: {CLASS_NAMES[i]} | {train_counts[i]} | {monitor_counts[i]} | {val_counts[i]} |")
    lines.extend(["", "## 训练规则", "",
                  "- 真正训练卷积参数：scratch 从零开始；partial 更新 ResNet layer3/layer4；full 更新整个 backbone。frozen 只训练分类头，作为对照。",
                  "- 所有方法用同一个 128 维隐藏层分类头；预训练模型用当前已标注训练图的初始特征 mean/std 固定归一化，scratch 使用恒等特征归一化。",
                  "- 微调有 head warm-up，随后使用较小的 backbone 学习率；预训练 BN 的 running stats 固定，选定层的 affine 参数可学习。",
                  "- 增强只在训练时应用独立的 90°倍数旋转、水平/垂直翻转；没有裁剪或颜色扰动。",
                  "- balanced 变体只根据当前训练标签计算平方根逆频率权重；不增加标签预算。",
                  "- 早停只监控验证 monitor 的 macro-F1，保留观察到的最佳 checkpoint；CE 仅记录，没有上一轮的 CE 强制停止条件。",
                  "- 为保证微调阶段有学习机会，warm-up 后有最小训练长度。小预算 demo 到达最大 epoch 并不证明模型充分收敛。", "",
                  "## 怎样与旧结果和文献比较", "",
                  "本次默认 1,000 个训练标签、128 像素与单 seed，训练样本也不一定与原 Random 查询轨迹相同，不能直接与旧 2,000-label 分数判断谁优谁劣。应先比较本表内的共同协议结果。",
                  "文献的 73.9% 是其他图像构造和划分、更多训练数据下的 accuracy，不是本 demo 的目标 macro-F1。此处同时报告两个指标以免混淆。",
                  "monitor 是完整 validation 的子集，完整 validation 的评分并非完全独立确认：其中部分图像已参与早停，整体又用于候选比较。",
                  "如果模型有明显潜力，之后应提高分辨率/训练预算、增加 seeds，再在原 train_pool 内重新执行主动学习。不得用全标签微调后的 encoder 冒充小标签预算。", "",
                  "## 回传文件", "", "- `image_demo_report.md`", "- `model_comparison.csv`", "- `figures/image_model_comparison.png`",
                  "- `figures/image_training_curves.png`", "- `figures/image_class4_class5.png`", ""])
    (folder / "image_demo_report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(folder / "RUN_COMPLETE.json", {"smoke": smoke, "test_evaluated": False,
                 "models": list(results), "train_size": len(subset["train"]), "best_on_validation": best})
    return folder
