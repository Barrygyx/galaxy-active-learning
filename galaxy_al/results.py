"""Research figures and descriptive paired-seed results; no invented significance."""
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import read_csv, write_csv, write_json
from .metrics import normalized_aulc


COLORS = {"random": "#667085", "uncertainty": "#dd8a29", "diversity": "#327aa8", "hybrid": "#19846b"}


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def plot_results(directory, rows, data, config, smoke=False):
    directory = Path(directory)
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["strategy"], row["labels"])].append(row)
    summary = []
    for (split, strategy, budget), values in groups.items():
        item = {"split": split, "strategy": strategy, "labels": budget, "n_seeds": len(values)}
        for metric in ("accuracy", "macro_f1", "balanced_accuracy", "rare_recall", "rare_precision", "rare_f1"):
            item[metric + "_mean"], item[metric + "_std"] = mean_std([v[metric] for v in values])
        summary.append(item)
    summary.sort(key=lambda r: (r["split"], r["strategy"], r["labels"]))
    write_csv(directory / "metrics_mean_std.csv", summary)
    for split in ("validation", "test"):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        for ax, metric, title in zip(axes, ("macro_f1", "rare_recall", "rare_f1"),
                                     ("Overall: macro-F1", "Rare class: recall", "Rare class: F1")):
            for strategy in config["strategies"]:
                values = sorted([r for r in summary if r["split"] == split and r["strategy"] == strategy],
                                key=lambda r: r["labels"])
                x = np.array([r["labels"] for r in values])
                means = np.array([r[metric + "_mean"] for r in values])
                std = np.array([r[metric + "_std"] for r in values])
                ax.plot(x, means, marker="o", markersize=4, label=strategy, color=COLORS[strategy])
                ax.fill_between(x, np.maximum(means - std, 0), np.minimum(means + std, 1),
                                alpha=0.10, color=COLORS[strategy])
            reference = [r for r in summary if r["split"] == split and r["strategy"] == "full_reference"]
            if reference:
                ax.axhline(reference[0][metric + "_mean"], color="#7b498f", ls="--", lw=1.2,
                           label=f"Full-label reference (n={reference[0]['labels']:,})")
            ax.set(title=title, xlabel="Queried training labels", ylabel=metric.replace("_", " "), ylim=(0, 1))
            ax.grid(alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
        mode = "SMOKE TEST — not research results" if smoke else "Galaxy10 DECaLS active learning"
        fig.suptitle(f"{mode} | {split} | mean +/- 1 sample SD across seeds", fontsize=12)
        fig.tight_layout(rect=[0, .13, 1, .94])
        fig.savefig(figures / f"{split}_learning_curves.png", dpi=180)
        fig.savefig(figures / f"{split}_learning_curves.pdf")
        plt.close(fig)

    coverage = read_csv(directory / "labeled_class_counts.csv")
    rare = [r for r in coverage if int(r["class_id"]) == config["rare_class"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy in config["strategies"]:
        grouped = defaultdict(list)
        for row in rare:
            if row["strategy"] == strategy:
                grouped[int(row["labels"])].append(int(row["queried_class_count"]))
        x = sorted(grouped)
        means = np.array([mean_std(grouped[k])[0] for k in x])
        std = np.array([mean_std(grouped[k])[1] for k in x])
        ax.plot(x, means, marker="o", color=COLORS[strategy], label=strategy)
        ax.fill_between(x, np.maximum(means - std, 0), means + std, color=COLORS[strategy], alpha=.1)
    ax.set(xlabel="Queried training labels", ylabel="Revealed rare-class labels",
           title=f"Rare-class coverage (class {config['rare_class']}) — mean +/- 1 SD")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "rare_class_label_coverage.png", dpi=180)
    plt.close(fig)

    # One curve per seed; compare equal budgets rather than pooling runs.
    curves = defaultdict(list)
    for row in rows:
        if row["strategy"] != "full_reference":
            curves[(row["split"], row["seed"], row["strategy"])].append(row)
    auc_rows, paired = [], []
    for (split, seed, strategy), values in curves.items():
        values.sort(key=lambda r: r["labels"])
        auc_rows.append({"split": split, "seed": seed, "strategy": strategy,
                         "normalized_macro_f1_aulc": normalized_aulc(
                             [v["labels"] for v in values], [v["macro_f1"] for v in values])})
    auc_lookup = {(r["split"], r["seed"], r["strategy"]): r["normalized_macro_f1_aulc"] for r in auc_rows}
    if "random" in config["strategies"]:
        for split in ("validation", "test"):
            for seed in config["seeds"]:
                base = curves[(split, seed, "random")][-1]
                for strategy in config["strategies"]:
                    if strategy == "random":
                        continue
                    end = curves[(split, seed, strategy)][-1]
                    paired.append({
                        "split": split, "seed": seed, "strategy": strategy, "baseline": "random",
                        "final_macro_f1_difference": end["macro_f1"] - base["macro_f1"],
                        "final_rare_recall_difference": end["rare_recall"] - base["rare_recall"],
                        "normalized_aulc_difference": auc_lookup[(split, seed, strategy)] - auc_lookup[(split, seed, "random")],
                    })
    write_csv(directory / "aulc_by_seed.csv", auc_rows)
    write_csv(directory / "paired_differences_vs_random.csv", paired,
              ["split", "seed", "strategy", "baseline", "final_macro_f1_difference",
               "final_rare_recall_difference", "normalized_aulc_difference"])

    savings = []
    for seed in config["seeds"]:
        for target in config["validation_f1_targets"]:
            def first_crossing(strategy):
                hits = [r["labels"] for r in curves[("validation", seed, strategy)] if r["macro_f1"] >= target]
                return min(hits) if hits else None
            random_budget = first_crossing("random") if "random" in config["strategies"] else None
            for strategy in config["strategies"]:
                budget = first_crossing(strategy)
                saving = ((random_budget - budget) / random_budget * 100
                          if budget is not None and random_budget is not None else None)
                savings.append({"seed": seed, "validation_macro_f1_target": target, "strategy": strategy,
                                "first_observed_budget": budget, "random_first_observed_budget": random_budget,
                                "label_saving_percent_vs_random": saving,
                                "status": "both_reached" if saving is not None else "target_not_reached_in_one_or_both"})
    write_csv(directory / "validation_label_savings.csv", savings)

    final = [r for r in summary if r["split"] == "test" and r["strategy"] != "full_reference"
             and r["labels"] == config["max_labels"]]
    lines = ["# Galaxy10 active-learning experiment", "",
             "SMOKE TEST ONLY — do not use these numbers in the research report." if smoke else
             "Exploratory paired-seed comparison on the cleaned Galaxy10 DECaLS subset.", "",
             "RQ: Can combining uncertainty and diversity reduce labeling effort while preserving rare-class recognition?", "",
             "## Fixed protocol", "",
             f"- Seeds: {config['seeds']}; same uniform initial sample within each seed across strategies.",
             f"- Annotation budgets: {config['initial_labels']} to {config['max_labels']}, increments of {config['query_batch_size']}.",
             f"- Fixed split fingerprint: `{data.split_sha256}`.",
             "- Frozen ImageNet-pretrained ResNet18, full-frame resize, 512-dimensional cached features.",
             "- Fresh ten-class MLP per round; feature standardization fitted only on currently labeled training samples.",
             "- Hybrid: entropy shortlist, then farthest-first diversity relative to all labeled samples.",
             "- Validation selects checkpoints. Test evaluation starts only after all acquisition trajectories finish.",
             f"- Fixed validation labels: {len(data.positions('validation'))}; fixed test labels: {len(data.positions('test'))}.",
             "  These label costs are additional to the reported queried-training-label budget.", "",
             "## Final-budget held-out test results", "",
             "| Strategy | Macro-F1 (mean ± SD) | Rare recall (mean ± SD) | Rare F1 (mean ± SD) |",
             "|---|---:|---:|---:|"]
    for row in final:
        lines.append(f"| {row['strategy']} | {row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} | "
                     f"{row['rare_recall_mean']:.4f} ± {row['rare_recall_std']:.4f} | "
                     f"{row['rare_f1_mean']:.4f} ± {row['rare_f1_std']:.4f} |")
    lines += ["", "## Interpretation", "",
              "Compare the full learning curves and normalized AULC, not only final accuracy. Inspect rare recall AND precision/F1.",
              "Validation label savings use the first observed budget crossing each predeclared threshold; no interpolation or extrapolation.",
              "Unreached targets remain blank, not zero savings. Positive savings favor the strategy; negative values favor random.",
              "Paired seed differences and sample SD are descriptive; three seeds do not establish statistical significance.",
              "A full-label reference is a separate, larger-label-budget comparison, not a guaranteed performance ceiling.", "",
              "## Scope and limitations", "",
              "- Results concern the curated subset excluding conflicting linked labels, not every raw image.",
              "- Exact hashing/coordinates do not rule out all near-duplicates or related neighboring objects.",
              "- External ImageNet labels are a shared pretrained resource, not part of the Galaxy10 query budget.",
              "- Frozen ImageNet features may not capture all astronomical morphology. CNN fine-tuning is a later robustness extension.",
              "- Small minority-class evaluation sets make recall variable. Model-seed SD does not measure all test-sampling uncertainty.",
              "- Use validation for any later tuning. Changing methods after inspecting test outcomes requires a fresh evaluation design.",
              "- A hybrid method that fails to beat random is still a valid outcome; do not claim an improvement without evidence.", ""]
    (directory / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(directory / "RUN_COMPLETE.json", {"smoke": smoke, "seeds": config["seeds"],
                                                "strategies": config["strategies"], "max_labels": config["max_labels"]})
    print(f"\nCOMPLETE. Report: {directory / 'experiment_report.md'}", flush=True)
    print(f"Figures: {figures}", flush=True)
