"""Equal-probability ensemble of already-trained pilot CNNs; no label-budget change."""
import argparse
from pathlib import Path
import numpy as np
from galaxy_image_demo.common import read_json, read_csv, write_json, write_csv, sha256_file, fingerprint
from galaxy_image_demo.training import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--members", nargs="+")
    args = parser.parse_args()
    root = args.project_dir.resolve()
    directory = args.experiment or Path(read_json(root / "outputs/latest_image_demo.json")["experiment_directory"])
    if not directory.is_absolute():
        directory = root / directory
    protocol = read_json(directory / "locked_protocol.json")
    members = args.members or ["resnet18_partial_aug", "mobilenet_small_aug"]
    if args.members is None and (directory / "models/efficientnet_b0_aug/result.json").exists():
        members.append("efficientnet_b0_aug")
    if len(set(members)) != len(members) or len(members) < 2:
        raise ValueError("Choose at least two distinct completed models.")
    source_ids = np.load(directory / "cache_source_indices.npy", allow_pickle=False)
    known = {}
    for row in read_csv(directory / "subsets.csv"):
        source, label = int(row["source_index"]), int(row["label"])
        if source in known and known[source] != label:
            raise ValueError("Inconsistent subset labels")
        known[source] = label
    accumulated, reference_rows, member_hashes = {}, {}, {}
    for name in members:
        folder = directory / "models" / name
        result = read_json(folder / "result.json")
        if result["identity"]["model_name"] != name or result["identity"]["pilot_protocol_sha256"] != fingerprint(protocol):
            raise ValueError("Ensemble member belongs to a different experiment protocol")
        file = folder / "predictions.npz"
        digest = sha256_file(file)
        if digest != result["artifact_sha256"]["predictions.npz"]:
            raise ValueError(f"Changed prediction file: {name}")
        member_hashes[name] = digest
        with np.load(file, allow_pickle=False) as saved:
            for split in ("train", "validation"):
                rows = saved[split + "_cache_rows"]
                probability = saved[split + "_probabilities"]
                if rows.ndim != 1 or (rows < 0).any() or (rows >= len(source_ids)).any():
                    raise ValueError("Invalid prediction row mapping")
                if probability.shape != (len(rows), 10) or not np.isfinite(probability).all() or (probability < 0).any() or not np.allclose(probability.sum(1), 1, atol=1e-5):
                    raise ValueError("Invalid ensemble probabilities")
                if split in reference_rows and not np.array_equal(rows, reference_rows[split]):
                    raise ValueError("Members were evaluated on different image rows")
                reference_rows[split] = rows.copy()
                accumulated[split] = accumulated.get(split, np.zeros_like(probability, dtype=np.float64)) + probability
    measured = {}
    for split in accumulated:
        accumulated[split] /= len(members)
        y = np.asarray([known[int(source_ids[p])] for p in reference_rows[split]], dtype=np.int64)
        measured[split] = metrics(y, accumulated[split])
    name = "equal_ensemble_" + str(len(members))
    out = directory / "ensembles" / ("__".join(members))
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, m in measured.items():
        rows.append({"model": name, "split": split, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                     "cross_entropy": m["cross_entropy"], "class4_f1": m["per_class"][4]["f1"],
                     "class5_f1": m["per_class"][5]["f1"]})
    write_csv(out / "ensemble_metrics.csv", rows)
    write_json(out / "ensemble_result.json", {"members": members, "weights": [1/len(members)]*len(members),
               "member_prediction_sha256": member_hashes, "test_evaluated": False, "smoke": protocol["smoke"],
               "training_labels": protocol["train_size"], "selection": "fixed equal weights; no ensemble weight tuning",
               "exploratory": True, **measured})
    result = measured["validation"]
    lines = ["# 已训练模型的等权集成", "", "**工程 smoke 测试，无科研含义。**" if protocol["smoke"] else "**真实数据上的探索性开发结果，单 seed；不是独立确认。**", "",
             "成员：" + ", ".join(members), "", "对成员的十类概率取等权平均，没有搜索集成权重；没有训练新模型。",
             f"共同训练标签数：{protocol['train_size']}；共同完整验证图数：{protocol['full_validation_size']}。",
             f"Accuracy：{result['accuracy']:.4f}；macro-F1：{result['macro_f1']:.4f}；class 4 F1：{result['per_class'][4]['f1']:.4f}；class 5 F1：{result['per_class'][5]['f1']:.4f}。", "",
             "成员是同一批图像训练的不同模型，不是独立重复实验。集成增加训练/推理成本，不增加独特训练标签数。",
             "成员在开发阶段选择，完整验证集已经参与候选比较，因此此处是探索性比较；测试集未评估。", ""]
    (out / "ensemble_report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(root / "outputs/latest_image_ensemble.json", {"ensemble_directory": str(out), "experiment_directory": str(directory)})
    print(f"Ensemble complete: accuracy={result['accuracy']:.4f}, macro-F1={result['macro_f1']:.4f}\n{out / 'ensemble_report.md'}", flush=True)


if __name__ == "__main__":
    main()
