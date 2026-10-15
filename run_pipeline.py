"""Single entry point: inspect protocol, extract features, run AL, then evaluate."""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "al_config.json")
    parser.add_argument("--smoke", action="store_true", help="Small integration test, separate from research results.")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    threads = int(config["cpu_threads"])
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TORCH_HOME"] = str(root / "cache/torch")
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    from galaxy_al.common import write_json
    from galaxy_al.dataset import load_prepared
    from galaxy_al.experiment import run_experiments
    from galaxy_al.features import extract_features
    from galaxy_al.results import plot_results
    if threads < 1:
        raise ValueError("cpu_threads must be positive.")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if config["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda. CPU is the tested default.")
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; set device to cpu.")
    device = torch.device(config["device"])
    if args.smoke:
        config.update({"seeds": [42], "initial_labels": 20, "query_batch_size": 20,
                       "max_labels": 60, "head_max_epochs": 3, "head_patience": 2,
                       "full_label_reference": False})
    integers = ("feature_batch_size", "image_size", "initial_labels", "query_batch_size", "max_labels",
                "hybrid_candidate_multiplier", "head_hidden_dim", "head_batch_size", "head_max_epochs", "head_patience")
    if any(not isinstance(config[k], int) or config[k] < 1 for k in integers):
        raise ValueError("Batch sizes, budgets, dimensions, epochs and patience must be positive integers.")
    if config["max_labels"] < config["initial_labels"]:
        raise ValueError("max_labels must be at least initial_labels.")
    expected = {"random", "uncertainty", "diversity", "hybrid"}
    if set(config["strategies"]) != expected or len(config["strategies"]) != 4:
        raise ValueError("This RQ protocol requires exactly random, uncertainty, diversity and hybrid.")
    if len(set(config["seeds"])) != len(config["seeds"]) or not config["seeds"]:
        raise ValueError("Use a nonempty list of unique seeds.")
    if not 0 <= config["rare_class"] <= 9:
        raise ValueError("rare_class must be between 0 and 9.")
    data = load_prepared(root, config, smoke=args.smoke)
    if config["max_labels"] > len(data.positions("train_pool")):
        raise ValueError("max_labels exceeds the training pool.")
    print(f"Mode: {'SMOKE TEST' if args.smoke else 'RESEARCH'} | device={device} | threads={threads}", flush=True)
    print(f"Fixed preparation: {data.directory.name}", flush=True)
    for name in ("train_pool", "validation", "test"):
        print(f"  {name}: {len(data.positions(name)):,}", flush=True)
    with threadpool_limits(limits=threads):
        features, cache, feature_key = extract_features(root, data, config, device, args.smoke)
        if not np.isfinite(features).all():
            raise RuntimeError("Nonfinite cached features.")
        directory, rows = run_experiments(root, features, feature_key, data, config, device, args.smoke)
        import h5py, matplotlib, torchvision
        write_json(directory / "environment.json", {
            "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
            "torchvision": torchvision.__version__, "h5py": h5py.__version__,
            "matplotlib": matplotlib.__version__, "feature_cache": str(cache),
            "device": str(device), "cpu_threads": threads,
        })
        plot_results(directory, rows, data, config, args.smoke)


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(f"\nSTOPPED: {exc}", file=sys.stderr)
        print("Completed feature batches and committed training rounds are retained. Fix the issue and rerun the same command.", file=sys.stderr)
        if not isinstance(exc, KeyboardInterrupt):
            traceback.print_exc()
        sys.exit(1)
