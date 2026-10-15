"""Small Galaxy10 image-model pilot. Train real CNNs, preserve the original split."""
import argparse
import hashlib
import os
import sys
import traceback
from pathlib import Path


def main():
    root_default = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=root_default)
    parser.add_argument("--config", type=Path, help="Defaults to PROJECT/al_config.json")
    parser.add_argument("--suite", choices=("quick", "all"), default="quick")
    parser.add_argument("--models", nargs="+", help="Explicit catalog names instead of suite")
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--monitor-size", type=int, default=600)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-epochs", type=int, default=6)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", action="store_true", help="Tiny engineering run, separate from pilot scores")
    args = parser.parse_args()
    root = args.project_dir.resolve()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TORCH_HOME"] = str(root / "cache/torch")
    import numpy as np
    import torch
    import torchvision
    import h5py
    from threadpoolctl import threadpool_limits
    from galaxy_image_demo.common import read_json, write_json, write_csv, fingerprint, sha256_file
    from galaxy_image_demo.data import load_prepared, select_subsets, cache_images
    from galaxy_image_demo.models import CATALOG, QUICK, build_model
    from galaxy_image_demo.training import initial_features, fit_model
    from galaxy_image_demo.reporting import report
    if min(args.threads, args.batch_size, args.epochs, args.patience, args.min_epochs) < 1:
        raise ValueError("Threads, batches, epochs and patience must be positive.")
    if args.image_size < 64 or args.monitor_size < 10 or args.warmup_epochs < 0 or args.seed < 0:
        raise ValueError("Image size >=64, monitor >=10, warmup >=0, seed >=0 required.")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable. AMD Windows users should use CPU.")
    device = torch.device(device_name)
    config_path = args.config or root / "al_config.json"
    if not config_path.is_absolute():
        config_path = root / config_path
    config = read_json(config_path)
    data = load_prepared(root, config, smoke=False)
    if args.smoke:
        args.train_size = min(40, len(data.positions("train_pool")))
        args.monitor_size = min(20, len(data.positions("validation")))
        args.image_size, args.batch_size = 64, 8
        args.epochs, args.patience, args.min_epochs, args.warmup_epochs = 2, 1, 2, 0
    if args.min_epochs > args.epochs or args.warmup_epochs >= args.epochs:
        raise ValueError("min-epochs must be <= epochs, warmup-epochs must be < epochs.")
    names = args.models or (list(CATALOG) if args.suite == "all" else QUICK)
    if not names or len(set(names)) != len(names) or any(n not in CATALOG for n in names):
        raise ValueError(f"Choose unique model names from: {list(CATALOG)}")
    selected, monitor, validation, train_monitor = select_subsets(data, args.train_size, args.monitor_size, args.seed)
    settings = {"seed": args.seed, "epochs": args.epochs, "patience": args.patience,
                "min_epochs": args.min_epochs, "warmup_epochs": args.warmup_epochs,
                "batch_size": args.batch_size, "min_delta": .0001,
                "head_lr": .001, "backbone_lr": .0001, "scratch_lr": .001, "weight_decay": .0001}
    print(f"Mode={'SMOKE' if args.smoke else 'IMAGE PILOT'}; device={device}; image size={args.image_size}; models={names}", flush=True)
    print(f"Shared random labeled subset={len(selected)}; early-stop monitor={len(monitor)}; full validation={len(validation)}", flush=True)
    print("No test evaluation. Scores are a pilot, not a reproduction of a full-data published benchmark.", flush=True)
    with threadpool_limits(limits=args.threads):
        images, positions, image_cache, manifest = cache_images(root, data, np.concatenate([selected, validation]), args.image_size, torch)
        mapping = {int(p): i for i, p in enumerate(positions)}
        subset = {name: np.asarray([mapping[int(p)] for p in values], dtype=np.int64) for name, values in
                  (("train", selected), ("monitor", monitor), ("validation", validation), ("train_monitor", train_monitor))}
        y = data.labels[positions]
        package = Path(__file__).resolve().parent
        code_paths = [Path(__file__), *sorted((package / "galaxy_image_demo").glob("*.py"))]
        protocol = {"split_sha256": data.split_sha256, "source_file": str(data.source), "prepared_dir": str(data.directory),
                    "image_cache": str(image_cache), "images_sha256": manifest["images_sha256"],
                    "train_source_ids_sha256": hashlib.sha256(data.ids[selected].tobytes()).hexdigest(),
                    "monitor_source_ids_sha256": hashlib.sha256(data.ids[monitor].tobytes()).hexdigest(),
                    "image_size": args.image_size, "train_size": len(selected), "monitor_size": len(monitor),
                    "full_validation_size": len(validation), "settings": settings, "device": str(device),
                    "threads": args.threads, "smoke": args.smoke, "test_evaluated": False,
                    "catalog": CATALOG, "augmentation": "independent quarter-turns, horizontal/vertical flips; train only",
                    "pretrained_batchnorm": "running stats frozen; selected affine parameters trainable",
                    "versions": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
                                 "torchvision": torchvision.__version__, "h5py": h5py.__version__},
                    "code_sha256": {str(p.relative_to(package)): sha256_file(p) for p in code_paths}}
        base_key = fingerprint(protocol)[:16]
        directory = root / "outputs/image_demo" / (("smoke_" if args.smoke else "pilot_") + base_key)
        directory.mkdir(parents=True, exist_ok=True)
        protocol_path = directory / "locked_protocol.json"
        if protocol_path.exists() and read_json(protocol_path) != protocol:
            raise ValueError("Pilot protocol mismatch.")
        write_json(protocol_path, protocol)
        np.save(directory / "cache_source_indices.npy", data.ids[positions], allow_pickle=False)
        write_csv(directory / "subsets.csv", [{"role": role, "source_index": int(data.ids[positions[row]]), "label": int(y[row])}
                  for role, rows in subset.items() for row in rows])
        print(f"Pilot directory: {directory}", flush=True)
        results = {}
        for name in names:
            print(f"\n===== {name} =====", flush=True)
            model, weights_digest = build_model(name, args.seed)
            spec = CATALOG[name]
            initial = None
            if spec["mode"] != "scratch":
                model.to(device)
                initial = initial_features(model, images, image_cache, spec["arch"], weights_digest,
                                           args.batch_size, device)
                model.cpu()
            identity = {"model_name": name, "specification": spec, "weights_sha256": weights_digest,
                        "pilot_protocol_sha256": fingerprint(protocol)}
            result = fit_model(model, images, initial, subset, y, settings, directory / "models" / name, identity, device)
            results[name] = result
            del model, initial
            if device.type == "cuda":
                torch.cuda.empty_cache()
        suite_name = "custom_" + fingerprint(names)[:8] if args.models else args.suite
        report_dir = report(directory, results, subset, y, data, protocol, args.smoke, suite_name)
        write_json(root / "outputs/latest_image_demo.json", {"experiment_directory": str(directory),
                   "report_directory": str(report_dir), "smoke": args.smoke, "models": names})
        print(f"\nDONE: {report_dir / 'image_demo_report.md'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(f"\nIMAGE DEMO STOPPED: {exc}", file=sys.stderr)
        if not isinstance(exc, KeyboardInterrupt):
            traceback.print_exc()
        print("Original experiment remains unchanged. Rerun the same command to resume completed epochs/models.", file=sys.stderr)
        sys.exit(1)
