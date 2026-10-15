"""Hash-checked, resumable features. Frozen eval-mode extraction uses no labels."""
from pathlib import Path
import hashlib
import time
import h5py
import numpy as np
import torch
import torchvision
from .common import code_hashes, fingerprint, read_json, sha256_file, write_json
from .model import build_encoder, preprocess


def save_preprocessing_preview(root, data, size, key):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = root / "outputs" / f"preprocessing_preview_{key}.png"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    positions = np.random.default_rng(42).permutation(data.positions("train_pool"))[:4]
    fig, axes = plt.subplots(len(positions), 2, figsize=(7, 3 * len(positions)), squeeze=False)
    with h5py.File(data.source, "r") as handle:
        for row, pos in enumerate(positions):
            original = np.asarray(handle["images"][int(data.ids[pos])])
            tensor = preprocess(original[None], size, torch.device("cpu"))[0]
            mean = torch.tensor([.485, .456, .406])[:, None, None]
            std = torch.tensor([.229, .224, .225])[:, None, None]
            display = (tensor * std + mean).permute(1, 2, 0).clamp(0, 1).numpy()
            axes[row, 0].imshow(original)
            axes[row, 1].imshow(display)
            axes[row, 0].set_title(f"Original | training source index {data.ids[pos]}", fontsize=9)
            axes[row, 1].set_title(f"Full frame resized to {size} x {size}", fontsize=9)
            for ax in axes[row]:
                ax.axis("off")
    fig.suptitle("Normalization inverted for display; no random augmentation", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(path, dpi=140)
    plt.close(fig)


def extract_features(root, data, config, device, smoke=False):
    signature = {
        "split_sha256": data.split_sha256,
        "source_size": data.source.stat().st_size,
        "source_mtime_ns": data.source.stat().st_mtime_ns,
        "ids_sha256": hashlib.sha256(data.ids.tobytes()).hexdigest(),
        "weights": "ResNet18_Weights.IMAGENET1K_V1",
        "transform": {"size": config["image_size"], "full_frame": True,
                      "interpolation": "bilinear_antialias", "scale": "uint8/255",
                      "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "code": code_hashes(["features.py", "model.py"]),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "batch_size": config["feature_batch_size"], "device": str(device), "smoke": smoke,
    }
    key = fingerprint(signature)[:16]
    cache = root / "cache/features" / key
    cache.mkdir(parents=True, exist_ok=True)
    path, done_path = cache / "features.npy", cache / "done.npy"
    manifest = cache / "feature_manifest.json"
    n = len(data.ids)
    if path.exists() and done_path.exists():
        features = np.lib.format.open_memmap(path, mode="r+")
        done = np.lib.format.open_memmap(done_path, mode="r+")
        if features.shape != (n, 512) or done.shape != (n,):
            raise ValueError("Incomplete/corrupted feature cache shape; remove this cache directory and retry.")
        invalid = np.flatnonzero(done & ~np.isfinite(features).all(axis=1))
        done[invalid] = False
        done.flush()
    else:
        features = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, 512))
        features[:] = np.nan
        features.flush()
        done = np.lib.format.open_memmap(done_path, mode="w+", dtype=bool, shape=(n,))
        done[:] = False
        done.flush()
    if done.all() and manifest.exists():
        saved = read_json(manifest)
        if saved.get("signature") != signature or sha256_file(path) != saved.get("features_sha256"):
            raise ValueError("Feature cache fingerprint/hash mismatch; inspect or remove this cache directory.")
        print(f"Reusing verified features: {n:,} x 512", flush=True)
        save_preprocessing_preview(root, data, config["image_size"], key)
        return np.load(path, mmap_mode="r", allow_pickle=False), cache, key + "_" + saved["features_sha256"][:16]

    encoder = build_encoder(root, device)
    pending = np.flatnonzero(~done)
    started, processed = time.monotonic(), 0
    print(f"Extracting frozen ResNet18 features: {len(pending):,} remaining / {n:,} total", flush=True)
    with h5py.File(data.source, "r") as f, torch.inference_mode():
        for start in range(0, len(pending), config["feature_batch_size"]):
            positions = pending[start:start + config["feature_batch_size"]]
            ids = data.ids[positions]
            # Fancy indexing avoids reading huge gaps when resuming sparse unfinished rows.
            images = np.asarray(f["images"][ids])
            for local, position in enumerate(positions):
                digest = hashlib.sha256(np.ascontiguousarray(images[local]).tobytes()).hexdigest()
                if digest != data.image_hashes[int(position)]:
                    raise ValueError(f"Raw image hash changed at original index {data.ids[position]}.")
            values = encoder(preprocess(images, config["image_size"], device)).cpu().numpy()
            if values.shape != (len(positions), 512) or not np.isfinite(values).all():
                raise RuntimeError("Encoder produced invalid features.")
            features[positions] = values
            features.flush()       # Commit values BEFORE marking rows complete.
            done[positions] = True
            done.flush()
            processed += len(positions)
            elapsed = time.monotonic() - started
            rate = processed / max(elapsed, 1e-9)
            remaining = len(pending) - processed
            print(f"  features {int(done.sum()):,}/{n:,} | {rate:.1f} images/s | ETA {remaining/rate/60:.1f} min", flush=True)
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not done.all():
        raise RuntimeError("Feature extraction is incomplete.")
    feature_digest = sha256_file(path)
    write_json(manifest, {"signature": signature, "features_sha256": feature_digest,
                          "shape": [n, 512], "dtype": "float32"})
    np.save(cache / "source_indices.npy", data.ids, allow_pickle=False)
    save_preprocessing_preview(root, data, config["image_size"], key)
    return np.load(path, mmap_mode="r", allow_pickle=False), cache, key + "_" + feature_digest[:16]
