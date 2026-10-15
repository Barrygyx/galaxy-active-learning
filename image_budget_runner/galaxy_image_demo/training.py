"""Budget-controlled CNN training with validation-F1 early stopping and resume."""
import copy
import hashlib
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .common import classification_metrics, fingerprint, read_json, sha256_file, write_json
from .models import augment_d4, configure_trainable, training_mode


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def save_torch(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def metrics(y, probabilities):
    result = classification_metrics(y, probabilities)
    result["cross_entropy"] = float(-np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-15, 1)).mean())
    return result


def predict(model, inputs, indices, batch_size, device, features=False):
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            x = torch.from_numpy(np.array(inputs[indices[start:start + batch_size]], dtype=np.float32)).to(device)
            logits = model.classify_features(x) if features else model(x)
            chunks.append(logits.softmax(1).cpu().numpy())
    p = np.concatenate(chunks)
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite model predictions.")
    return p


def initial_features(model, images, folder, architecture, weight_digest, batch_size, device):
    """Fixed initial encoder outputs, no label fitting; reused by related variants."""
    identity = {"architecture": architecture, "weights_sha256": weight_digest,
                "images_sha256": read_json(folder / "manifest.json")["images_sha256"],
                "device": str(device), "torch": torch.__version__,
                "training_code": sha256_file(Path(__file__))}
    key = fingerprint(identity)[:16]
    path, manifest_path = folder / f"initial_features_{key}.npy", folder / f"initial_features_{key}.json"
    if path.exists() and manifest_path.exists():
        record = read_json(manifest_path)
        if record["identity"] != identity or record["sha256"] != sha256_file(path):
            raise ValueError("Initial encoder feature cache changed.")
        features = np.load(path, mmap_mode="r", allow_pickle=False)
        if not np.isfinite(features).all() or features.shape[0] != len(images):
            raise ValueError("Initial feature cache invalid.")
        return features
    model.eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            x = torch.from_numpy(np.array(images[start:start + batch_size], dtype=np.float32)).to(device)
            parts.append(model.encode(x).cpu().numpy())
            if start == 0 or (start // batch_size) % 20 == 0:
                print(f"  initial {architecture} features {min(start+batch_size,len(images))}/{len(images)}", flush=True)
    result = np.concatenate(parts)
    if not np.isfinite(result).all():
        raise ValueError("Initial encoder produced nonfinite features.")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, result, allow_pickle=False)
    os.replace(temporary, path)
    write_json(manifest_path, {"identity": identity, "sha256": sha256_file(path)})
    return np.load(path, mmap_mode="r", allow_pickle=False)


def class_weights(labels):
    counts = np.bincount(labels, minlength=10)
    weights = np.zeros(10, dtype=np.float32)
    present = counts > 0
    weights[present] = np.sqrt(1.0 / counts[present])
    weights[present] /= weights[present].mean()
    return weights


def fit_model(model, images, initial, subset, y, settings, directory, identity, device,
              interrupt_after_epoch=None):
    """One seed, one image model. Test positions never enter this function."""
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    if result_path.exists():
        result = read_json(result_path)
        if result["identity"] != identity:
            raise ValueError("Saved model identity mismatch.")
        for name, expected in result["artifact_sha256"].items():
            if sha256_file(directory / name) != expected:
                raise ValueError(f"Completed model artifact changed: {directory / name}")
        print(f"Resume completed model: {identity['model_name']}", flush=True)
        return result
    train, monitor, validation, train_monitor = (subset[k] for k in ("train", "monitor", "validation", "train_monitor"))
    labels = y[train]
    frozen = model.specification["mode"] == "frozen"
    pretrained = model.specification["mode"] not in ("scratch",)
    use_features = frozen
    inputs = initial if use_features else images
    # All pretrained CNNs of a given architecture start with identical normalization
    # fitted ONLY to their shared labeled training subset's initial features.
    if pretrained:
        mean = np.asarray(initial[train], dtype=np.float64).mean(0).astype(np.float32)
        std = np.asarray(initial[train], dtype=np.float64).std(0).astype(np.float32)
        std[std < 1e-6] = 1
        model.feature_mean.copy_(torch.from_numpy(mean))
        model.feature_std.copy_(torch.from_numpy(std))
    model.to(device)
    configure_trainable(model, warmup=False)
    backbone_parameters = [p for p in model.backbone.parameters() if p.requires_grad]
    groups = [{"params": model.head.parameters(), "lr": settings["head_lr"]}]
    if backbone_parameters:
        groups.append({"params": backbone_parameters,
                       "lr": settings["scratch_lr"] if not pretrained else settings["backbone_lr"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=settings["weight_decay"])
    weight = class_weights(labels) if model.specification.get("balanced") else None
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight, device=device) if weight is not None else None)
    shuffle_rng = torch.Generator().manual_seed(settings["seed"] + 101)
    augment_rng = torch.Generator().manual_seed(settings["seed"] + 202)
    torch.manual_seed(settings["seed"] + 303)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(settings["seed"] + 303)
    history, best_state, best_epoch, best_f1, best_ce = [], None, 0, -1.0, float("inf")
    anchor, stale, elapsed, epoch_done = -1.0, 0, 0.0, 0
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    progress_path = directory / "progress.pt"
    if progress_path.exists():
        state = torch.load(progress_path, map_location="cpu", weights_only=True)
        if state["identity"] != identity:
            raise ValueError("Interrupted model protocol changed.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, best_state, best_epoch = state["history"], state["best_state"], state["best_epoch"]
        best_f1, best_ce, anchor, stale = state["best_f1"], state["best_ce"], state["anchor"], state["stale"]
        elapsed, epoch_done = state["elapsed"], state["epoch"]
        torch.set_rng_state(state["torch_rng"])
        shuffle_rng.set_state(state["shuffle_rng"])
        augment_rng.set_state(state["augment_rng"])
        if device.type == "cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"Resume {identity['model_name']} after epoch {epoch_done}", flush=True)
    stop_reason = "max_epochs"
    # The last committed epoch may already satisfy the stop rule; do not add an
    # extra epoch after an interruption between progress commit and finalization.
    early_already = epoch_done >= settings["min_epochs"] and stale >= settings["patience"]
    if early_already:
        stop_reason = "validation_macro_f1_patience"
    else:
        for epoch in range(epoch_done + 1, settings["epochs"] + 1):
            started = time.monotonic()
            warmup = pretrained and not frozen and epoch <= settings["warmup_epochs"]
            configure_trainable(model, warmup=warmup)
            training_mode(model, warmup=warmup)
            order = torch.randperm(len(train), generator=shuffle_rng).numpy()
            online_loss, batches = 0.0, 0
            for offset in range(0, len(train), settings["batch_size"]):
                selected = train[order[offset:offset + settings["batch_size"]]]
                x = torch.from_numpy(np.array(inputs[selected], dtype=np.float32))
                if model.specification["augment"] and not use_features:
                    x = augment_d4(x, augment_rng)
                x = x.to(device)
                target = torch.from_numpy(y[selected].astype(np.int64)).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model.classify_features(x) if use_features else model(x)
                loss = criterion(logits, target)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss.")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                online_loss += loss.item() * len(selected)
                batches += 1
                if batches == 1 or batches % 20 == 0:
                    print(f"  {identity['model_name']} epoch {epoch}: trained {min(offset+len(selected),len(train))}/{len(train)}", flush=True)
            p_val = predict(model, inputs, monitor, settings["batch_size"], device, use_features)
            p_train = predict(model, inputs, train_monitor, settings["batch_size"], device, use_features)
            mv, mt = metrics(y[monitor], p_val), metrics(y[train_monitor], p_train)
            f1, ce = mv["macro_f1"], mv["cross_entropy"]
            if f1 > best_f1 + 1e-12 or (abs(f1 - best_f1) <= 1e-12 and ce < best_ce):
                best_state, best_epoch, best_f1, best_ce = cpu_state(model), epoch, f1, ce
            if f1 > anchor + settings["min_delta"]:
                anchor, stale = f1, 0
            else:
                stale += 1
            # Guarantee the tuning phase has a chance after head warm-up.
            if warmup:
                stale = 0
            seconds = time.monotonic() - started
            elapsed += seconds
            history.append({"epoch": epoch, "stage": "head_warmup" if warmup else "training",
                            "online_loss": online_loss / len(train), "train_monitor_macro_f1": mt["macro_f1"],
                            "train_monitor_ce": mt["cross_entropy"], "monitor_macro_f1": f1,
                            "monitor_accuracy": mv["accuracy"], "monitor_ce": ce,
                            "monitor_class4_f1": mv["per_class"][4]["f1"],
                            "monitor_class5_f1": mv["per_class"][5]["f1"], "epoch_seconds": seconds})
            save_torch(progress_path, {"identity": identity, "epoch": epoch, "model": cpu_state(model),
                "optimizer": optimizer.state_dict(), "history": history, "best_state": best_state,
                "best_epoch": best_epoch, "best_f1": best_f1, "best_ce": best_ce, "anchor": anchor, "stale": stale,
                "elapsed": elapsed, "torch_rng": torch.get_rng_state(), "shuffle_rng": shuffle_rng.get_state(),
                "augment_rng": augment_rng.get_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []})
            print(f"  epoch {epoch:02d}: monitor F1={f1:.4f}, acc={mv['accuracy']:.4f}, CE={ce:.4f}, {seconds:.1f}s", flush=True)
            if interrupt_after_epoch == epoch:
                raise RuntimeError("Simulated interruption after committed epoch.")
            if epoch >= settings["min_epochs"] and stale >= settings["patience"]:
                stop_reason = "validation_macro_f1_patience"
                break
    if best_state is None:
        raise ValueError("No model checkpoint was trained.")
    model.load_state_dict(best_state)
    p_train = predict(model, inputs, train, settings["batch_size"], device, use_features)
    p_validation = predict(model, inputs, validation, settings["batch_size"], device, use_features)
    p_monitor = predict(model, inputs, monitor, settings["batch_size"], device, use_features)
    measured = {"train": metrics(y[train], p_train), "validation": metrics(y[validation], p_validation),
                "monitor": metrics(y[monitor], p_monitor)}
    save_torch(directory / "best_model.pt", {"model_name": identity["model_name"], "state_dict": best_state,
               "best_epoch": best_epoch, "identity": identity, "settings": settings})
    write_json(directory / "history.json", history)
    temporary = directory / "predictions.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, train_cache_rows=train, validation_cache_rows=validation,
                            train_probabilities=p_train, validation_probabilities=p_validation)
    os.replace(temporary, directory / "predictions.npz")
    result = {"identity": identity, **measured, "best_epoch": best_epoch, "epochs_run": len(history),
              "stop_reason": stop_reason, "training_seconds": elapsed,
              "trainable_parameters_after_warmup": trainable_count,
              "total_parameters": sum(p.numel() for p in model.parameters()),
              "class_weights": weight.tolist() if weight is not None else None,
              "artifact_sha256": {name: sha256_file(directory / name) for name in
                                  ("best_model.pt", "history.json", "predictions.npz")}}
    write_json(result_path, result)
    progress_path.unlink(missing_ok=True)
    print(f"Finished {identity['model_name']}: FULL validation F1={measured['validation']['macro_f1']:.4f}, accuracy={measured['validation']['accuracy']:.4f}", flush=True)
    return result
