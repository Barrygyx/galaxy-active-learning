"""Frozen external pretrained features and a fresh MLP at every annotation budget."""
import copy
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from .metrics import classification_metrics


def build_encoder(root, device):
    os.environ["TORCH_HOME"] = str(root / "cache/torch")
    try:
        network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    except Exception as exc:
        raise RuntimeError(
            "Cannot load pretrained ResNet18 weights. Check the internet connection and rerun; "
            "existing completed work will be reused. There is no random-weight fallback. "
            f"Original error: {exc}"
        ) from exc
    network.fc = nn.Identity()
    network.requires_grad_(False)
    return network.to(device).eval()


def preprocess(images, size, device):
    # Full-frame resize preserves the field of view; unlike the default weights
    # transform, this intentionally uses no center crop. Same rule for ALL splits.
    x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2)
    x = x.to(device=device, dtype=torch.float32).div_(255.0)
    x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear",
                                       align_corners=False, antialias=True)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (x - mean) / std


class Head(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                    nn.Dropout(dropout), nn.Linear(hidden_dim, 10))

    def forward(self, features):
        return self.layers(features)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def train_head(train_x, train_y, validation_x, validation_y, config, seed, device):
    """Only revealed training labels enter this function. No pool/test labels."""
    set_seed(seed)
    mean = np.asarray(train_x, dtype=np.float64).mean(axis=0).astype(np.float32)
    std = np.asarray(train_x, dtype=np.float64).std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    train = torch.tensor((train_x - mean) / std, dtype=torch.float32, device=device)
    targets = torch.tensor(train_y, dtype=torch.long, device=device)
    validation = torch.tensor((validation_x - mean) / std, dtype=torch.float32, device=device)
    network = Head(train.shape[1], config["head_hidden_dim"], config["head_dropout"]).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config["head_learning_rate"],
                                 weight_decay=config["head_weight_decay"])
    criterion = nn.CrossEntropyLoss()
    best_score, best_loss, best_state, best_epoch, stale = -1.0, float("inf"), None, 0, 0
    history = []
    for epoch in range(1, config["head_max_epochs"] + 1):
        network.train()
        order = torch.randperm(len(train), device=device)
        train_loss = 0.0
        for start in range(0, len(order), config["head_batch_size"]):
            batch = order[start:start + config["head_batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(network(train[batch]), targets[batch])
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)
        network.eval()
        with torch.inference_mode():
            logits = network(validation)
            probability = logits.softmax(dim=1).cpu().numpy()
            val_loss = nn.functional.cross_entropy(
                logits, torch.tensor(validation_y, dtype=torch.long, device=device)).item()
        score = classification_metrics(validation_y, probability)["macro_f1"]
        history.append({"epoch": epoch, "training_loss": train_loss / len(train),
                        "validation_loss": val_loss, "validation_macro_f1": score})
        improved = score > best_score + 1e-6 or (abs(score - best_score) <= 1e-6 and val_loss < best_loss - 1e-6)
        if improved:
            best_score, best_loss, best_epoch = score, val_loss, epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in network.state_dict().items()})
            stale = 0
        else:
            stale += 1
        if stale >= config["head_patience"]:
            break
    checkpoint = {
        "state_dict": best_state, "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
        "input_dim": int(train.shape[1]), "hidden_dim": int(config["head_hidden_dim"]),
        "dropout": float(config["head_dropout"]), "best_epoch": best_epoch, "training_seed": seed,
    }
    return checkpoint, history


def predict(checkpoint, features, device, batch_size=4096):
    network = Head(checkpoint["input_dim"], checkpoint["hidden_dim"], checkpoint["dropout"])
    network.load_state_dict(checkpoint["state_dict"])
    network.to(device).eval()
    mean, std = checkpoint["mean"].numpy(), checkpoint["std"].numpy()
    result = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            normalized = (features[start:start + batch_size] - mean) / std
            x = torch.tensor(normalized, dtype=torch.float32, device=device)
            result.append(network(x).softmax(dim=1).cpu().numpy())
    return np.concatenate(result) if result else np.empty((0, 10), dtype=np.float32)


def save_checkpoint(path, checkpoint):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, tmp)
    os.replace(tmp, path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)
