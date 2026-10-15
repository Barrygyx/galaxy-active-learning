"""Real image CNNs: scratch, frozen probes, partial and full fine-tuning."""
import hashlib
import re
from pathlib import Path
from urllib.parse import urlparse
import torch
from torch import nn
from torchvision import models


CATALOG = {
    "tiny_cnn_aug": {"arch": "tiny", "mode": "scratch", "augment": True},
    "resnet18_frozen": {"arch": "resnet18", "mode": "frozen", "augment": False},
    "resnet18_partial_aug": {"arch": "resnet18", "mode": "partial", "augment": True},
    "resnet18_full_aug": {"arch": "resnet18", "mode": "full", "augment": True},
    "mobilenet_small_aug": {"arch": "mobilenet", "mode": "full", "augment": True},
    "resnet18_scratch_aug": {"arch": "resnet18", "mode": "scratch", "augment": True},
    "resnet18_partial_noaug": {"arch": "resnet18", "mode": "partial", "augment": False},
    "efficientnet_b0_aug": {"arch": "efficientnet", "mode": "full", "augment": True},
    "resnet18_partial_balanced_aug": {"arch": "resnet18", "mode": "partial", "augment": True, "balanced": True},
}
QUICK = list(CATALOG)[:5]


def weights_for(arch):
    return {"resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
            "mobilenet": models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
            "efficientnet": models.EfficientNet_B0_Weights.IMAGENET1K_V1}[arch]


def weight_file_hash(arch, required=False):
    from .common import sha256_file
    weight = weights_for(arch)
    path = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weight.url).path).name
    if not path.exists():
        if required:
            raise ValueError(f"Expected downloaded weights missing: {path}")
        return None
    digest = sha256_file(path)
    match = re.search(r"-([0-9a-f]{8,})\.", path.name)
    if match and not digest.startswith(match[1]):
        raise ValueError(f"Pretrained weight checksum mismatch: {path}. Inspect this cache file.")
    return digest


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        layers, before = [], 3
        for after in (16, 32, 64):
            layers.extend([nn.Conv2d(before, after, 3, padding=1), nn.GroupNorm(4, after), nn.ReLU(), nn.MaxPool2d(2)])
            before = after
        layers.extend([nn.Conv2d(64, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(),
                       nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten()])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class GalaxyCNN(nn.Module):
    def __init__(self, backbone, dimension, specification):
        super().__init__()
        self.backbone = backbone
        self.specification = specification
        self.head = nn.Sequential(nn.Linear(dimension, 128), nn.ReLU(), nn.Dropout(.1), nn.Linear(128, 10))
        self.register_buffer("feature_mean", torch.zeros(dimension))
        self.register_buffer("feature_std", torch.ones(dimension))
        self.register_buffer("image_mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def encode(self, images):
        return self.backbone((images - self.image_mean) / self.image_std)

    def classify_features(self, features):
        return self.head((features - self.feature_mean) / self.feature_std)

    def forward(self, images):
        return self.classify_features(self.encode(images))


def build_model(name, seed=42, download=True):
    """download=False only reconstructs architecture for loading a saved full state."""
    torch.manual_seed(seed)
    spec = dict(CATALOG[name])
    arch, mode = spec["arch"], spec["mode"]
    pretrained = arch != "tiny" and mode != "scratch"
    digest = None
    if pretrained and download:
        weight_file_hash(arch)  # Validate existing bytes BEFORE torch loads them.
    weights = weights_for(arch) if pretrained and download else None
    try:
        if arch == "tiny":
            backbone, dimension = TinyBackbone(), 512
        elif arch == "resnet18":
            backbone, dimension = models.resnet18(weights=weights), 512
            backbone.fc = nn.Identity()
        elif arch == "mobilenet":
            backbone, dimension = models.mobilenet_v3_small(weights=weights), 576
            backbone.classifier = nn.Identity()
        else:
            backbone, dimension = models.efficientnet_b0(weights=weights), 1280
            backbone.classifier = nn.Identity()
    except Exception as exc:
        raise RuntimeError(f"Could not build {name}. Pretrained variants require official weights; there is no random fallback. {exc}") from exc
    if pretrained and download:
        digest = weight_file_hash(arch, required=True)
    model = GalaxyCNN(backbone, dimension, spec)
    configure_trainable(model, warmup=False)
    return model, digest


def configure_trainable(model, warmup=False):
    mode = model.specification["mode"]
    for name, parameter in model.backbone.named_parameters():
        allowed = mode in ("scratch", "full") or (mode == "partial" and name.startswith(("layer3.", "layer4.")))
        parameter.requires_grad_(allowed and not warmup)
    model.head.requires_grad_(True)


def training_mode(model, warmup=False):
    model.train()
    if warmup or model.specification["mode"] == "frozen":
        model.backbone.eval()
    elif model.specification["mode"] != "scratch":
        # Small-label fine-tuning: keep external BN running statistics fixed.
        # Affine parameters still learn when their block is selected for tuning.
        for layer in model.backbone.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                layer.eval()


def augment_d4(images, generator):
    """Independent exact quarter-turns and flips. No crop or interpolation loss."""
    result = images.clone()
    k = torch.randint(4, (len(images),), generator=generator)
    horizontal = torch.rand(len(images), generator=generator) < .5
    vertical = torch.rand(len(images), generator=generator) < .5
    for i in range(len(images)):
        sample = torch.rot90(images[i], int(k[i]), dims=(-2, -1))
        if horizontal[i]:
            sample = sample.flip(-1)
        if vertical[i]:
            sample = sample.flip(-2)
        result[i] = sample
    return result
