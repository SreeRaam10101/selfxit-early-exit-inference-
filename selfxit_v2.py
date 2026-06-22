#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SelfXit v2 — Extended Early-Exit ResNet

Improvements over v1:
  1.  Configurable exit placement      --exits 2 3 4
  2.  Conv block in exit heads         --exit_conv
  3.  Joint end-to-end training        --joint_training / --exit_loss_weight
  4.  Curriculum temperature annealing --T_start / --T_end
  5.  Entropy + confidence static gate --tau_entropy
  6.  Budget-aware inference           --compute_budget (0–1 FLOPs fraction)
  7.  MAC / FLOPs profiling            printed alongside latency
  8.  Per-class exit analysis          printed after evaluation
  9.  Pareto frontier sweep            --sweep
  10. Checkpoint save / resume         --checkpoint_dir / --resume
  11. TensorBoard logging              optional (auto-detected)
  12. TinyImageNet-200 support         --dataset tinyimagenet --data_root <path>

Run example (CIFAR-100, joint training, sweep):
  python selfxit_v2.py \\
      --dataset cifar100 --model resnet18 \\
      --epochs_backbone 5 --joint_training \\
      --epochs_exits 15 --epochs_gates 5 \\
      --T_start 4.0 --T_end 1.0 \\
      --policy both --sweep \\
      --checkpoint_dir ./ckpts
"""

import argparse
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import torchvision
import torchvision.transforms as T

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except Exception:
    _TB_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import numpy as np
    _PLOT_AVAILABLE = True
except Exception:
    _PLOT_AVAILABLE = False

try:
    from selfxit_kernels import gate_features_triton
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False

_USE_TRITON_GATE = False  # set True via --triton_gate at runtime


# ---------------------------------------------------------------------------
#  Logger
# ---------------------------------------------------------------------------

class Logger:
    """Wraps TensorBoard SummaryWriter with a print fallback."""

    def __init__(self, log_dir: Optional[str] = None):
        self.writer = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            if _TB_AVAILABLE:
                self.writer = SummaryWriter(log_dir=log_dir)
                print(f"[TensorBoard] Writing to {log_dir}")
            else:
                print("[Logger] tensorboard not installed — using stdout only.")

    def scalar(self, tag: str, value: float, step: int):
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def close(self):
        if self.writer:
            self.writer.close()


# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _sync(device: torch.device) -> None:
    """Block until all queued work on this device has completed.

    CUDA/MPS kernels are launched asynchronously — timing code without this
    measures enqueue time, not compute time.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def entropy_from_probs(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Shannon entropy H = -sum(p log p)."""
    return -(probs * (probs + eps).log()).sum(dim=1)


# ---------------------------------------------------------------------------
#  MAC profiling
# ---------------------------------------------------------------------------

@dataclass
class MACProfile:
    stem_macs: int
    layer_macs: Dict[int, int]   # layer index → MACs
    fc_macs: int
    head_macs: Dict[int, int]    # exit index → MACs

    def _backbone_up_to(self, layer: int) -> int:
        total = self.stem_macs
        for l in range(1, layer + 1):
            total += self.layer_macs.get(l, 0)
        if layer >= 4:
            total += self.fc_macs
        return total

    def total_backbone_macs(self) -> int:
        return self._backbone_up_to(4)

    def macs_for_exit(self, exit_idx: int, layer_idx: int) -> int:
        """Backbone MACs up to layer_idx + exit head MACs."""
        return self._backbone_up_to(layer_idx) + self.head_macs.get(exit_idx, 0)

    def flops_fraction(self, exit_idx: int, layer_idx: int) -> float:
        denom = max(self.total_backbone_macs(), 1)
        return self.macs_for_exit(exit_idx, layer_idx) / denom


def _hook_count_macs(module: nn.Module, x_in: torch.Tensor) -> Tuple[int, torch.Tensor]:
    """Run module on a single-sample input and count MACs via hooks."""
    macs = [0]
    hooks = []

    def conv_hook(m, inp, out):
        c_out, c_in_g = m.weight.shape[:2]
        k2 = m.weight.shape[2] * m.weight.shape[3]
        h, w = out.shape[2], out.shape[3]
        macs[0] += c_out * c_in_g * k2 * h * w

    def linear_hook(m, inp, out):
        macs[0] += m.in_features * m.out_features

    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    # Ensure single sample
    x = x_in[:1] if x_in.size(0) > 1 else x_in
    with torch.no_grad():
        out = module(x)

    for h in hooks:
        h.remove()
    return macs[0], out


class _Stem(nn.Module):
    """Wraps conv1 + bn1 + relu + maxpool as a single profiler-friendly module."""
    def __init__(self, bb: nn.Module):
        super().__init__()
        self.conv1 = bb.conv1
        self.bn1 = bb.bn1
        self.relu = bb.relu
        self.maxpool = bb.maxpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool(self.relu(self.bn1(self.conv1(x))))


def profile_model_macs(model: "EarlyExitResNet",
                       input_shape: Tuple[int, ...],
                       device: torch.device) -> MACProfile:
    """Profile MACs per backbone segment and per exit head."""
    model.to(device)
    model.eval()
    bb = model.backbone
    x0 = torch.zeros(1, *input_shape, device=device)

    stem_macs, x_stem = _hook_count_macs(_Stem(bb), x0)
    l1_macs, x_l1 = _hook_count_macs(bb.layer1, x_stem)
    l2_macs, x_l2 = _hook_count_macs(bb.layer2, x_l1)
    l3_macs, x_l3 = _hook_count_macs(bb.layer3, x_l2)
    l4_macs, x_l4 = _hook_count_macs(bb.layer4, x_l3)

    x_pool = bb.avgpool(x_l4)
    x_flat = torch.flatten(x_pool, 1)
    fc_macs, _ = _hook_count_macs(bb.fc, x_flat)

    layer_feats = {1: x_l1, 2: x_l2, 3: x_l3, 4: x_l4}
    head_macs: Dict[int, int] = {}
    for i, (layer_idx, head) in enumerate(zip(model.exit_layers, model.exit_heads)):
        m, _ = _hook_count_macs(head, layer_feats[layer_idx])
        head_macs[i] = m

    return MACProfile(
        stem_macs=stem_macs,
        layer_macs={1: l1_macs, 2: l2_macs, 3: l3_macs, 4: l4_macs},
        fc_macs=fc_macs,
        head_macs=head_macs,
    )


# ---------------------------------------------------------------------------
#  Datasets
# ---------------------------------------------------------------------------

def _split_train_val(full_train_aug: torch.utils.data.Dataset,
                     full_train_noaug: torch.utils.data.Dataset,
                     val_frac: float,
                     seed: int = 42) -> Tuple[Subset, Subset]:
    """
    Split a training dataset into (train, val) by a fixed-seed random
    permutation of indices.

    full_train_aug and full_train_noaug must be two Dataset instances built
    over identical underlying data in identical order (e.g. the same
    CIFAR10(train=True, ...) call with only the transform differing), so
    index i refers to the same sample in both. The train subset is drawn
    from full_train_aug (train-time augmentation); the val subset is drawn
    from full_train_noaug (test-time transform) so calibration/threshold
    tuning never sees randomly cropped/flipped pixels.
    """
    n_total = len(full_train_aug)
    n_val = int(n_total * val_frac)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=gen).tolist()
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return Subset(full_train_aug, train_idx), Subset(full_train_noaug, val_idx)


def _cifar_loaders(dataset: str, batch_size: int,
                   num_workers: int, val_frac: float
                   ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    assert dataset in ("cifar10", "cifar100")
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" \
        else torchvision.datasets.CIFAR100
    full_train_aug   = cls(root="./data", train=True,  download=True, transform=train_tf)
    full_train_noaug = cls(root="./data", train=True,  download=True, transform=test_tf)
    testset           = cls(root="./data", train=False, download=True, transform=test_tf)

    trainset, valset = _split_train_val(full_train_aug, full_train_noaug, val_frac)

    pin = torch.cuda.is_available()
    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin)
    return (DataLoader(trainset, shuffle=True,  **kwargs),
            DataLoader(valset,   shuffle=False, **kwargs),
            DataLoader(testset,  shuffle=False, **kwargs))


def _setup_tinyimagenet_val(data_root: str) -> str:
    """
    Reorganise TinyImageNet val/images/ into ImageFolder format.
    Creates <data_root>/val_organized/<class>/<img> if not already present.

    Verifies the existing directory's file count against the annotations
    file before trusting it — a prior run interrupted mid-copy would
    otherwise leave a partial val_organized/ that looks "done" (the
    directory exists) but is missing files, with no error anywhere.
    """
    organized = os.path.join(data_root, "val_organized")
    ann_path = os.path.join(data_root, "val", "val_annotations.txt")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(
            f"TinyImageNet val annotations not found: {ann_path}\n"
            "Download TinyImageNet and point --data_root to the extracted folder."
        )
    with open(ann_path) as f:
        annotations = [line.strip().split("\t") for line in f]

    if os.path.isdir(organized):
        existing_count = sum(len(files) for _, _, files in os.walk(organized))
        if existing_count == len(annotations):
            return organized
        print(f"[Dataset] {organized} exists but has {existing_count} files, "
              f"expected {len(annotations)} — rebuilding.")
        shutil.rmtree(organized)

    os.makedirs(organized, exist_ok=True)
    for img_name, class_id, *_rest in annotations:
        src = os.path.join(data_root, "val", "images", img_name)
        dst_dir = os.path.join(organized, class_id)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(dst_dir, img_name))
    print(f"[Dataset] TinyImageNet val organised → {organized}")
    return organized


def _tinyimagenet_loaders(data_root: str, batch_size: int,
                          num_workers: int, val_frac: float
                          ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    mean = (0.4802, 0.4481, 0.3975)
    std  = (0.2770, 0.2691, 0.2821)

    train_tf = T.Compose([
        T.RandomCrop(64, padding=8),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train_dir = os.path.join(data_root, "train")
    val_dir = _setup_tinyimagenet_val(data_root)

    full_train_aug   = torchvision.datasets.ImageFolder(train_dir, transform=train_tf)
    full_train_noaug = torchvision.datasets.ImageFolder(train_dir, transform=test_tf)
    testset           = torchvision.datasets.ImageFolder(val_dir,   transform=test_tf)

    trainset, valset = _split_train_val(full_train_aug, full_train_noaug, val_frac)

    pin = torch.cuda.is_available()
    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin)
    return (DataLoader(trainset, shuffle=True,  **kwargs),
            DataLoader(valset,   shuffle=False, **kwargs),
            DataLoader(testset,  shuffle=False, **kwargs))


def _channels_last_loader(loader: DataLoader) -> DataLoader:
    """Wrap a DataLoader so every image batch is emitted in channels_last format."""
    class _CLLoader:
        def __init__(self, l): self._l = l
        def __iter__(self):
            for imgs, lbls in self._l:
                yield imgs.to(memory_format=torch.channels_last), lbls
        def __len__(self): return len(self._l)
        def __getattr__(self, k): return getattr(self._l, k)
    return _CLLoader(loader)


def get_loaders(args) -> Tuple[DataLoader, DataLoader, DataLoader, int, Tuple[int, ...]]:
    """Returns (trainloader, valloader, testloader, num_classes, input_shape)."""
    if args.dataset in ("cifar10", "cifar100"):
        num_classes = 10 if args.dataset == "cifar10" else 100
        train, val, test = _cifar_loaders(args.dataset, args.batch_size,
                                          args.num_workers, args.val_frac)
        return train, val, test, num_classes, (3, 32, 32)
    else:
        train, val, test = _tinyimagenet_loaders(args.data_root, args.batch_size,
                                                  args.num_workers, args.val_frac)
        return train, val, test, 200, (3, 64, 64)


# ---------------------------------------------------------------------------
#  Model components
# ---------------------------------------------------------------------------

class SpatialAttentionPool(nn.Module):
    """
    Learned spatial pooling — drop-in replacement for AdaptiveAvgPool2d((1,1)).

    A 1×1 conv scores every spatial location, softmax over the H·W positions
    turns the scores into attention weights that sum to 1, and the feature map
    is collapsed by a weighted sum instead of a flat average. This preserves
    the discriminative regions that plain average pooling washes out — most
    useful at early exits where the feature map is still large.

    Returns [B, C, 1, 1] so the downstream flatten/fc path is unchanged.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.attn = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        scores  = self.attn(x).view(B, 1, H * W)     # [B, 1, HW]
        weights = F.softmax(scores, dim=-1)          # [B, 1, HW] — sums to 1 over space
        flat    = x.view(B, C, H * W)                # [B, C, HW]
        pooled  = (flat * weights).sum(dim=-1)       # [B, C]
        return pooled.view(B, C, 1, 1)


class ExitHead(nn.Module):
    """
    Classifier attached to an intermediate ResNet feature map.

    If use_conv=True adds a 3×3 conv block before pooling to let the head
    learn richer spatial features rather than only relying on global avg-pool.

    If use_attention=True the global average pool is replaced by a learned
    SpatialAttentionPool.

    If shared_proj is not None the head uses a three-stage classifier:
        adapter (in_channels → embed_dim)  [per-exit]
          → shared_proj (embed_dim → embed_dim)  [ONE module across all exits]
          → classifier (embed_dim → num_classes)  [per-exit]
    The shared module is owned by EarlyExitResNet and held here via a
    list-wrapped attribute so it is not re-registered per head.
    """
    def __init__(self, in_channels: int, num_classes: int,
                 hidden_dim: int = 512, use_conv: bool = False,
                 use_attention: bool = False,
                 shared_proj: Optional[nn.Module] = None,
                 embed_dim: int = 128):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
            )
        self.pool = (SpatialAttentionPool(in_channels) if use_attention
                     else nn.AdaptiveAvgPool2d((1, 1)))

        self.shared = shared_proj is not None
        self.drop = nn.Dropout(0.1)
        if self.shared:
            self.adapter    = nn.Linear(in_channels, embed_dim)
            self._shared    = [shared_proj]   # list-wrapped: not registered as a submodule
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.fc1 = nn.Linear(in_channels, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            x = self.conv(x)
        x = self.pool(x).flatten(1)
        if self.shared:
            x = F.relu(self.adapter(x))
            x = F.relu(self._shared[0](x))
            return self.classifier(self.drop(x))
        return self.fc2(self.drop(F.relu(self.fc1(x))))


class GateMLP(nn.Module):
    """
    5-dim feature vector → binary exit/continue decision.

    Features: [max_conf, entropy, logit_margin, depth_norm, logits_l2_norm]
    """
    def __init__(self, in_dim: int = 5, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# Channel widths per layer for supported backbones
_FEAT_DIMS = {
    "resnet18": {1: 64,   2: 128,  3: 256,  4: 512},
    "resnet50": {1: 256,  2: 512,  3: 1024, 4: 2048},
}


class EarlyExitResNet(nn.Module):
    """
    ResNet18/50 with configurable early exit points.

    exit_layers: list of layer indices (1–4) where exits are attached.
                 Default [2, 3, 4] matches v1 behaviour.
    use_exit_conv: if True, each ExitHead gets an extra 3×3 conv.
    cifar_stem: replace 7×7 conv + maxpool with 3×3 conv + identity
                (standard for CIFAR-32; for TinyImageNet-64 keep False).
    """
    def __init__(self, model_name: str, num_classes: int,
                 exit_layers: List[int] = (2, 3, 4),
                 use_exit_conv: bool = False,
                 cifar_stem: bool = True,
                 use_attention: bool = False,
                 shared_projection: bool = False,
                 shared_embed_dim: int = 128):
        super().__init__()
        assert model_name in ("resnet18", "resnet50")
        self.exit_layers = list(exit_layers)
        self.num_exits = len(exit_layers)
        feat_dims = _FEAT_DIMS[model_name]

        if model_name == "resnet18":
            backbone = torchvision.models.resnet18(weights=None)
        else:
            backbone = torchvision.models.resnet50(weights=None)

        if cifar_stem:
            backbone.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            backbone.maxpool = nn.Identity()

        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.backbone = backbone

        # One projection shared across all exit heads (owned here so it appears
        # exactly once in state_dict / parameters); None keeps the v1 layout.
        self.shared_proj = (nn.Linear(shared_embed_dim, shared_embed_dim)
                            if shared_projection else None)

        self.exit_heads = nn.ModuleList([
            ExitHead(feat_dims[l], num_classes, use_conv=use_exit_conv,
                     use_attention=use_attention,
                     shared_proj=self.shared_proj, embed_dim=shared_embed_dim)
            for l in exit_layers
        ])
        self.gates = nn.ModuleList([GateMLP() for _ in exit_layers])

    # ---- forward helpers ------------------------------------------------

    def _backbone_features(self, x: torch.Tensor) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """Run full backbone, capturing feature maps at each exit layer."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        feats: Dict[int, torch.Tensor] = {}
        for layer_idx in range(1, 5):
            layer = getattr(self.backbone, f"layer{layer_idx}")
            x = layer(x)
            if layer_idx in self.exit_layers:
                feats[layer_idx] = x

        pooled = self.backbone.avgpool(x)
        pooled = torch.flatten(pooled, 1)
        final_logits = self.backbone.fc(pooled)
        return feats, final_logits

    def forward_with_exits(self, x: torch.Tensor
                           ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        feats, final_logits = self._backbone_features(x)
        exit_logits = [
            head(feats[l])
            for head, l in zip(self.exit_heads, self.exit_layers)
        ]
        return exit_logits, final_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def _cascade_steps(self, x: torch.Tensor):
        """
        Lazily run the backbone layer-by-layer for a single sample (B == 1).

        Yields (exit_idx, logits, feat) for each configured exit head, in
        order, then a final (num_exits, final_logits, None) for the backbone
        head. Because this is a generator, layers after the one a consumer
        stops at are never executed.
        """
        out = self.backbone.conv1(x)
        out = self.backbone.bn1(out)
        out = self.backbone.relu(out)
        out = self.backbone.maxpool(out)

        exit_map = {l: i for i, l in enumerate(self.exit_layers)}
        for layer_idx in range(1, 5):
            out = getattr(self.backbone, f"layer{layer_idx}")(out)
            if layer_idx in exit_map:
                exit_idx = exit_map[layer_idx]
                logits = self.exit_heads[exit_idx](out)
                yield exit_idx, logits, out

        pooled = torch.flatten(self.backbone.avgpool(out), 1)
        final_logits = self.backbone.fc(pooled)
        yield self.num_exits, final_logits, None

    # ---- temperature helpers --------------------------------------------

    def _get_temperature(self, exit_idx: int) -> float:
        """Return calibration temperature for exit_idx (1.0 if not calibrated)."""
        temps = getattr(self, 'exit_temperatures', None)
        if temps is None or exit_idx >= len(temps):
            return 1.0
        return float(temps[exit_idx])

    def _scaled_probs(self, logits: torch.Tensor, exit_idx: int) -> torch.Tensor:
        T = self._get_temperature(exit_idx)
        return F.softmax(logits / T, 1)

    # ---- gate features --------------------------------------------------

    def _gate_features(self, probs: torch.Tensor,
                       logits: torch.Tensor,
                       depth_norm: float) -> torch.Tensor:
        if _USE_TRITON_GATE and _TRITON_AVAILABLE and probs.is_cuda:
            return gate_features_triton(probs, logits, depth_norm)
        with torch.no_grad():
            max_conf, _ = probs.max(1)
            ent = entropy_from_probs(probs)
            top2, _ = torch.topk(logits, k=2, dim=1)
            margin = top2[:, 0] - top2[:, 1]
            l2 = logits.norm(p=2, dim=1)
        return torch.stack([
            max_conf, ent, margin,
            torch.full_like(max_conf, depth_norm),
            l2,
        ], dim=1)

    # ---- inference policies ---------------------------------------------

    def inference_static(self, x: torch.Tensor,
                         tau: float = 0.9,
                         tau_entropy: Optional[float] = None,
                         cascade: bool = False
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Exit when max_conf >= tau (and optionally entropy <= tau_entropy).

        cascade=True: batch_size==1 only. Runs the backbone layer-by-layer
        via _cascade_steps and returns as soon as an exit fires, so layers
        after the exit point are never executed.
        """
        if cascade:
            assert x.size(0) == 1, "cascade=True only supports batch size 1"
            for exit_idx, logits, _ in self._cascade_steps(x):
                if exit_idx == self.num_exits:
                    return logits, torch.tensor([exit_idx], device=x.device)
                probs = self._scaled_probs(logits, exit_idx)
                max_conf, _ = probs.max(1)
                condition = max_conf >= tau
                if tau_entropy is not None:
                    ent = entropy_from_probs(probs)
                    condition = condition & (ent <= tau_entropy)
                if condition.item():
                    return logits, torch.tensor([exit_idx], device=x.device)

        B = x.size(0)
        num_cls = self.backbone.fc.out_features
        logits_out = torch.zeros(B, num_cls, device=x.device)
        exit_ids   = torch.full((B,), self.num_exits, dtype=torch.long, device=x.device)

        exit_logits_list, final_logits = self.forward_with_exits(x)
        decided = torch.zeros(B, dtype=torch.bool, device=x.device)

        for i, exit_logits in enumerate(exit_logits_list):
            probs = self._scaled_probs(exit_logits, i)
            max_conf, _ = probs.max(1)
            condition = (max_conf >= tau) & (~decided)
            if tau_entropy is not None:
                ent = entropy_from_probs(probs)
                condition = condition & (ent <= tau_entropy)
            if condition.any():
                idx = condition.nonzero(as_tuple=False).squeeze(1)
                logits_out[idx] = exit_logits[idx]
                exit_ids[idx] = i
                decided[idx] = True

        remaining = ~decided
        if remaining.any():
            idx = remaining.nonzero(as_tuple=False).squeeze(1)
            logits_out[idx] = final_logits[idx]

        return logits_out, exit_ids

    def inference_dynamic(self, x: torch.Tensor,
                          gate_threshold: float = 0.8,
                          cascade: bool = False
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gate MLP decides exit at each checkpoint.

        cascade=True: batch_size==1 only. Runs the backbone layer-by-layer
        via _cascade_steps and returns as soon as the gate fires, so layers
        after the exit point are never executed.
        """
        if cascade:
            assert x.size(0) == 1, "cascade=True only supports batch size 1"
            for exit_idx, logits, feat in self._cascade_steps(x):
                if exit_idx == self.num_exits:
                    return logits, torch.tensor([exit_idx], device=x.device)
                probs = self._scaled_probs(logits, exit_idx)
                dnorm = (exit_idx + 1) / self.num_exits
                feats = self._gate_features(probs, logits, dnorm)
                gate_prob = torch.sigmoid(self.gates[exit_idx](feats))
                if (gate_prob >= gate_threshold).item():
                    return logits, torch.tensor([exit_idx], device=x.device)

        B = x.size(0)
        num_cls = self.backbone.fc.out_features
        logits_out = torch.zeros(B, num_cls, device=x.device)
        exit_ids   = torch.full((B,), self.num_exits, dtype=torch.long, device=x.device)

        exit_logits_list, final_logits = self.forward_with_exits(x)
        depth_norms = [i / self.num_exits for i in range(1, self.num_exits + 1)]
        decided = torch.zeros(B, dtype=torch.bool, device=x.device)

        for i, (exit_logits, gate, dnorm) in enumerate(
                zip(exit_logits_list, self.gates, depth_norms)):
            probs = self._scaled_probs(exit_logits, i)
            feats = self._gate_features(probs, exit_logits, dnorm)
            gate_prob = torch.sigmoid(gate(feats))
            should_exit = (gate_prob >= gate_threshold) & (~decided)
            if should_exit.any():
                idx = should_exit.nonzero(as_tuple=False).squeeze(1)
                logits_out[idx] = exit_logits[idx]
                exit_ids[idx] = i
                decided[idx] = True

        remaining = ~decided
        if remaining.any():
            idx = remaining.nonzero(as_tuple=False).squeeze(1)
            logits_out[idx] = final_logits[idx]

        return logits_out, exit_ids


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------

def calibrate_temperature(model: EarlyExitResNet,
                          valloader: DataLoader,
                          device: torch.device) -> List[float]:
    """
    Post-training temperature scaling: find per-exit scalar T that minimises
    NLL on the validation set via L-BFGS. Stores results in model.exit_temperatures.
    Returns the list of temperatures [T_exit0, T_exit1, ..., T_backbone].
    """
    model.eval()
    model.to(device)
    n_exits = model.num_exits + 1

    # Collect all logits and labels per exit point
    all_logits: List[List[torch.Tensor]] = [[] for _ in range(n_exits)]
    all_labels: List[List[torch.Tensor]] = [[] for _ in range(n_exits)]

    with torch.no_grad():
        for images, targets in valloader:
            images, targets = images.to(device), targets.to(device)
            exit_logits_list, final_logits = model.forward_with_exits(images)
            for i, logits in enumerate(exit_logits_list):
                all_logits[i].append(logits.cpu())
                all_labels[i].append(targets.cpu())
            all_logits[model.num_exits].append(final_logits.cpu())
            all_labels[model.num_exits].append(targets.cpu())

    temperatures: List[float] = []
    nll = nn.CrossEntropyLoss()

    for i in range(n_exits):
        logits_cat = torch.cat(all_logits[i])
        labels_cat = torch.cat(all_labels[i])
        T = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            loss = nll(logits_cat / T, labels_cat)
            loss.backward()
            return loss

        optimizer.step(closure)
        T_val = max(T.item(), 0.1)  # clamp: T must be positive
        temperatures.append(T_val)
        label = f"Exit {i}" if i < model.num_exits else "Backbone"
        print(f"  [Calibrate] {label}: T = {T_val:.4f}")

    model.exit_temperatures = temperatures
    return temperatures


def freeze(model: EarlyExitResNet):
    for p in model.backbone.parameters():
        p.requires_grad_(False)


def unfreeze(model: EarlyExitResNet):
    for p in model.backbone.parameters():
        p.requires_grad_(True)


def train_backbone(model: EarlyExitResNet,
                   trainloader: DataLoader,
                   testloader: DataLoader,
                   device: torch.device,
                   epochs: int,
                   lr: float,
                   logger: Logger):
    if epochs == 0:
        print("[Backbone] Skipping (epochs=0).")
        return
    print(f"[Backbone] Training {epochs} epochs, lr={lr}")
    model.to(device)
    unfreeze(model)
    optimizer = torch.optim.SGD(
        model.backbone.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(0.5 * epochs), int(0.75 * epochs)],
        gamma=0.1,
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, targets in trainloader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, targets in testloader:
                images, targets = images.to(device), targets.to(device)
                preds = model(images).argmax(1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        acc = correct / total * 100
        avg_loss = total_loss / len(trainloader.dataset)
        print(f"  [Backbone][{epoch+1}/{epochs}] loss={avg_loss:.4f} acc={acc:.2f}%")
        logger.scalar("backbone/loss", avg_loss, epoch)
        logger.scalar("backbone/acc",  acc,      epoch)


def _symmetric_kl(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """Mean symmetric KL = 0.5*(KL(p‖q)+KL(q‖p)) between two logit tensors.

    Gradients flow to both arguments on purpose — the term pulls the two
    predictions toward each other rather than distilling one into the other.
    """
    log_p = F.log_softmax(logits_p, dim=1)
    log_q = F.log_softmax(logits_q, dim=1)
    p, q = log_p.exp(), log_q.exp()
    kl_pq = (p * (log_p - log_q)).sum(1).mean()
    kl_qp = (q * (log_q - log_p)).sum(1).mean()
    return 0.5 * (kl_pq + kl_qp)


def train_joint(model: EarlyExitResNet,
                trainloader: DataLoader,
                testloader: DataLoader,
                device: torch.device,
                epochs: int,
                lr: float,
                exit_loss_weight: float,
                logger: Logger,
                consistency_weight: float = 0.0):
    """
    Joint end-to-end training: backbone + exit heads together.
    L = L_backbone + w * sum(L_exit_i)  using ground-truth cross-entropy.

    If consistency_weight > 0, adds a symmetric-KL term between each adjacent
    pair in the chain [exit0, …, exitN-1, backbone] so earlier exits are
    nudged to agree with deeper predictions (more reliable gate confidences).
    """
    if epochs == 0:
        print("[Joint] Skipping (epochs=0).")
        return
    # Depth-weighted supervision: earlier exits are harder → more weight.
    # Linear schedule: exit 0 → exit_loss_weight, last exit → exit_loss_weight/num_exits.
    # Average across exits equals exit_loss_weight, so the hyperparameter meaning is preserved.
    n = model.num_exits
    depth_weights = [exit_loss_weight * (n - i) / n for i in range(n)]
    print(f"[Joint] Training {epochs} epochs, lr={lr}, "
          f"exit_weights={[f'{w:.3f}' for w in depth_weights]}")
    model.to(device)
    unfreeze(model)

    # Train backbone + all exit heads jointly
    params = list(model.backbone.parameters())
    for head in model.exit_heads:
        params += list(head.parameters())
    # Shared projection (if any) is owned by the model, not the heads — add once.
    if getattr(model, "shared_proj", None) is not None:
        params += list(model.shared_proj.parameters())

    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(0.5 * epochs), int(0.75 * epochs)],
        gamma=0.1,
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, targets in trainloader:
            images, targets = images.to(device), targets.to(device)
            exit_logits_list, final_logits = model.forward_with_exits(images)

            loss = criterion(final_logits, targets)
            for i, exit_logits in enumerate(exit_logits_list):
                loss = loss + depth_weights[i] * criterion(exit_logits, targets)

            if consistency_weight > 0.0:
                chain = exit_logits_list + [final_logits]
                cons = sum(_symmetric_kl(a, b)
                           for a, b in zip(chain[:-1], chain[1:]))
                loss = loss + consistency_weight * cons

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, targets in testloader:
                images, targets = images.to(device), targets.to(device)
                preds = model(images).argmax(1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        acc = correct / total * 100
        avg_loss = total_loss / len(trainloader.dataset)
        print(f"  [Joint][{epoch+1}/{epochs}] loss={avg_loss:.4f} backbone_acc={acc:.2f}%")
        logger.scalar("joint/loss", avg_loss, epoch)
        logger.scalar("joint/acc",  acc,      epoch)


def train_exit_heads_distillation(model: EarlyExitResNet,
                                  trainloader: DataLoader,
                                  device: torch.device,
                                  epochs: int,
                                  lr: float,
                                  T_start: float,
                                  T_end: float,
                                  logger: Logger):
    """
    Distil from frozen backbone into exit heads with curriculum temperature
    that anneals linearly from T_start → T_end over epochs.
    """
    print(f"[Exits] Training {epochs} epochs, lr={lr}, T: {T_start}→{T_end}")
    model.to(device)
    freeze(model)
    # requires_grad=False (freeze) does not stop BatchNorm running-stat
    # updates — those are buffers, not parameters, and update on any
    # train-mode forward regardless of grad. Keep the backbone in eval mode
    # so its BN stats don't drift while "frozen"; exit heads need
    # train-mode behavior (dropout, and BatchNorm if --exit_conv is set —
    # that BN is intentionally still trained here, it belongs to the head).
    model.eval()
    for head in model.exit_heads:
        head.train()

    params = [p for head in model.exit_heads for p in head.parameters()]
    optimizer = torch.optim.Adam(params, lr=lr)
    kldiv = nn.KLDivLoss(reduction="batchmean")

    for epoch in range(epochs):
        # Linearly anneal temperature
        T = T_start + (T_end - T_start) * epoch / max(epochs - 1, 1)
        total_loss = 0.0
        for images, _ in trainloader:
            images = images.to(device)
            # Single backbone forward serves both the teacher signal
            # (final_logits) and the exit-head inputs — the backbone is
            # frozen and in eval mode, so running it twice was pure waste.
            exit_logits_list, final_logits = model.forward_with_exits(images)
            teacher_probs = F.softmax(final_logits.detach() / T, 1)
            loss = sum(
                kldiv(F.log_softmax(el / T, 1), teacher_probs)
                for el in exit_logits_list
            ) / len(exit_logits_list)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(trainloader.dataset)
        print(f"  [Exits][{epoch+1}/{epochs}] T={T:.2f} loss={avg_loss:.4f}")
        logger.scalar("exits/loss",  avg_loss, epoch)
        logger.scalar("exits/T",     T,        epoch)


def collect_gate_data(model: EarlyExitResNet,
                      trainloader: DataLoader,
                      device: torch.device,
                      max_batches: int,
                      gate_label_conf: float) -> Dict[str, torch.Tensor]:
    print(f"[Gates] Collecting data (max {max_batches} batches)...")
    model.to(device)
    model.eval()
    freeze(model)

    buckets: Dict[str, list] = {
        f"{k}{i}": [] for k in ("feats", "labels") for i in range(model.num_exits)
    }
    depth_norms = [i / model.num_exits for i in range(1, model.num_exits + 1)]

    with torch.no_grad():
        for b_idx, (images, _) in enumerate(trainloader):
            if b_idx >= max_batches:
                break
            images = images.to(device)
            exit_logits_list, final_logits = model.forward_with_exits(images)
            final_preds = final_logits.argmax(1)

            for i, (exit_logits, dnorm) in enumerate(
                    zip(exit_logits_list, depth_norms)):
                probs = model._scaled_probs(exit_logits, i)
                max_conf, _ = probs.max(1)
                labels = ((exit_logits.argmax(1) == final_preds) &
                          (max_conf >= gate_label_conf)).float()
                feats = model._gate_features(probs, exit_logits, dnorm)
                buckets[f"feats{i}"].append(feats.cpu())
                buckets[f"labels{i}"].append(labels.cpu())

    data: Dict[str, torch.Tensor] = {}
    for k, v in buckets.items():
        data[k] = torch.cat(v, 0) if v else torch.empty(0)

    for i in range(model.num_exits):
        feats = data[f"feats{i}"]
        labels = data[f"labels{i}"]
        if feats.numel():
            pos = int(labels.sum().item())
            print(f"  Exit {i}: {feats.size(0)} samples (pos={pos}, neg={feats.size(0)-pos})")
    return data


def train_gates(model: EarlyExitResNet,
                gate_data: Dict[str, torch.Tensor],
                device: torch.device,
                epochs: int,
                lr: float,
                batch_size: int,
                logger: Logger):
    print(f"[Gates] Training {epochs} epochs, lr={lr}")
    model.to(device)
    freeze(model)
    model.train()

    for i, gate in enumerate(model.gates):
        feats  = gate_data[f"feats{i}"]
        labels = gate_data[f"labels{i}"]
        if feats.size(0) == 0:
            print(f"  [Gates] Gate {i}: no data, skipping")
            continue

        pos = labels.sum().item()
        neg = labels.numel() - pos
        pos_weight = torch.tensor([neg / max(pos, 1e-6)], device=device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        loader = DataLoader(
            torch.utils.data.TensorDataset(feats, labels),
            batch_size=batch_size, shuffle=True,
        )
        optimizer = torch.optim.Adam(gate.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0.0
            for Xb, yb in loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = bce(gate(Xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * Xb.size(0)
            avg_loss = total_loss / len(loader.dataset)
            logger.scalar(f"gate{i}/loss", avg_loss, epoch)

        print(f"  [Gates] Gate {i}: trained {epochs} epochs, final loss={avg_loss:.4f}")


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    policy:             str
    description:        str
    accuracy:           float
    avg_latency_ms:     float
    exit_distribution:  List[float]
    avg_flops_fraction: float = 0.0
    per_class_exits:    Dict[int, List[float]] = field(default_factory=dict)


def evaluate_policy(model: EarlyExitResNet,
                    testloader: DataLoader,
                    device: torch.device,
                    policy: str,
                    tau: float = 0.9,
                    tau_entropy: Optional[float] = None,
                    gate_threshold: float = 0.8,
                    mac_profile: Optional[MACProfile] = None,
                    ) -> EvalResult:
    assert policy in ("static", "dynamic")
    model.to(device)
    model.eval()
    freeze(model)

    num_exits = model.num_exits + 1   # exits + final backbone
    num_cls = model.backbone.fc.out_features

    total_correct = 0
    total_samples = 0
    total_time    = 0.0
    total_flops   = 0.0
    exit_counts   = torch.zeros(num_exits, dtype=torch.long)
    # per-class: class_id → list of exit_ids
    class_exit_map: Dict[int, List[int]] = {}

    with torch.no_grad():
        for images, targets in testloader:
            images, targets = images.to(device), targets.to(device)
            _sync(device)
            t0 = time.time()
            if policy == "static":
                logits, exit_ids = model.inference_static(
                    images, tau=tau, tau_entropy=tau_entropy)
            else:
                logits, exit_ids = model.inference_dynamic(
                    images, gate_threshold=gate_threshold)
            _sync(device)
            total_time += time.time() - t0

            preds = logits.argmax(1)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)

            for e in range(num_exits):
                exit_counts[e] += (exit_ids == e).sum().item()

            # FLOPs
            if mac_profile:
                for b in range(images.size(0)):
                    eid = exit_ids[b].item()
                    if eid < model.num_exits:
                        l_idx = model.exit_layers[eid]
                        total_flops += mac_profile.flops_fraction(eid, l_idx)
                    else:
                        total_flops += 1.0

            # Per-class exits
            for b in range(targets.size(0)):
                c = targets[b].item()
                class_exit_map.setdefault(c, []).append(exit_ids[b].item())

    acc  = total_correct / total_samples * 100
    dist = (exit_counts.float() / total_samples * 100).tolist()
    avg_lat = (total_time / len(testloader)) * 1000

    avg_flops = (total_flops / total_samples) if mac_profile else 0.0

    # Per-class exit distribution (fraction per exit point)
    per_class: Dict[int, List[float]] = {}
    for cls_id, exits in class_exit_map.items():
        counts = [0.0] * num_exits
        for e in exits:
            counts[e] += 1
        total_c = max(len(exits), 1)
        per_class[cls_id] = [c / total_c * 100 for c in counts]

    desc = (f"policy=static tau={tau}"
            + (f" tau_entropy={tau_entropy}" if tau_entropy else "")
            if policy == "static"
            else f"policy=dynamic gate_threshold={gate_threshold}")

    return EvalResult(
        policy=policy,
        description=desc,
        accuracy=acc,
        avg_latency_ms=avg_lat,
        exit_distribution=dist,
        avg_flops_fraction=avg_flops,
        per_class_exits=per_class,
    )


def print_result(r: EvalResult, model: EarlyExitResNet):
    n = model.num_exits
    print(f"\nPolicy       : {r.description}")
    print(f"Accuracy     : {r.accuracy:.2f}%")
    print(f"Latency/batch: {r.avg_latency_ms:.3f} ms")
    if r.avg_flops_fraction > 0:
        print(f"Avg FLOPs    : {r.avg_flops_fraction * 100:.1f}% of full backbone")
    print("Exit dist    :")
    for i in range(n):
        l = model.exit_layers[i]
        print(f"  Exit {i} (after layer{l}): {r.exit_distribution[i]:.1f}%")
    print(f"  Final backbone       : {r.exit_distribution[n]:.1f}%")


def print_per_class_analysis(r: EvalResult, model: EarlyExitResNet,
                             top_k: int = 10):
    """Print which classes exit earliest/latest on average."""
    n = model.num_exits
    # Average exit index per class (lower = exits earlier)
    avg_exit: Dict[int, float] = {}
    for cls_id, dist in r.per_class_exits.items():
        avg_exit[cls_id] = sum(i * d / 100 for i, d in enumerate(dist))

    sorted_cls = sorted(avg_exit.items(), key=lambda x: x[1])
    print(f"\nPer-class exit analysis (top {top_k} earliest / {top_k} latest):")
    print("  Earliest exiting classes (avg exit point):")
    for cls_id, avg in sorted_cls[:top_k]:
        print(f"    class {cls_id:3d}: avg_exit={avg:.2f}")
    print("  Latest exiting classes:")
    for cls_id, avg in sorted_cls[-top_k:]:
        print(f"    class {cls_id:3d}: avg_exit={avg:.2f}")


# ---------------------------------------------------------------------------
#  Pareto sweep
# ---------------------------------------------------------------------------

def pareto_sweep(model: EarlyExitResNet,
                 testloader: DataLoader,
                 device: torch.device,
                 policy: str,
                 mac_profile: Optional[MACProfile],
                 tau_values: Optional[List[float]] = None,
                 threshold_values: Optional[List[float]] = None,
                 ) -> List[Dict]:
    """
    Sweep tau (static) or gate_threshold (dynamic) and print the
    accuracy–FLOPs Pareto frontier. Returns collected points for plotting.
    """
    if policy == "static":
        values = tau_values or [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
        label = "tau"
    else:
        values = threshold_values or [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        label = "gate_threshold"

    print(f"\n{'='*60}")
    print(f"Pareto Sweep — policy={policy}")
    print(f"{'='*60}")
    header = f"  {label:<18}  acc(%)   flops(%)  lat(ms)"
    print(header)
    print("  " + "-" * (len(header) - 2))

    points: List[Dict] = []
    for v in values:
        if policy == "static":
            r = evaluate_policy(model, testloader, device, "static",
                                tau=v, mac_profile=mac_profile)
        else:
            r = evaluate_policy(model, testloader, device, "dynamic",
                                gate_threshold=v, mac_profile=mac_profile)
        flops_str = f"{r.avg_flops_fraction*100:.1f}" if r.avg_flops_fraction else "n/a"
        print(f"  {v:<18.3f}  {r.accuracy:6.2f}   {flops_str:>7}   {r.avg_latency_ms:.2f}")
        if r.avg_flops_fraction:
            points.append({
                "threshold": v,
                "accuracy": r.accuracy,
                "flops_pct": r.avg_flops_fraction * 100,
                "latency_ms": r.avg_latency_ms,
            })
    return points


def plot_pareto_curves(static_pts: List[Dict],
                       dynamic_pts: List[Dict],
                       plot_dir: str,
                       dataset: str = "") -> None:
    """Save accuracy vs FLOPs Pareto curve to plot_dir/pareto.png."""
    if not _PLOT_AVAILABLE:
        print("[Plot] matplotlib/numpy not available — skipping Pareto plot.")
        return
    os.makedirs(plot_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    title_suffix = f" — {dataset}" if dataset else ""

    for ax, pts, label, color in [
        (axes[0], static_pts,  "Static (τ sweep)",        "steelblue"),
        (axes[1], dynamic_pts, "Dynamic (gate threshold)", "darkorange"),
    ]:
        if not pts:
            ax.set_title(f"{label}\n(no FLOPs data)")
            continue
        xs = [p["flops_pct"] for p in pts]
        ys = [p["accuracy"]  for p in pts]
        ax.plot(xs, ys, "o-", color=color, linewidth=2, markersize=6)
        for p in pts:
            ax.annotate(f"{p['threshold']:.2f}",
                        (p["flops_pct"], p["accuracy"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_xlabel("Avg FLOPs (%)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{label}{title_suffix}")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"SelfXit Accuracy–Compute Pareto{title_suffix}", fontsize=13)
    fig.tight_layout()
    out_path = os.path.join(plot_dir, "pareto.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Pareto curve saved → {out_path}")


def plot_difficulty_analysis(model: EarlyExitResNet,
                             testloader: DataLoader,
                             device: torch.device,
                             policy: str,
                             plot_dir: str,
                             dataset: str = "",
                             tau: float = 0.9,
                             gate_threshold: float = 0.8) -> None:
    """
    Validate the gate hypothesis: easy samples (large logit margin) exit early,
    hard samples (small margin) reach the full backbone.
    Plots a violin chart of logit margin distribution per exit point.
    """
    if not _PLOT_AVAILABLE:
        print("[Plot] matplotlib/numpy not available — skipping difficulty analysis.")
        return

    model.eval()
    model.to(device)
    n_exits = model.num_exits + 1
    per_exit_margins: Dict[int, List[float]] = {i: [] for i in range(n_exits)}

    with torch.no_grad():
        for images, _ in testloader:
            images = images.to(device)
            if policy == "static":
                logits_out, exit_ids = model.inference_static(images, tau=tau)
            else:
                logits_out, exit_ids = model.inference_dynamic(
                    images, gate_threshold=gate_threshold)
            top2 = torch.topk(logits_out, k=2, dim=1).values
            margins = (top2[:, 0] - top2[:, 1]).cpu().tolist()
            for margin, eid in zip(margins, exit_ids.cpu().tolist()):
                per_exit_margins[eid].append(margin)

    labels = [f"Exit {i}\n(layer{model.exit_layers[i]})"
              for i in range(model.num_exits)] + ["Full\nbackbone"]
    data = [per_exit_margins[i] for i in range(n_exits)]
    non_empty = [(lbl, d) for lbl, d in zip(labels, data) if d]
    if not non_empty:
        print("[Plot] No samples collected for difficulty analysis.")
        return

    os.makedirs(plot_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    vp = ax.violinplot([d for _, d in non_empty], showmedians=True)
    for body in vp["bodies"]:
        body.set_alpha(0.7)
    ax.set_xticks(range(1, len(non_empty) + 1))
    ax.set_xticklabels([lbl for lbl, _ in non_empty])
    ax.set_ylabel("Logit margin (top-1 − top-2)")
    ax.set_xlabel("Exit point")
    title_suffix = f" — {dataset}" if dataset else ""
    ax.set_title(f"Sample Difficulty vs Exit Depth{title_suffix}\n"
                 f"(higher margin = more confident = easier)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(plot_dir, "difficulty.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Difficulty analysis saved → {out_path}")


# ---------------------------------------------------------------------------
#  Budget-aware inference (find gate_threshold for target FLOPs)
# ---------------------------------------------------------------------------

def find_budget_threshold(model: EarlyExitResNet,
                          testloader: DataLoader,
                          device: torch.device,
                          target_budget: float,
                          mac_profile: MACProfile,
                          max_batches: int = 50) -> float:
    """
    Binary search for the gate_threshold that achieves avg FLOPs ≈ target_budget.
    Higher threshold → harder to exit → more FLOPs.
    """
    print(f"[Budget] Searching for threshold that achieves {target_budget*100:.0f}% FLOPs...")

    def avg_flops(threshold: float) -> float:
        model.eval()
        total_f = 0.0
        total_n = 0
        with torch.no_grad():
            for b_idx, (images, _) in enumerate(testloader):
                if b_idx >= max_batches:
                    break
                images = images.to(device)
                _, exit_ids = model.inference_dynamic(images, gate_threshold=threshold)
                for b in range(images.size(0)):
                    eid = exit_ids[b].item()
                    if eid < model.num_exits:
                        total_f += mac_profile.flops_fraction(eid, model.exit_layers[eid])
                    else:
                        total_f += 1.0
                total_n += images.size(0)
        return total_f / max(total_n, 1)

    lo, hi = 0.05, 0.99
    for _ in range(18):
        mid = (lo + hi) / 2
        f = avg_flops(mid)
        if f > target_budget:
            hi = mid   # too many FLOPs → lower threshold (exit sooner)
        else:
            lo = mid   # too few FLOPs → raise threshold

    best = (lo + hi) / 2
    print(f"[Budget] Found gate_threshold={best:.4f} → avg {avg_flops(best)*100:.1f}% FLOPs")
    return best


# ---------------------------------------------------------------------------
#  Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(model: EarlyExitResNet, path: str, tag: str = "model"):
    import json
    os.makedirs(path, exist_ok=True)
    fpath = os.path.join(path, f"{tag}.pt")
    torch.save(model.state_dict(), fpath)
    print(f"[Checkpoint] Saved → {fpath}")
    temps = getattr(model, 'exit_temperatures', None)
    if temps is not None:
        tpath = os.path.join(path, f"{tag}_temperatures.json")
        with open(tpath, "w") as f:
            json.dump(temps, f)
        print(f"[Checkpoint] Temperatures saved → {tpath}")


def load_checkpoint(model: EarlyExitResNet, path: str, tag: str = "model",
                    device: torch.device = torch.device("cpu")):
    import json
    fpath = os.path.join(path, f"{tag}.pt")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Checkpoint not found: {fpath}")
    model.load_state_dict(torch.load(fpath, map_location=device, weights_only=True))
    print(f"[Checkpoint] Loaded ← {fpath}")
    tpath = os.path.join(path, f"{tag}_temperatures.json")
    if os.path.exists(tpath):
        with open(tpath) as f:
            model.exit_temperatures = json.load(f)
        print(f"[Checkpoint] Temperatures loaded ← {tpath} {model.exit_temperatures}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def benchmark_single_sample(model: EarlyExitResNet,
                            testloader: DataLoader,
                            device: torch.device,
                            policy: str,
                            n_runs: int = 1000,
                            tau: float = 0.9,
                            gate_threshold: float = 0.8) -> None:
    """
    Run n_runs single-sample forward passes and report P50/P95/P99 latency,
    broken down by which exit was taken, for both the full-backbone
    (cascade=False) and early-stop (cascade=True) inference paths.

    Every run also checks that cascade and non-cascade agree (same exit_id,
    allclose logits) — this is the correctness guard for the lazy
    layer-by-layer path in _cascade_steps.
    Batch-size-1 latency is the metric that matters for online serving.
    """
    if not _PLOT_AVAILABLE:
        print("[Benchmark] numpy not available — skipping single-sample benchmark.")
        return

    model.eval()
    model.to(device)

    # Collect all test images into a flat list for random sampling
    all_images: List[torch.Tensor] = []
    for imgs, _ in testloader:
        for img in imgs:
            all_images.append(img)

    import random
    n_exits = model.num_exits + 1  # exits + final backbone

    def run_inference(img: torch.Tensor, cascade: bool):
        if policy == "static":
            return model.inference_static(img, tau=tau, cascade=cascade)
        return model.inference_dynamic(img, gate_threshold=gate_threshold, cascade=cascade)

    results: Dict[bool, Dict[str, List]] = {
        False: {"latencies": [], "exits": []},
        True: {"latencies": [], "exits": []},
    }

    with torch.no_grad():
        for _ in range(n_runs):
            img = random.choice(all_images).unsqueeze(0).to(device)

            _sync(device)
            t0 = time.perf_counter()
            logits_full, eid_full = run_inference(img, cascade=False)
            _sync(device)
            results[False]["latencies"].append((time.perf_counter() - t0) * 1000)
            results[False]["exits"].append(eid_full[0].item())

            _sync(device)
            t0 = time.perf_counter()
            logits_cascade, eid_cascade = run_inference(img, cascade=True)
            _sync(device)
            results[True]["latencies"].append((time.perf_counter() - t0) * 1000)
            results[True]["exits"].append(eid_cascade[0].item())

            assert eid_cascade.item() == eid_full.item(), (
                f"cascade/non-cascade exit mismatch: "
                f"{eid_cascade.item()} vs {eid_full.item()}")
            assert torch.allclose(logits_cascade, logits_full, atol=1e-5), (
                "cascade/non-cascade logits mismatch")

    labels = [f"Exit {i}" for i in range(model.num_exits)] + ["Full backbone"]

    def print_table(title: str, latencies: List[float], exit_taken: List[int]):
        lats = np.array(latencies)
        exits = np.array(exit_taken)
        print(f"\n{'='*60}")
        print(f"Single-Sample Latency Benchmark — {title}  (policy={policy}, n={n_runs})")
        print(f"{'='*60}")
        print(f"  Overall   P50={np.percentile(lats,50):.2f}ms  "
              f"P95={np.percentile(lats,95):.2f}ms  "
              f"P99={np.percentile(lats,99):.2f}ms")
        print()
        for e in range(n_exits):
            mask = exits == e
            if mask.sum() == 0:
                continue
            e_lats = lats[mask]
            pct = mask.sum() / n_runs * 100
            print(f"  {labels[e]:<14} ({pct:5.1f}% of samples)  "
                  f"P50={np.percentile(e_lats,50):.2f}ms  "
                  f"P95={np.percentile(e_lats,95):.2f}ms  "
                  f"P99={np.percentile(e_lats,99):.2f}ms")

    print_table("full backbone (no cascade)", results[False]["latencies"], results[False]["exits"])
    print_table("cascade (early-stop)", results[True]["latencies"], results[True]["exits"])


def main():
    parser = argparse.ArgumentParser(
        description="SelfXit v2 — Extended Early-Exit ResNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    parser.add_argument("--dataset", default="cifar10",
                        choices=["cifar10", "cifar100", "tinyimagenet"])
    parser.add_argument("--data_root", default="./data/tiny-imagenet-200",
                        help="Root dir for TinyImageNet.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_frac", type=float, default=0.1,
                        help="Fraction of the training set held out as a "
                             "validation split for calibration/placement-probe/"
                             "budget-search/sweep tuning (never the test set).")

    # Model
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--exits", type=int, nargs="+", default=[2, 3, 4],
                        help="Backbone layers to attach exit heads to (1–4).")
    parser.add_argument("--exit_conv", action="store_true",
                        help="Add a 3×3 conv block to each exit head.")
    parser.add_argument("--exit_attention", action="store_true",
                        help="Replace avg-pool in exit heads with learned spatial attention pooling.")
    parser.add_argument("--shared_projection", action="store_true",
                        help="Route exit heads through one shared projection layer (adapter→shared→classifier).")
    parser.add_argument("--shared_embed_dim", type=int, default=128,
                        help="Embedding width for --shared_projection.")
    parser.add_argument("--cifar_stem", action="store_true", default=True,
                        help="Use 3×3 conv stem (for CIFAR/TinyImageNet).")
    parser.add_argument("--no_cifar_stem", dest="cifar_stem", action="store_false")

    # Training epochs
    parser.add_argument("--epochs_backbone", type=int, default=0,
                        help="Epochs for standalone backbone training (sequential mode).")
    parser.add_argument("--joint_training", action="store_true",
                        help="Train backbone + exits jointly instead of sequentially.")
    parser.add_argument("--epochs_joint", type=int, default=30,
                        help="Epochs for joint training.")
    parser.add_argument("--exit_loss_weight", type=float, default=0.5,
                        help="Weight for exit losses in joint training.")
    parser.add_argument("--consistency_weight", type=float, default=0.0,
                        help="Weight for symmetric-KL consistency between adjacent exits (joint training).")
    parser.add_argument("--epochs_exits", type=int, default=20,
                        help="Epochs for distillation training of exit heads (sequential mode).")
    parser.add_argument("--epochs_gates", type=int, default=5)

    # LR
    parser.add_argument("--lr_backbone", type=float, default=0.1)
    parser.add_argument("--lr_exits",    type=float, default=1e-3)
    parser.add_argument("--lr_gates",    type=float, default=1e-3)
    parser.add_argument("--lr_joint",    type=float, default=0.05)

    # Distillation
    parser.add_argument("--T_start", type=float, default=4.0,
                        help="Initial distillation temperature (curriculum).")
    parser.add_argument("--T_end",   type=float, default=1.0,
                        help="Final distillation temperature.")

    # Inference
    parser.add_argument("--tau",           type=float, default=0.9)
    parser.add_argument("--tau_entropy",   type=float, default=None,
                        help="Max entropy for static exit (set to enable joint conf+entropy gate).")
    parser.add_argument("--gate_threshold", type=float, default=0.8)
    parser.add_argument("--gate_label_conf", type=float, default=0.8)
    parser.add_argument("--gate_max_batches", type=int, default=500)
    parser.add_argument("--compute_budget", type=float, default=None,
                        help="Target avg FLOPs fraction (0–1). Overrides gate_threshold.")

    # Evaluation
    parser.add_argument("--policy", default="both",
                        choices=["static", "dynamic", "both"])
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--sweep", action="store_true",
                        help="Run Pareto sweep over threshold values.")
    parser.add_argument("--plot_dir", type=str, default=None,
                        help="Directory to save Pareto plot (requires --sweep).")
    parser.add_argument("--per_class", action="store_true",
                        help="Print per-class exit analysis after evaluation.")
    parser.add_argument("--top_k_classes", type=int, default=10,
                        help="How many earliest/latest classes to print.")
    parser.add_argument("--benchmark_single", action="store_true",
                        help="Run single-sample (batch=1) latency benchmark.")
    parser.add_argument("--benchmark_n_runs", type=int, default=1000,
                        help="Number of single-sample runs for --benchmark_single.")
    parser.add_argument("--difficulty_plot", action="store_true",
                        help="Plot logit margin vs exit depth (requires --plot_dir).")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run per-exit temperature scaling calibration after training.")

    # Checkpoints / logging
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory to save/load checkpoints.")
    parser.add_argument("--resume", action="store_true",
                        help="Load checkpoint from --checkpoint_dir before training.")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="TensorBoard log directory.")
    parser.add_argument("--channels_last", action="store_true",
                        help="Use NHWC (channels_last) memory format for ~33%% throughput gain on NVIDIA GPUs.")
    parser.add_argument("--triton_gate", action="store_true",
                        help="Use fused Triton kernel for gate feature computation (CUDA only).")

    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    logger = Logger(log_dir=args.log_dir)

    trainloader, valloader, testloader, num_classes, input_shape = get_loaders(args)

    if args.channels_last and device.type == "cuda":
        trainloader = _channels_last_loader(trainloader)
        valloader   = _channels_last_loader(valloader)
        testloader  = _channels_last_loader(testloader)
        print("[channels_last] DataLoaders wrapped for NHWC format.")

    if args.triton_gate:
        if _TRITON_AVAILABLE and device.type == "cuda":
            global _USE_TRITON_GATE
            _USE_TRITON_GATE = True
            print("[Triton] Gate feature kernel enabled.")
        else:
            print("[Triton] Warning: --triton_gate requested but "
                  f"{'Triton not installed' if not _TRITON_AVAILABLE else 'device is not CUDA'}. "
                  "Falling back to PyTorch.")
    cifar_stem = args.cifar_stem and args.dataset != "tinyimagenet"
    if args.dataset == "tinyimagenet":
        cifar_stem = True   # still use 3×3 stem but keep maxpool (handled inside backbone)

    model = EarlyExitResNet(
        model_name=args.model,
        num_classes=num_classes,
        exit_layers=args.exits,
        use_exit_conv=args.exit_conv,
        cifar_stem=(args.dataset in ("cifar10", "cifar100")),
        use_attention=args.exit_attention,
        shared_projection=args.shared_projection,
        shared_embed_dim=args.shared_embed_dim,
    )
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        print("[channels_last] Model converted to NHWC memory format.")

    if args.resume and args.checkpoint_dir:
        load_checkpoint(model, args.checkpoint_dir, device=device)

    # ---- Training -------------------------------------------------------
    if not args.eval_only:
        if args.joint_training:
            train_joint(model, trainloader, valloader, device,
                        epochs=args.epochs_joint,
                        lr=args.lr_joint,
                        exit_loss_weight=args.exit_loss_weight,
                        logger=logger,
                        consistency_weight=args.consistency_weight)
        else:
            train_backbone(model, trainloader, testloader, device,
                           epochs=args.epochs_backbone,
                           lr=args.lr_backbone,
                           logger=logger)
            train_exit_heads_distillation(model, trainloader, device,
                                          epochs=args.epochs_exits,
                                          lr=args.lr_exits,
                                          T_start=args.T_start,
                                          T_end=args.T_end,
                                          logger=logger)

        # Calibrate before gate training (not after) so collect_gate_data's
        # gate features are computed from the same scaled probs the gate
        # will see at inference (A2). No-op when --calibrate is off, since
        # _get_temperature() defaults to T=1.0 either way.
        if args.calibrate:
            print("\n[Calibrate] Running per-exit temperature scaling...")
            calibrate_temperature(model, valloader, device)

        gate_data = collect_gate_data(model, trainloader, device,
                                      max_batches=args.gate_max_batches,
                                      gate_label_conf=args.gate_label_conf)
        train_gates(model, gate_data, device,
                    epochs=args.epochs_gates,
                    lr=args.lr_gates,
                    batch_size=512,
                    logger=logger)

        if args.checkpoint_dir:
            save_checkpoint(model, args.checkpoint_dir)
    elif args.calibrate:
        # eval_only + calibrate: calibrate the loaded checkpoint's heads
        # without retraining anything (gates are untouched in this branch,
        # same as before this change).
        print("\n[Calibrate] Running per-exit temperature scaling...")
        calibrate_temperature(model, valloader, device)
        if args.checkpoint_dir:
            save_checkpoint(model, args.checkpoint_dir)

    # ---- MAC profiling --------------------------------------------------
    mac_profile: Optional[MACProfile] = None
    try:
        mac_profile = profile_model_macs(model, input_shape, device)
        total = mac_profile.total_backbone_macs()
        print(f"\n[FLOPs Profile] Total backbone MACs: {total / 1e6:.1f} M")
        for i, l in enumerate(model.exit_layers):
            frac = mac_profile.flops_fraction(i, l)
            print(f"  Exit {i} (layer{l}): {frac*100:.1f}% of backbone")
    except Exception as e:
        print(f"[FLOPs Profile] Skipped ({e})")

    # ---- Budget-aware gate_threshold ------------------------------------
    gate_threshold = args.gate_threshold
    if args.compute_budget is not None and mac_profile is not None:
        gate_threshold = find_budget_threshold(
            model, valloader, device,
            target_budget=args.compute_budget,
            mac_profile=mac_profile,
        )

    # ---- Pareto sweep ---------------------------------------------------
    static_pts: List[Dict] = []
    dynamic_pts: List[Dict] = []
    if args.sweep:
        if args.policy in ("static", "both"):
            static_pts = pareto_sweep(model, valloader, device, "static",
                                      mac_profile=mac_profile)
        if args.policy in ("dynamic", "both"):
            dynamic_pts = pareto_sweep(model, valloader, device, "dynamic",
                                       mac_profile=mac_profile)
        if args.plot_dir:
            plot_pareto_curves(static_pts, dynamic_pts,
                               args.plot_dir, dataset=args.dataset)

    # ---- Standard evaluation --------------------------------------------
    print("\n" + "=" * 60)
    print("Standard Evaluation")
    print("=" * 60)

    results: List[EvalResult] = []
    if args.policy in ("static", "both"):
        r = evaluate_policy(model, testloader, device, "static",
                            tau=args.tau,
                            tau_entropy=args.tau_entropy,
                            mac_profile=mac_profile)
        results.append(r)
        print_result(r, model)
        if args.per_class:
            print_per_class_analysis(r, model, top_k=args.top_k_classes)

    if args.policy in ("dynamic", "both"):
        r = evaluate_policy(model, testloader, device, "dynamic",
                            gate_threshold=gate_threshold,
                            mac_profile=mac_profile)
        results.append(r)
        print_result(r, model)
        if args.per_class:
            print_per_class_analysis(r, model, top_k=args.top_k_classes)

    # ---- Difficulty analysis plot ---------------------------------------
    if args.difficulty_plot and args.plot_dir:
        diff_policy = "dynamic" if args.policy == "dynamic" else "static"
        plot_difficulty_analysis(model, testloader, device,
                                 policy=diff_policy,
                                 plot_dir=args.plot_dir,
                                 dataset=args.dataset,
                                 tau=args.tau,
                                 gate_threshold=gate_threshold)

    # ---- Single-sample latency benchmark --------------------------------
    if args.benchmark_single:
        bench_policy = "dynamic" if args.policy == "dynamic" else "static"
        benchmark_single_sample(model, testloader, device,
                                policy=bench_policy,
                                n_runs=args.benchmark_n_runs,
                                tau=args.tau,
                                gate_threshold=gate_threshold)

    logger.close()


if __name__ == "__main__":
    main()
