#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Unified SelfXit + Dynamic MLP Early Exits

Changes vs previous version:
  - Better gate training labels:
      * label = 1 only if exit agrees with backbone AND exit is confident
        (max_conf >= gate_label_conf)
  - Automatic class-imbalance handling for gates using pos_weight in BCE.
  - Slightly richer gate features (added L2 norm of logits).
  - Configurable gate decision threshold (--gate_threshold, default 0.8).
  - Longer default training for exits (epochs_exits=20).
  - pin_memory disabled on MPS / CPU (enabled only on CUDA).

Backbone:
  - torchvision ResNet18 / ResNet50, CIFAR-style modifications.

Datasets:
  - CIFAR-10 / CIFAR-100.

Run example:
  python selfxit_unified_improved.py \
      --dataset cifar100 \
      --model resnet18 \
      --epochs_backbone 5 \
      --epochs_exits 20 \
      --epochs_gates 5 \
      --policy both \
      --gate_threshold 0.8 \
      --gate_label_conf 0.8
"""

import argparse
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T


# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def entropy_from_probs(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Shannon entropy of probabilities: sum(-p log p)."""
    return -(probs * (probs + eps).log()).sum(dim=1)


# ---------------------------------------------------------------------------
#  Dataset loading
# ---------------------------------------------------------------------------

def get_cifar_loaders(dataset: str,
                      batch_size: int,
                      num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    assert dataset in ("cifar10", "cifar100")
    if dataset == "cifar10":
        num_classes = 10
    else:
        num_classes = 100

    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616)),
    ])

    if dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
    else:
        trainset = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform_test
        )

    # pin_memory only helps on CUDA, and warns on MPS.
    pin_memory = torch.cuda.is_available()

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return trainloader, testloader


# ---------------------------------------------------------------------------
#  Early-exit ResNet backbone + exit heads + gate MLPs
# ---------------------------------------------------------------------------

class ExitHead(nn.Module):
    """
    Simple classifier attached to an intermediate ResNet feature map.
    """
    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int = 512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)          # [B, C, 1, 1]
        x = x.view(x.size(0), -1) # [B, C]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GateMLP(nn.Module):
    """
    Lightweight MLP that predicts exit vs continue from a small feature vector.

    Input features (5-dim):
      [max_conf, entropy, logit_margin, depth_norm, logits_l2_norm]
    """
    def __init__(self, in_dim: int = 5, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class EarlyExitResNet(nn.Module):
    """
    ResNet18/50 with three early exits (after layer2, layer3, layer4)
    plus a final backbone head.
    """
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        assert model_name in ("resnet18", "resnet50")
        self.model_name = model_name
        if model_name == "resnet18":
            backbone = torchvision.models.resnet18(weights=None)
            feat_dim_layer2 = 128
            feat_dim_layer3 = 256
            feat_dim_layer4 = 512
        else:
            backbone = torchvision.models.resnet50(weights=None)
            feat_dim_layer2 = 512
            feat_dim_layer3 = 1024
            feat_dim_layer4 = 2048

        # CIFAR-specific changes
        backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        backbone.maxpool = nn.Identity()

        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.backbone = backbone

        # Early-exit heads
        self.exit1 = ExitHead(feat_dim_layer2, num_classes)
        self.exit2 = ExitHead(feat_dim_layer3, num_classes)
        self.exit3 = ExitHead(feat_dim_layer4, num_classes)

        # Gate MLPs
        self.gate1 = GateMLP()
        self.gate2 = GateMLP()
        self.gate3 = GateMLP()

    # -----------------------------
    # Basic forwards
    # -----------------------------

    def forward_backbone_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward_with_exits(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Returns: ([exit1_logits, exit2_logits, exit3_logits], final_backbone_logits)
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        feats2 = x

        x = self.backbone.layer3(x)
        feats3 = x

        x = self.backbone.layer4(x)
        feats4 = x

        pooled = self.backbone.avgpool(feats4)
        pooled = torch.flatten(pooled, 1)
        final_logits = self.backbone.fc(pooled)

        exit_logits1 = self.exit1(feats2)
        exit_logits2 = self.exit2(feats3)
        exit_logits3 = self.exit3(feats4)

        return [exit_logits1, exit_logits2, exit_logits3], final_logits

    # -----------------------------
    # Gate features
    # -----------------------------

    def _make_gate_features(self,
                            exit_probs: torch.Tensor,
                            exit_logits: torch.Tensor,
                            depth_norm: float) -> torch.Tensor:
        """
        Build feature vector for gate MLP:

          - max_conf: max softmax probability
          - entropy: softmax entropy
          - logit_margin: gap between top-1 and top-2 logits
          - depth_norm: scalar [0,1] indicating depth
          - logits_l2_norm: ||logits||_2
        """
        with torch.no_grad():
            max_conf, _ = exit_probs.max(dim=1)
            ent = entropy_from_probs(exit_probs)
            top2_vals, _ = torch.topk(exit_logits, k=2, dim=1)
            margin = top2_vals[:, 0] - top2_vals[:, 1]
            l2 = exit_logits.norm(p=2, dim=1)

        feats = torch.stack(
            [max_conf,
             ent,
             margin,
             torch.full_like(max_conf, depth_norm),
             l2],
            dim=1
        )
        return feats

    # -----------------------------
    # Inference policies
    # -----------------------------

    def inference_static(self,
                         x: torch.Tensor,
                         tau: float = 0.9) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Static SelfXit-style threshold: exit when max_conf >= tau.
        """
        B = x.size(0)
        logits_out = torch.zeros(B, self.backbone.fc.out_features, device=x.device)
        exit_ids = torch.zeros(B, dtype=torch.long, device=x.device)

        exits_logits, final_logits = self.forward_with_exits(x)
        exits_probs = [F.softmax(l, dim=1) for l in exits_logits]

        decided = torch.zeros(B, dtype=torch.bool, device=x.device)

        for i, exit_logits in enumerate(exits_logits):
            probs = exits_probs[i]
            max_conf, _ = probs.max(dim=1)
            should_exit = (max_conf >= tau) & (~decided)
            if should_exit.any():
                idx = should_exit.nonzero(as_tuple=False).squeeze(1)
                logits_out[idx] = exit_logits[idx]
                exit_ids[idx] = i
                decided[idx] = True

        remaining = (~decided)
        if remaining.any():
            idx = remaining.nonzero(as_tuple=False).squeeze(1)
            logits_out[idx] = final_logits[idx]
            exit_ids[idx] = 3

        return logits_out, exit_ids

    def inference_dynamic(self,
                          x: torch.Tensor,
                          gate_threshold: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dynamic policy: gate MLP decides exit vs continue at each exit.
        Uses sigmoid(gate(features)) >= gate_threshold as exit rule.
        """
        B = x.size(0)
        logits_out = torch.zeros(B, self.backbone.fc.out_features, device=x.device)
        exit_ids = torch.zeros(B, dtype=torch.long, device=x.device)

        exits_logits, final_logits = self.forward_with_exits(x)
        exits_probs = [F.softmax(l, dim=1) for l in exits_logits]

        gates = [self.gate1, self.gate2, self.gate3]
        depth_norms = [0.33, 0.66, 1.0]

        decided = torch.zeros(B, dtype=torch.bool, device=x.device)

        for i, (exit_logits, exit_probs, gate, dnorm) in enumerate(
                zip(exits_logits, exits_probs, gates, depth_norms)):
            feats = self._make_gate_features(exit_probs, exit_logits, dnorm)
            gate_logit = gate(feats)
            gate_prob = torch.sigmoid(gate_logit)  # [B]
            should_exit = (gate_prob >= gate_threshold) & (~decided)
            if should_exit.any():
                idx = should_exit.nonzero(as_tuple=False).squeeze(1)
                logits_out[idx] = exit_logits[idx]
                exit_ids[idx] = i
                decided[idx] = True

        remaining = (~decided)
        if remaining.any():
            idx = remaining.nonzero(as_tuple=False).squeeze(1)
            logits_out[idx] = final_logits[idx]
            exit_ids[idx] = 3

        return logits_out, exit_ids


# ---------------------------------------------------------------------------
#  Training routines
# ---------------------------------------------------------------------------

def train_backbone(model: EarlyExitResNet,
                   trainloader: DataLoader,
                   testloader: DataLoader,
                   device: torch.device,
                   epochs: int = 10,
                   lr: float = 0.1):
    print(f"[Backbone] Training for {epochs} epochs, lr={lr}")
    model.to(device)
    optimizer = torch.optim.SGD(model.backbone.parameters(),
                                lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(0.5 * epochs), int(0.75 * epochs)], gamma=0.1
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, targets in trainloader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model.forward_backbone_logits(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss = running_loss / len(trainloader.dataset)

        # Eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in testloader:
                images = images.to(device)
                targets = targets.to(device)
                logits = model.forward_backbone_logits(images)
                preds = logits.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        acc = correct / total * 100.0
        print(f"[Backbone][Epoch {epoch+1}/{epochs}] "
              f"loss={train_loss:.4f} test_acc={acc:.2f}%")


def freeze_backbone(model: EarlyExitResNet):
    for p in model.backbone.parameters():
        p.requires_grad = False


def train_exit_heads_distillation(model: EarlyExitResNet,
                                  trainloader: DataLoader,
                                  device: torch.device,
                                  epochs: int = 20,
                                  lr: float = 1e-3,
                                  temperature: float = 1.0):
    """
    Train exit heads by distilling from the backbone (unsupervised).
    """
    print(f"[Exits] Training exit heads for {epochs} epochs, lr={lr}, T={temperature}")
    model.to(device)
    model.train()
    freeze_backbone(model)

    params = list(model.exit1.parameters()) + \
             list(model.exit2.parameters()) + \
             list(model.exit3.parameters())

    optimizer = torch.optim.Adam(params, lr=lr)
    kldiv = nn.KLDivLoss(reduction="batchmean")

    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in trainloader:
            images = images.to(device)

            with torch.no_grad():
                teacher_logits = model.forward_backbone_logits(images)
                teacher_probs = F.softmax(teacher_logits / temperature, dim=1)

            exit_logits_list, _ = model.forward_with_exits(images)

            loss = 0.0
            for exit_logits in exit_logits_list:
                student_log_probs = F.log_softmax(exit_logits / temperature, dim=1)
                loss += kldiv(student_log_probs, teacher_probs)

            loss = loss / len(exit_logits_list)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(trainloader.dataset)
        print(f"[Exits][Epoch {epoch+1}/{epochs}] loss={train_loss:.4f}")


def collect_gate_training_data(model: EarlyExitResNet,
                               trainloader: DataLoader,
                               device: torch.device,
                               max_batches: int,
                               gate_label_conf: float) -> Dict[str, torch.Tensor]:
    """
    Collect features + pseudo labels for gate training.

    For each exit k and sample i:
      - exit_logits_k, exit_probs_k
      - teacher_logits (final backbone)

      label = 1 if:
        * exit_pred == teacher_pred
        * AND exit max_conf >= gate_label_conf
      else 0
    """
    print(f"[Gates] Collecting training data (up to {max_batches} batches)...")
    model.to(device)
    model.eval()
    freeze_backbone(model)

    data = {
        "feats1": [], "labels1": [],
        "feats2": [], "labels2": [],
        "feats3": [], "labels3": [],
    }

    with torch.no_grad():
        for b_idx, (images, _) in enumerate(trainloader):
            if b_idx >= max_batches:
                break
            images = images.to(device)

            exit_logits_list, final_logits = model.forward_with_exits(images)
            final_preds = final_logits.argmax(dim=1)

            depth_norms = [0.33, 0.66, 1.0]
            prefixes = ["1", "2", "3"]

            for exit_logits, dnorm, prefix in zip(
                exit_logits_list, depth_norms, prefixes
            ):
                exit_probs = F.softmax(exit_logits, dim=1)
                exit_preds = exit_logits.argmax(dim=1)
                max_conf, _ = exit_probs.max(dim=1)

                # label = 1 if exit is confident AND agrees with backbone
                labels = ((exit_preds == final_preds) &
                          (max_conf >= gate_label_conf)).float()

                feats = model._make_gate_features(exit_probs, exit_logits, dnorm)

                data[f"feats{prefix}"].append(feats.cpu())
                data[f"labels{prefix}"].append(labels.cpu())

    for k in list(data.keys()):
        if len(data[k]) > 0:
            data[k] = torch.cat(data[k], dim=0)
        else:
            data[k] = torch.empty(0)

    for exit_idx in [1, 2, 3]:
        feats = data[f"feats{exit_idx}"]
        labels = data[f"labels{exit_idx}"]
        if feats.numel() == 0:
            print(f"[Gates] Exit {exit_idx}: collected 0 samples")
        else:
            pos = labels.sum().item()
            neg = labels.numel() - pos
            print(f"[Gates] Exit {exit_idx}: collected {feats.size(0)} samples "
                  f"(pos={int(pos)}, neg={int(neg)})")

    return data


def train_gates(model: EarlyExitResNet,
                gate_data: Dict[str, torch.Tensor],
                device: torch.device,
                epochs: int = 5,
                lr: float = 1e-3,
                batch_size: int = 512):
    print(f"[Gates] Training gate MLPs for {epochs} epochs, lr={lr}")
    model.to(device)
    freeze_backbone(model)
    model.train()

    def _train_single_gate(gate: GateMLP,
                           feats: torch.Tensor,
                           labels: torch.Tensor,
                           name: str):
        if feats.size(0) == 0:
            print(f"[Gates] No data for {name}, skipping")
            return

        # Automatic pos_weight to counter class imbalance
        pos = labels.sum().item()
        total = labels.numel()
        neg = total - pos
        if pos == 0 or neg == 0:
            pos_weight = torch.tensor([1.0], device=device)
        else:
            pos_weight = torch.tensor([neg / max(pos, 1e-6)], device=device)

        dataset = torch.utils.data.TensorDataset(feats, labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(gate.parameters(), lr=lr)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for epoch in range(epochs):
            running_loss = 0.0
            for Xb, yb in loader:
                Xb = Xb.to(device)
                yb = yb.to(device)

                logits = gate(Xb)
                loss = bce(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * Xb.size(0)
            avg_loss = running_loss / len(loader.dataset)
            print(f"[Gates][{name}][Epoch {epoch+1}/{epochs}] loss={avg_loss:.4f}")

    _train_single_gate(model.gate1,
                       gate_data["feats1"],
                       gate_data["labels1"],
                       "gate1")
    _train_single_gate(model.gate2,
                       gate_data["feats2"],
                       gate_data["labels2"],
                       "gate2")
    _train_single_gate(model.gate3,
                       gate_data["feats3"],
                       gate_data["labels3"],
                       "gate3")


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvalResults:
    policy: str
    accuracy: float
    avg_latency_ms: float
    exit_distribution: List[float]
    description: str = ""


def evaluate_policy(model: EarlyExitResNet,
                    testloader: DataLoader,
                    device: torch.device,
                    policy: str,
                    tau: float = 0.9,
                    gate_threshold: float = 0.8) -> EvalResults:
    """
    Evaluate either:
      - policy == "static": static SelfXit threshold tau
      - policy == "dynamic": MLP gate policy (gate_threshold)
    """
    assert policy in ("static", "dynamic")
    model.to(device)
    model.eval()
    freeze_backbone(model)

    total_correct = 0
    total_samples = 0
    total_time = 0.0
    exit_counts = torch.zeros(4, dtype=torch.long)

    with torch.no_grad():
        for images, targets in testloader:
            images = images.to(device)
            targets = targets.to(device)

            t0 = time.time()
            if policy == "static":
                logits, exit_ids = model.inference_static(images, tau=tau)
            else:
                logits, exit_ids = model.inference_dynamic(
                    images, gate_threshold=gate_threshold
                )
            t1 = time.time()

            total_time += (t1 - t0)
            total_samples += targets.size(0)

            preds = logits.argmax(dim=1)
            total_correct += (preds == targets).sum().item()

            for e in range(4):
                exit_counts[e] += (exit_ids == e).sum().item()

    acc = total_correct / total_samples * 100.0
    avg_latency_ms = (total_time / len(testloader)) * 1000.0
    dist = (exit_counts.float() / total_samples * 100.0).tolist()

    desc = f"policy=static, tau={tau}" if policy == "static" \
           else f"policy=dynamic, gate_threshold={gate_threshold}"

    return EvalResults(
        policy=policy,
        accuracy=acc,
        avg_latency_ms=avg_latency_ms,
        exit_distribution=dist,
        description=desc
    )


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Improved SelfXit + Dynamic MLP Early Exits")

    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100"])
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "resnet50"])

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs_backbone", type=int, default=0,
                        help="If >0, train backbone from scratch.")
    parser.add_argument("--epochs_exits", type=int, default=20)
    parser.add_argument("--epochs_gates", type=int, default=5)

    parser.add_argument("--lr_backbone", type=float, default=0.1)
    parser.add_argument("--lr_exits", type=float, default=1e-3)
    parser.add_argument("--lr_gates", type=float, default=1e-3)

    parser.add_argument("--temperature", type=float, default=1.0,
                        help="KD temperature for exit training.")

    parser.add_argument("--tau", type=float, default=0.9,
                        help="Static confidence threshold for SelfXit-like policy.")
    parser.add_argument("--gate_threshold", type=float, default=0.8,
                        help="Decision threshold on gate sigmoid prob for exiting.")
    parser.add_argument("--gate_label_conf", type=float, default=0.8,
                        help="Min exit confidence for a sample to be labeled as a positive (good exit) for gate training.")

    parser.add_argument("--gate_max_batches", type=int, default=500,
                        help="Max number of train batches to use when collecting gate data.")

    parser.add_argument("--policy", type=str, default="both",
                        choices=["static", "dynamic", "both"],
                        help="Which policy to evaluate.")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip all training (you should load checkpoints manually).")

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    trainloader, testloader = get_cifar_loaders(args.dataset, args.batch_size)
    num_classes = 10 if args.dataset == "cifar10" else 100

    model = EarlyExitResNet(args.model, num_classes=num_classes)

    # ---------------- Training phase ----------------
    if not args.eval_only:
        if args.epochs_backbone > 0:
            train_backbone(model, trainloader, testloader,
                           device=device,
                           epochs=args.epochs_backbone,
                           lr=args.lr_backbone)
        else:
            print("[Backbone] Skipping backbone training (epochs_backbone=0). "
                  "You may want to load a pretrained checkpoint here.")

        train_exit_heads_distillation(model,
                                      trainloader,
                                      device=device,
                                      epochs=args.epochs_exits,
                                      lr=args.lr_exits,
                                      temperature=args.temperature)

        gate_data = collect_gate_training_data(model,
                                               trainloader,
                                               device=device,
                                               max_batches=args.gate_max_batches,
                                               gate_label_conf=args.gate_label_conf)

        train_gates(model,
                    gate_data,
                    device=device,
                    epochs=args.epochs_gates,
                    lr=args.lr_gates)

        # Optional: torch.save(model.state_dict(), "selfxit_improved.pth")
    else:
        print("[Eval-only] You should load a pretrained checkpoint here.")

    # ---------------- Evaluation phase ----------------
    results = []

    if args.policy in ("static", "both"):
        res_static = evaluate_policy(model,
                                     testloader,
                                     device=device,
                                     policy="static",
                                     tau=args.tau,
                                     gate_threshold=args.gate_threshold)
        results.append(res_static)

    if args.policy in ("dynamic", "both"):
        res_dyn = evaluate_policy(model,
                                  testloader,
                                  device=device,
                                  policy="dynamic",
                                  tau=args.tau,
                                  gate_threshold=args.gate_threshold)
        results.append(res_dyn)

    print("\n================= Evaluation Summary =================")
    for r in results:
        print(f"Policy: {r.policy}")
        print(f"  Description: {r.description}")
        print(f"  Accuracy: {r.accuracy:.2f}%")
        print(f"  Avg Latency per batch: {r.avg_latency_ms:.3f} ms")
        print(f"  Exit distribution (% of samples):")
        print(f"    Exit 0 (after layer2): {r.exit_distribution[0]:.2f}%")
        print(f"    Exit 1 (after layer3): {r.exit_distribution[1]:.2f}%")
        print(f"    Exit 2 (after layer4): {r.exit_distribution[2]:.2f}%")
        print(f"    Exit 3 (final backbone): {r.exit_distribution[3]:.2f}%")
        print("--------------------------------------------------")


if __name__ == "__main__":
    main()
