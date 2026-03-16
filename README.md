# SelfXit — Early-Exit Inference for ResNet

SelfXit is an early-exit inference framework that attaches lightweight exit heads at intermediate layers of a ResNet backbone. Easy samples exit early, skipping unnecessary computation; hard samples are processed through the full network. Trained on CIFAR-10, CIFAR-100, and TinyImageNet-200.

---

## How It Works

```
Input → [Layer1] → [Layer2] → Exit Head 1 → easy? ✓ exit
                             ↓ hard
                 → [Layer3] → Exit Head 2 → easy? ✓ exit
                             ↓ hard
                 → [Layer4] → Exit Head 3 → easy? ✓ exit
                             ↓ hard
                          → Final Head → output
```

Each exit head uses a confidence threshold (static policy) or a trained Gate MLP (dynamic policy) to decide whether to exit or continue.

---

## Features

- ResNet18 / ResNet50 backbone with CIFAR-style stem (no maxpool)
- Configurable exit placement (`--exits 1 2 3 4`)
- Optional conv block in exit heads (`--exit_conv`)
- Two inference policies: **static** (confidence threshold τ) and **dynamic** (Gate MLP)
- Joint end-to-end training (`--joint_training`)
- Curriculum temperature annealing (`--T_start / --T_end`)
- Entropy + confidence joint static gate (`--tau_entropy`)
- Budget-aware inference (`--compute_budget`)
- MAC / FLOPs profiling per exit point
- Per-class exit analysis (`--per_class`)
- Pareto frontier sweep (`--sweep`)
- Checkpoint save / resume (`--checkpoint_dir / --resume`)
- TensorBoard logging (`--log_dir`)
- TinyImageNet-200 support (`--dataset tinyimagenet`)

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

**Recommended full run:**
```bash
python selfxit_v2.py \
    --dataset cifar10 --model resnet18 \
    --joint_training --epochs_joint 30 \
    --T_start 4.0 --T_end 1.0 \
    --policy both --sweep --per_class \
    --checkpoint_dir ./ckpts --log_dir ./runs
```

**Quick test:**
```bash
python selfxit_v2.py --dataset cifar10 --model resnet18 --policy static
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `cifar10` | `cifar10`, `cifar100`, `tinyimagenet` |
| `--model` | `resnet18` | `resnet18`, `resnet50` |
| `--exits` | `1 2 3` | Which layers to attach exits |
| `--joint_training` | off | Train backbone + heads end-to-end |
| `--policy` | `static` | `static`, `dynamic`, `both` |
| `--tau` | `0.8` | Confidence threshold for static exit |
| `--compute_budget` | `1.0` | FLOPs budget fraction (0.0–1.0) |
| `--sweep` | off | Sweep τ values and plot Pareto frontier |
| `--per_class` | off | Print per-class exit distribution |

---

## Files

| File | Description |
|---|---|
| `selfxit_v2.py` | Main implementation (v2) with all features |
| `selfxit_unified copy.py` | Original v1 reference implementation |
| `Concepts.md` | Detailed explanation of every concept in the codebase |

---

## Architecture Details

**Gate MLP** — takes 5 features per sample and outputs a binary exit/continue decision:
- `max_conf` — max softmax confidence
- `entropy` — prediction entropy
- `logit_margin` — gap between top-2 logits
- `depth_norm` — normalized exit depth
- `logits_l2_norm` — L2 norm of logits

**Training stages:**
1. Backbone pre-training (SGD + cosine LR)
2. Exit head training via KL distillation from backbone
3. Gate MLP training (BCE + pos_weight for exit encouragement)

---

## Datasets

| Dataset | Classes | Train | Test | Resolution |
|---|---|---|---|---|
| CIFAR-10 | 10 | 50K | 10K | 32×32 |
| CIFAR-100 | 100 | 50K | 10K | 32×32 |
| TinyImageNet-200 | 200 | 100K | 10K | 64×64 |

CIFAR datasets are downloaded automatically. For TinyImageNet, download from [cs231n](http://cs231n.stanford.edu/tiny-imagenet-200.zip) and place in `data/tiny-imagenet-200/`.
