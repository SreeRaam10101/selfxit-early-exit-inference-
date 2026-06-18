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

Each exit head uses a confidence threshold (static policy) or a trained Gate MLP (dynamic policy) to decide whether to exit or continue. With **cascade inference**, later backbone layers are never executed once an exit fires — no wasted compute.

---

## Results (ResNet18, CIFAR-10, 30 epochs)

| Policy | Accuracy | Avg FLOPs |
|---|---|---|
| Static (τ=0.9) | 92.42% | 61.7% of full backbone |
| Dynamic (gate MLP) | 92.50% | 65.2% of full backbone |

**Systems optimisations on NVIDIA L4:**

| Feature | Gain |
|---|---|
| `--channels_last` (NHWC memory format) | **1.34× throughput**, P99 latency 73ms → 9.4ms |
| `--triton_gate` (fused gate-feature kernel) | **4.7× faster** gate computation |
| `--benchmark_single --cascade` (batch-1 early stopping) | **~2× lower P50** for early-exit samples |

---

## Features

**Inference**
- Two policies: **static** (confidence threshold τ) and **dynamic** (Gate MLP)
- **Cascade inference** — backbone stops at the fired exit (batch size 1); `--benchmark_single` reports cascade vs full-backbone P50/P95/P99 with a built-in correctness check
- **Temperature calibration** (`--calibrate`) — per-exit scalar T fitted via L-BFGS on validation set; saved as a JSON sidecar alongside the checkpoint and auto-loaded on `--resume`

**Training**
- Joint end-to-end training (`--joint_training`) with **depth-weighted exit loss** (earlier exits get stronger supervision)
- Curriculum temperature annealing (`--T_start / --T_end`)
- Entropy + confidence joint static gate (`--tau_entropy`)
- Budget-aware inference (`--compute_budget`)

**Analysis & plotting**
- Pareto frontier sweep (`--sweep`) with accuracy vs FLOPs plot (`--plot_dir`)
- Per-sample difficulty analysis — violin plot of logit margin by exit depth (`--difficulty_plot`)
- Per-class exit distribution (`--per_class`)
- MAC / FLOPs profiling per exit point

**Systems (CUDA)**
- `--channels_last` — NHWC memory format, eliminates internal layout-conversion overhead (~1.34× throughput on NVIDIA GPUs)
- `--triton_gate` — single fused Triton kernel replacing 5 separate PyTorch ops for gate feature computation (~4.7×  faster)

**Infrastructure**
- Configurable exit placement (`--exits 1 2 3 4`)
- Optional conv block in exit heads (`--exit_conv`)
- Checkpoint save / resume (`--checkpoint_dir / --resume`)
- TensorBoard logging (`--log_dir`)
- TinyImageNet-200 support (`--dataset tinyimagenet`)
- ResNet18 / ResNet50 backbone

---

## Installation

```bash
pip install -r requirements.txt
```

For the Triton gate kernel (CUDA only):
```bash
pip install triton>=2.0.0
```

---

## Usage

**Recommended full training run:**
```bash
python selfxit_v2.py \
    --dataset cifar10 --model resnet18 \
    --joint_training --epochs_joint 30 \
    --T_start 4.0 --T_end 1.0 \
    --policy both --sweep --per_class \
    --checkpoint_dir ./ckpts --log_dir ./runs
```

**Evaluate with all systems optimisations (NVIDIA GPU):**
```bash
python selfxit_v2.py \
    --dataset cifar10 --model resnet18 \
    --resume --checkpoint_dir ./ckpts \
    --eval_only --policy both \
    --channels_last --triton_gate --calibrate
```

**Batch-1 cascade latency benchmark:**
```bash
python selfxit_v2.py --dataset cifar10 --model resnet18 \
    --resume --checkpoint_dir ./ckpts \
    --eval_only --policy both --benchmark_single
```

**Triton kernel standalone benchmark:**
```bash
python selfxit_kernels.py --num_classes 10   # CIFAR-10
python selfxit_kernels.py --num_classes 100  # CIFAR-100
```

**Quick smoke test:**
```bash
python selfxit_v2.py --dataset cifar10 --model resnet18 --policy static
```

---

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `cifar10` | `cifar10`, `cifar100`, `tinyimagenet` |
| `--model` | `resnet18` | `resnet18`, `resnet50` |
| `--exits` | `1 2 3` | Which layers to attach exits |
| `--joint_training` | off | Train backbone + heads end-to-end |
| `--epochs_joint` | `30` | Epochs for joint training |
| `--T_start / --T_end` | `4.0 / 1.0` | Curriculum distillation temperature |
| `--policy` | `static` | `static`, `dynamic`, `both` |
| `--tau` | `0.8` | Confidence threshold for static exit |
| `--compute_budget` | `1.0` | FLOPs budget fraction (0.0–1.0) |
| `--calibrate` | off | Post-training per-exit temperature scaling |
| `--channels_last` | off | NHWC memory format (CUDA only) |
| `--triton_gate` | off | Fused Triton gate-feature kernel (CUDA only) |
| `--sweep` | off | Sweep τ and plot Pareto frontier |
| `--plot_dir` | None | Directory for Pareto + difficulty plots |
| `--per_class` | off | Per-class exit distribution |
| `--benchmark_single` | off | Batch-1 latency benchmark (cascade vs full backbone) |
| `--benchmark_n_runs` | `1000` | Number of runs for `--benchmark_single` |
| `--difficulty_plot` | off | Violin plot of logit margin vs exit depth |
| `--checkpoint_dir` | None | Where to save/load checkpoints |
| `--resume` | off | Load checkpoint before training/eval |
| `--log_dir` | None | TensorBoard log directory |

---

## Files

| File | Description |
|---|---|
| `selfxit_v2.py` | Main implementation with all features |
| `selfxit_v1.py` | Original v1 reference — do not modify |
| `selfxit_kernels.py` | Triton fused gate-feature kernel + correctness check + benchmark |
| `channels_last_bench.py` | NHWC vs NCHW throughput benchmark |
| `Concepts.md` | Detailed explanation of every design decision |

---

## Architecture Details

**Gate MLP** — takes 5 features per sample and outputs a binary exit/continue decision:

| Feature | Description |
|---|---|
| `max_conf` | Max softmax confidence (temperature-scaled when calibrated) |
| `entropy` | Prediction entropy |
| `logit_margin` | Gap between top-2 logits |
| `depth_norm` | Normalised exit depth |
| `logits_l2_norm` | L2 norm of logits |

**Training stages:**
1. Backbone pre-training (SGD + cosine LR)
2. Exit head training via KL distillation from backbone
3. Gate MLP training (BCE + pos_weight)

*(Or all stages jointly via `--joint_training`)*

---

## Datasets

| Dataset | Classes | Train | Test | Resolution |
|---|---|---|---|---|
| CIFAR-10 | 10 | 50K | 10K | 32×32 |
| CIFAR-100 | 100 | 50K | 10K | 32×32 |
| TinyImageNet-200 | 200 | 100K | 10K | 64×64 |

CIFAR datasets download automatically. For TinyImageNet, download from [cs231n](http://cs231n.stanford.edu/tiny-imagenet-200.zip) and place in `data/tiny-imagenet-200/`.
