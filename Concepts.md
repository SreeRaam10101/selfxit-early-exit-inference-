# SelfXit v2 — Complete Beginner's Guide to Every Concept

> This document explains **every single concept** used in `selfxit_v2.py`, starting from the absolute basics (Python, variables, functions) all the way up to the deep learning ideas (ResNets, early exits, knowledge distillation). Nothing is skipped.

---

## Table of Contents

1. [Python Basics Used in the File](#1-python-basics)
2. [Imports & Libraries](#2-imports--libraries)
3. [What is a Neural Network?](#3-what-is-a-neural-network)
4. [What is Deep Learning & PyTorch?](#4-pytorch--deep-learning)
5. [Tensors — PyTorchs Core Data Structure](#5-tensors)
6. [Neural Network Layers](#6-neural-network-layers)
7. [Training a Neural Network](#7-training-a-neural-network)
8. [Datasets & DataLoaders](#8-datasets--dataloaders)
9. [ResNet — The Backbone Model](#9-resnet--the-backbone-model)
10. [Early Exit — The Key Idea of This Project](#10-early-exit)
11. [Exit Heads (ExitHead class)](#11-exit-heads)
12. [Gate MLP (GateMLP class)](#12-gate-mlp)
13. [EarlyExitResNet — Putting It All Together](#13-earlyexitresnet-class)
14. [Training Strategies](#14-training-strategies)
15. [Knowledge Distillation](#15-knowledge-distillation)
16. [Inference Policies — Static vs Dynamic](#16-inference-policies)
17. [Shannon Entropy](#17-shannon-entropy)
18. [FLOPs / MACs Profiling](#18-flops--macs-profiling)
19. [Pareto Frontier Sweep](#19-pareto-frontier-sweep)
20. [Checkpointing](#20-checkpointing)
21. [TensorBoard Logging](#21-tensorboard-logging)
22. [argparse — Command Line Arguments](#22-argparse)
23. [Dataclasses](#23-dataclasses)
24. [Type Hints](#24-type-hints)
25. [Binary Search for Budget-Aware Inference](#25-binary-search)
26. [The Big Picture — How Everything Fits Together](#26-the-big-picture)

---

## 1. Python Basics

Before diving into the AI concepts, let's cover the basic Python patterns used everywhere in this file.

### Variables and Data Types

```python
epochs = 30          # integer — whole number
lr = 0.05            # float — decimal number
name = "resnet18"    # string — text
flag = True          # boolean — True or False
```

### Functions

A function is a reusable block of code. You define it once and call it many times.

```python
def add(a, b):       # "def" defines a function; a and b are parameters
    return a + b     # "return" sends back the result

result = add(3, 5)   # calling the function → result is 8
```

In this file, every major operation is wrapped in a function, e.g. `train_backbone(...)`, `evaluate_policy(...)`.

### Lists and Dictionaries

```python
# List — ordered collection
exit_layers = [2, 3, 4]   # exits are at layer 2, 3, and 4

# Dictionary — key → value pairs
feat_dims = {"resnet18": {1: 64, 2: 128}}   # look up channel width by model name + layer
```

### Loops

```python
for epoch in range(30):      # repeat 30 times; epoch goes 0, 1, 2, ... 29
    print(epoch)

for images, targets in trainloader:   # unpack each batch into images and labels
    ...
```

### Classes and Objects

Python classes are blueprints for creating objects that bundle data and behaviour together.

```python
class Dog:
    def __init__(self, name):   # __init__ runs when you create a new Dog
        self.name = name        # self.name stores the name on this specific object

    def bark(self):
        print(f"{self.name} says woof!")

rex = Dog("Rex")   # create an instance (object) of Dog
rex.bark()         # calls the method → "Rex says woof!"
```

Every neural network component in this file (`ExitHead`, `GateMLP`, `EarlyExitResNet`, etc.) is a Python class.

### `super().__init__()`

When a class inherits from another, `super().__init__()` calls the parent's setup code. In PyTorch, every neural network class must inherit from `nn.Module` and call `super().__init__()` first.

```python
class MyLayer(nn.Module):
    def __init__(self):
        super().__init__()   # REQUIRED — sets up PyTorch internals
        self.fc = nn.Linear(10, 5)
```

### `if __name__ == "__main__":`

```python
if __name__ == "__main__":
    main()
```

This line means: "Only run `main()` if this file is being executed directly (not imported by another file)." It is the entry point of the entire program.

---

## 2. Imports & Libraries

At the top of the file you see:

```python
import argparse
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T
```

### What each one does

| Import | Purpose |
|--------|---------|
| `argparse` | Parse command-line arguments (like `--epochs 30`) |
| `os` | Work with files and directories (create folders, check paths) |
| `shutil` | Copy/move files (used to reorganise TinyImageNet dataset) |
| `time` | Measure how long inference takes |
| `dataclasses` | Clean way to create simple data-holder classes |
| `typing` | Declare what types of data a function expects/returns |
| `torch` | PyTorch core — tensors, automatic differentiation |
| `torch.nn` | Neural network building blocks (layers, loss functions) |
| `torch.nn.functional` | Stateless operations like softmax, relu, log_softmax |
| `torch.utils.data.DataLoader` | Efficiently feed batches of data to the model |
| `torchvision` | Pre-built datasets (CIFAR-10, etc.) and model architectures |
| `torchvision.transforms` | Image preprocessing (crop, flip, normalize) |

---

## 3. What is a Neural Network?

A neural network is a mathematical function that maps inputs (e.g., a photo) to outputs (e.g., "this is a cat").

It is inspired loosely by the brain. It has many **layers**. Each layer applies a mathematical transformation to its input and passes the result to the next layer. The network **learns** these transformations from examples — we show it millions of labelled images and adjust the numbers (called **weights** or **parameters**) until it makes correct predictions.

```
Input image (pixels)
      ↓
  Layer 1  →  detects edges
      ↓
  Layer 2  →  detects shapes (circles, corners)
      ↓
  Layer 3  →  detects parts (eyes, wheels)
      ↓
  Output   →  "cat" / "dog" / "car" (classification)
```

### Classification

In this project, the network is doing **image classification** — given an image, output one label from a fixed set of classes (10 for CIFAR-10, 100 for CIFAR-100, 200 for TinyImageNet).

### Logits and Probabilities

The final layer outputs raw numbers called **logits** — one number per class. They can be positive or negative. To turn them into probabilities (0 to 1, summing to 1), we apply **softmax**:

```
logits:       [2.1,  0.5,  -1.3]    ← raw scores for 3 classes
              ↓ softmax
probabilities: [0.76, 0.16,  0.08]  ← proper probabilities, sum to 1
```

The class with the highest probability is the prediction.

---

## 4. PyTorch & Deep Learning

**PyTorch** is a Python library for building and training neural networks. Think of it as a toolkit with:

- **Tensors** — multi-dimensional arrays (like numpy, but GPU-compatible and differentiable)
- **Autograd** — automatically computes gradients for training
- **nn.Module** — base class for all neural network components
- **Optimizers** — algorithms (SGD, Adam) that update network weights

### GPU Acceleration

Training on images is very slow on a CPU. GPUs (graphics cards) can run thousands of simple operations in parallel. This code detects the best available device:

```python
def get_device() -> torch.device:
    if torch.backends.mps.is_available():   # Apple Silicon GPU
        return torch.device("mps")
    if torch.cuda.is_available():           # NVIDIA GPU
        return torch.device("cuda")
    return torch.device("cpu")             # fallback: CPU
```

You move both the model and data to this device so everything runs on the same hardware.

---

## 5. Tensors

A **tensor** is a multi-dimensional array — the fundamental unit of data in PyTorch.

```
Scalar (0D):  42
Vector (1D):  [1, 2, 3]
Matrix (2D):  [[1, 2], [3, 4]]
3D tensor:    shape (batch, height, width)  — e.g. grayscale images
4D tensor:    shape (batch, channels, height, width)  — e.g. RGB images
```

In this project, images are 4D tensors: `(batch_size, 3, 32, 32)` meaning a batch of images, each with 3 colour channels (R, G, B), each 32×32 pixels.

### Common tensor operations used in the code

```python
x.size(0)        # get the batch dimension (number of images)
x.max(1)         # get the maximum value along dimension 1 (across classes)
x.argmax(1)      # index of the maximum value (the predicted class)
x.flatten(1)     # collapse all dimensions from 1 onward into a single vector
torch.zeros(B, num_cls)   # create a tensor filled with zeros
torch.full((B,), val)     # create a tensor filled with the same value
```

### `.to(device)`

Moving tensors to GPU:
```python
images = images.to(device)   # send image data to GPU
model = model.to(device)     # send model weights to GPU
```

Both must be on the same device, otherwise PyTorch throws an error.

---

## 6. Neural Network Layers

Every neural network is built from **layers**. Each layer transforms its input tensor to an output tensor. Here are the specific layers used in this project:

### `nn.Conv2d` — Convolutional Layer

This is the most important layer for image processing. It applies a small **filter** (kernel) that slides across the image, detecting local patterns like edges, textures, and shapes.

```
Input: (batch, in_channels, H, W)
Output: (batch, out_channels, new_H, new_W)
```

Parameters:
- `in_channels` — number of input channels (3 for RGB)
- `out_channels` — number of filters to learn
- `kernel_size` — size of the filter (e.g., 3 = 3×3 filter)
- `padding` — adds zeros around the border to preserve spatial size

```python
nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
# takes a 3-channel image, learns 64 different filters, each 3×3
```

### `nn.BatchNorm2d` — Batch Normalisation

After each convolutional layer, outputs can have very different scales, which makes training unstable. BatchNorm normalises (standardises) the output to have mean=0 and variance=1, then lets the network learn a small scale and shift. This makes training much faster and more stable.

```python
nn.BatchNorm2d(64)  # normalise across 64 channels
```

### `nn.ReLU` — Activation Function

Neural networks without non-linear functions are just linear algebra — they cannot learn complex patterns. **ReLU** (Rectified Linear Unit) adds non-linearity in the simplest way possible:

```
ReLU(x) = max(0, x)
         = x if x > 0
         = 0 if x ≤ 0
```

The `inplace=True` argument modifies the tensor in place (saves memory).

### `nn.Linear` — Fully Connected Layer

Every neuron connects to every neuron in the previous layer. Used at the end of the network to map from feature vectors to class scores.

```python
nn.Linear(512, 100)   # takes 512-dimensional vector, outputs 100-dimensional vector
# 512 × 100 = 51,200 learnable weights
```

### `nn.AdaptiveAvgPool2d((1, 1))` — Global Average Pooling

Takes a feature map of any spatial size (e.g., 8×8 or 4×4) and averages it down to a single 1×1 value per channel. This is how spatial features are compressed into a flat vector before the final classifier.

```
Input: (batch, 512, 8, 8)   → 512 channels, 8×8 spatial
         ↓ AdaptiveAvgPool2d((1, 1))
Output: (batch, 512, 1, 1)  → 512 numbers, one per channel
         ↓ flatten
Output: (batch, 512)        → flat vector, ready for nn.Linear
```

### `nn.MaxPool2d` — Max Pooling

Takes the maximum value in each small region, reducing spatial size (e.g., halving from 32×32 to 16×16). Used in the ResNet stem for large images. For CIFAR (32×32 images), the code replaces it with `nn.Identity()` to avoid making images too small too quickly.

### `nn.Identity()`

Does absolutely nothing — it just passes its input through unchanged. It is used here as a placeholder: the CIFAR-adapted ResNet replaces the 7×7 convolution and maxpool with a 3×3 convolution and Identity, preserving the small 32×32 image resolution.

### `nn.Dropout`

During training, randomly sets a fraction of neuron outputs to zero. This is **regularisation** — it forces the network to not rely too heavily on any single neuron, reducing overfitting (memorising training data instead of learning general patterns).

```python
nn.Dropout(0.1)   # randomly zero 10% of values during training; does nothing during eval
```

### `nn.Sequential`

Chains layers together in order — the output of one feeds directly into the next.

```python
nn.Sequential(
    nn.Conv2d(64, 64, 3, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
)
```

### `nn.ModuleList`

A Python list that properly registers its contents as part of the model (so weights are tracked, saved, moved to GPU, etc.).

```python
self.exit_heads = nn.ModuleList([ExitHead(...) for l in exit_layers])
# a list of exit head modules, one per early exit point
```

### `nn.KLDivLoss` — Kullback-Leibler Divergence Loss

Measures how different two probability distributions are. Used in knowledge distillation to train exit heads to mimic the teacher's output distribution (not just predict the hard label).

### `nn.BCEWithLogitsLoss` — Binary Cross-Entropy with Logits

Used to train the gate network (a binary classifier: "should we exit here? yes/no"). Combines a sigmoid activation and binary cross-entropy loss in one numerically stable operation.

---

## 7. Training a Neural Network

Training is the process of adjusting weights so the network makes better predictions.

### The Training Loop

```python
for epoch in range(num_epochs):          # loop over the whole dataset many times
    for images, targets in trainloader:  # loop over mini-batches
        optimizer.zero_grad()            # reset gradients from previous step
        predictions = model(images)      # forward pass: compute predictions
        loss = criterion(predictions, targets)  # measure how wrong we are
        loss.backward()                  # backward pass: compute gradients
        optimizer.step()                 # update weights using gradients
```

Each pass through the entire training set is called an **epoch**.

### Forward Pass

The image flows through the network layer by layer, producing predictions. This is just function evaluation.

### Loss Function

The **loss** (also called error or cost) is a number that measures how wrong the predictions are. Lower is better. 

**Cross-Entropy Loss** (`nn.CrossEntropyLoss`) is the standard loss for classification:
- It is large when the model is very confident about the wrong class
- It is small when the model is very confident about the right class

### Backward Pass — Backpropagation

PyTorch automatically computes how much each weight contributed to the loss using the **chain rule of calculus**. This produces **gradients** — the direction to nudge each weight to reduce the loss.

```python
loss.backward()   # PyTorch walks backward through all operations and computes gradients
```

### Optimizers

An optimizer uses the gradients to update the weights.

**SGD (Stochastic Gradient Descent):**
```python
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
```
- `lr` (learning rate): how big each step is. Too big → overshoots. Too small → too slow.
- `momentum`: like rolling a ball downhill — uses previous gradient direction to smooth updates
- `weight_decay`: L2 regularisation — gently penalises large weights to prevent overfitting

**Adam:**
```python
optimizer = torch.optim.Adam(params, lr=1e-3)
```
Adam adapts the learning rate for each parameter individually, often working well out of the box.

### Learning Rate Scheduler

The learning rate is typically reduced over training. Here, `MultiStepLR` divides it by 10 at 50% and 75% of training:

```python
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[15, 22], gamma=0.1)
# at epoch 15: lr = lr × 0.1
# at epoch 22: lr = lr × 0.1 × 0.1
scheduler.step()   # call once per epoch
```

### `model.train()` vs `model.eval()`

```python
model.train()   # training mode: Dropout randomly zeros neurons; BatchNorm uses batch statistics
model.eval()    # evaluation mode: Dropout disabled; BatchNorm uses running statistics
```

Always switch to `model.eval()` before measuring accuracy.

### `torch.no_grad()`

```python
with torch.no_grad():
    predictions = model(images)
```

During evaluation, we don't need gradients (we're not updating weights). `no_grad()` skips gradient computation, saving memory and making it faster.

### Freezing Parameters

```python
for p in model.backbone.parameters():
    p.requires_grad_(False)   # do not compute gradients for backbone weights
```

When training the exit heads, we freeze the backbone so only the exit heads are updated. This is called **transfer learning** or **sequential training**.

---

## 8. Datasets & DataLoaders

### CIFAR-10 and CIFAR-100

Standard image classification benchmarks:
- **CIFAR-10**: 60,000 images, 10 classes (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck)
- **CIFAR-100**: 60,000 images, 100 fine-grained classes
- Image size: 32×32 pixels, RGB

### TinyImageNet-200

A smaller version of ImageNet:
- 100,000 training images, 10,000 validation images, 200 classes
- Image size: 64×64 pixels

### Image Transforms (Data Augmentation)

Before feeding images to the network, we apply transformations:

```python
T.Compose([
    T.RandomCrop(32, padding=4),    # randomly crop to 32×32 (first pad with 4 pixels)
    T.RandomHorizontalFlip(),        # randomly flip left/right (50% chance)
    T.ToTensor(),                    # convert PIL image (0–255) → tensor (0.0–1.0)
    T.Normalize(mean, std),          # standardise: (pixel - mean) / std
])
```

**Why augment?** Each epoch, the network sees slightly different versions of each image. This effectively multiplies the dataset and helps the network generalise — it learns "a cat is a cat whether it's on the left or right".

**Normalisation** with dataset-specific mean and std makes all pixel values have similar scale, which helps training converge faster.

### DataLoader

```python
DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
```

- `batch_size`: how many images to process at once (more = faster, but needs more GPU memory)
- `shuffle=True`: randomise order each epoch (prevents the network from memorising batch order)
- `num_workers`: background CPU threads that pre-load batches while GPU is busy
- `pin_memory=True`: page-locks CPU memory for faster GPU transfer

---

## 9. ResNet — The Backbone Model

**ResNet** (Residual Network) is the standard convolutional neural network used as the backbone in this project. It was introduced in 2015 and won the ImageNet competition that year.

### The Vanishing Gradient Problem

Very deep networks (many layers) were difficult to train: gradients become tiny as they travel backwards, so early layers barely learn. This is the **vanishing gradient problem**.

### Residual Connections (Skip Connections)

ResNet's key innovation: instead of just learning `F(x)`, each block learns `F(x) + x` (the residual). The original input `x` is added back (the "shortcut connection").

```
Input x
  ↓
Conv → BN → ReLU → Conv → BN
  ↓                       ↓
  +──────────────────────+     (add original x back)
  ↓
ReLU
↓
Output: F(x) + x
```

This means gradients can flow directly through the shortcut path all the way back to early layers, enabling training of networks 50–150+ layers deep.

### ResNet Architecture

```
Input image (3 × 32 × 32 for CIFAR)
    ↓
Stem: Conv1 → BN → ReLU → (MaxPool, or Identity for CIFAR)
    ↓
Layer 1:  stack of residual blocks, 64 channels
    ↓
Layer 2:  stack of residual blocks, 128 channels
    ↓
Layer 3:  stack of residual blocks, 256 channels
    ↓
Layer 4:  stack of residual blocks, 512 channels
    ↓
Global Average Pool → Flatten
    ↓
Fully Connected → num_classes logits
```

### ResNet18 vs ResNet50

| Model | Depth | Channels (L1/L2/L3/L4) | Parameters |
|-------|-------|------------------------|------------|
| ResNet18 | 18 layers | 64/128/256/512 | ~11M |
| ResNet50 | 50 layers | 256/512/1024/2048 | ~25M |

ResNet50 uses **bottleneck blocks** (1×1 conv → 3×3 conv → 1×1 conv) for efficiency, giving it more representational power with manageable compute.

### CIFAR Stem Adaptation

The standard ResNet stem uses a large 7×7 convolution followed by MaxPool, designed for ImageNet's 224×224 images. For tiny 32×32 CIFAR images, this would reduce size too aggressively. So:

```python
if cifar_stem:
    backbone.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)  # 3×3 instead of 7×7
    backbone.maxpool = nn.Identity()  # skip the max pool
```

---

## 10. Early Exit

This is the central idea of the entire project.

### The Problem: All Inputs Are Not Equal

A standard neural network processes every image through all its layers, taking the same amount of compute regardless of how easy or hard the image is. A photo of a bright red fire truck is easy — even a shallow network can recognise it. A blurry, partially-occluded cat in low light is hard — you need deep reasoning.

**Why make easy images go through the whole network? That's wasteful.**

### The Solution: Multiple Exit Points

Early exit adds classifier heads at intermediate layers of the network. When an intermediate head is confident enough, it outputs a prediction immediately — we **exit early** and skip the remaining layers.

```
Input image
    ↓
Layer 1 → [Exit Head 1] → if confident: output prediction ✓
    ↓ (not confident)
Layer 2 → [Exit Head 2] → if confident: output prediction ✓
    ↓ (not confident)
Layer 3 → [Exit Head 3] → if confident: output prediction ✓
    ↓ (not confident)
Layer 4 → Final Classifier → output prediction ✓
```

### Benefits

- **Faster average inference** — easy images exit at layer 2, hard ones go all the way
- **Lower energy/compute** — proportional savings when most inputs are easy
- **Same accuracy for easy inputs** — the early head is correct for these anyway
- **Graceful degradation** — under a compute budget, accuracy degrades gradually

### The Concept of "Confidence"

An exit head outputs a probability distribution over classes. If `max(probabilities) >= 0.9`, the network is very confident (90%) about its top prediction → exit. If `max(probabilities) = 0.4`, the network is unsure → continue deeper.

This threshold is controlled by the `tau` parameter.

---

## 11. Exit Heads

```python
class ExitHead(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_dim=512, use_conv=False):
        super().__init__()
        if use_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
            )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1  = nn.Linear(in_channels, hidden_dim)
        self.drop = nn.Dropout(0.1)
        self.fc2  = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        if self.use_conv:
            x = self.conv(x)              # optional: extra conv to learn spatial features
        x = self.pool(x).flatten(1)       # (batch, C, H, W) → (batch, C)
        return self.fc2(self.drop(F.relu(self.fc1(x))))  # classify
```

**What it does:** Takes the feature map from an intermediate layer, pools it to a vector, and classifies it into `num_classes`. It is a small classifier attached to the side of the main network.

**`forward` method:** This is called when you run `head(x)` or `model(x)`. PyTorch calls `forward` automatically.

**`F.relu(...)`:** The functional form of ReLU — equivalent to `nn.ReLU()` but applied inline without being a stored module.

---

## 12. Gate MLP

```python
class GateMLP(nn.Module):
    def __init__(self, in_dim=5, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),   # output: single number (logit for "exit yes/no")
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)   # remove the trailing size-1 dimension
```

### What is a Gate?

The **gate** is a small neural network that decides, at each exit point, whether to exit or continue processing. Instead of a fixed threshold rule ("exit if confidence > 0.9"), the gate **learns** a decision function from data.

### The 5 Input Features

The gate receives 5 numbers describing the exit head's output at this point:

1. **max_conf** — the highest probability among all classes (how confident the exit head is)
2. **entropy** — a measure of uncertainty (see Section 17)
3. **logit_margin** — the difference between the top-2 class scores (large margin = confident)
4. **depth_norm** — normalised depth (e.g., 0.33, 0.67, 1.0 for 3 exits) — tells the gate how deep in the network we are
5. **logits_l2_norm** — the overall magnitude of the logit vector

### Why is this better than a fixed threshold?

A fixed threshold `tau = 0.9` treats confidence as the only signal. The gate can use **all 5 features** and learn a non-linear decision boundary. For example: "exit at layer 2 if confidence is high AND entropy is low AND we are only at depth 0.33" vs "exit at layer 4 if confidence is moderate but the logit margin is large."

---

## 13. EarlyExitResNet Class

This is the main model class that assembles everything:

```python
class EarlyExitResNet(nn.Module):
    def __init__(self, model_name, num_classes, exit_layers=(2,3,4), ...):
        super().__init__()
        # 1. Load the base ResNet (with no pre-trained weights)
        backbone = torchvision.models.resnet18(weights=None)
        # 2. Adapt for CIFAR (small images)
        if cifar_stem:
            backbone.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            backbone.maxpool = nn.Identity()
        # 3. Replace final FC layer to match number of classes
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.backbone = backbone
        # 4. Attach one ExitHead per exit layer
        self.exit_heads = nn.ModuleList([ExitHead(feat_dims[l], num_classes) for l in exit_layers])
        # 5. Attach one GateMLP per exit layer
        self.gates = nn.ModuleList([GateMLP() for _ in exit_layers])
```

### `_backbone_features` method

Manually runs each section of the backbone step by step, capturing the intermediate feature maps:

```python
x = conv1 → bn1 → relu → maxpool    # stem
x = layer1(x)  → save if layer 1 is an exit layer
x = layer2(x)  → save if layer 2 is an exit layer
x = layer3(x)  → save if layer 3 is an exit layer
x = layer4(x)  → save if layer 4 is an exit layer
final_logits = avgpool → flatten → fc
```

### `forward_with_exits` method

Returns both the list of exit logits and the final backbone logits — used during training and evaluation.

### `getattr(self.backbone, f"layer{layer_idx}")`

`getattr(obj, name)` retrieves an attribute by name as a string. `f"layer{layer_idx}"` is an f-string that creates `"layer1"`, `"layer2"`, etc. So this dynamically accesses `backbone.layer1`, `backbone.layer2`, etc. in a loop.

---

## 14. Training Strategies

The project offers two training approaches:

### Sequential Training (3-phase)

1. **Phase 1 — Backbone:** Train the full ResNet normally with CrossEntropyLoss. Exit heads are ignored.
2. **Phase 2 — Exit Heads:** Freeze backbone. Train exit heads using knowledge distillation (see Section 15).
3. **Phase 3 — Gates:** Collect data from the trained model, then train GateMLPs as binary classifiers.

### Joint Training

Train backbone + all exit heads simultaneously. The total loss is:

```
L_total = L_backbone + weight × (L_exit1 + L_exit2 + L_exit3)
```

All parts are updated in every training step. This is simpler but requires careful tuning of `exit_loss_weight`.

---

## 15. Knowledge Distillation

This is one of the most important techniques in the project.

### The Idea

The main backbone (`teacher`) has been fully trained and makes good predictions. We want the exit heads (`students`) to imitate the teacher.

Instead of training students to predict the hard label ("this is class 7"), we train them to match the teacher's full probability distribution:

```
Teacher output:  [0.01, 0.02, 0.85, 0.03, 0.09]   (class 2 = 85%)
Student should   [0.01, 0.02, 0.85, 0.03, 0.09]   (match this soft distribution)
vs hard label:   [   0,    0,    1,    0,    0]   (only class 2 = 100%)
```

**Why is the soft distribution better?** It encodes similarity between classes. "Looks a bit like class 4 too" is useful information that the hard label throws away.

### Temperature

To make the teacher's distribution softer (more informative), we divide logits by a **temperature T** before softmax:

```
Soft probs = softmax(logits / T)
```

- T = 1: normal softmax
- T = 4: much softer distribution — small classes get higher probabilities, encoding more information

### Curriculum Temperature Annealing

Start with a high temperature (T=4.0 → very soft, rich information) and gradually reduce it to T=1.0 (normal) over training:

```python
T = T_start + (T_end - T_start) * epoch / max(epochs - 1, 1)
# epoch 0:  T = 4.0  (very soft teacher)
# epoch 10: T = 2.5  (moderately soft)
# epoch 19: T = 1.0  (hard)
```

Early in training, soft distributions help the student learn the shape of the output space. Later, hard targets fine-tune for exact correctness.

### KL Divergence Loss

The loss that measures how different the student's distribution is from the teacher's:

```python
kldiv = nn.KLDivLoss(reduction="batchmean")
loss = kldiv(F.log_softmax(student_logits / T, 1), teacher_probs)
```

Note: `KLDivLoss` expects log-probabilities as input and probabilities as target, hence `log_softmax` for the student.

---

## 16. Inference Policies

After training, how do we actually decide when to exit?

### Static Policy (`inference_static`)

A simple rule: exit at the first exit where `max(softmax(logits)) >= tau`.

```python
probs = F.softmax(exit_logits, dim=1)
max_conf, _ = probs.max(1)        # highest probability across all classes, per sample
condition = (max_conf >= tau)     # True for samples confident enough to exit
```

Optional: also require `entropy <= tau_entropy` (see Section 17).

**Key implementation detail:** The code processes a whole batch simultaneously. A boolean mask `decided` tracks which samples have already exited, and `~decided` (logical NOT) means "not yet decided." Only undecided samples are eligible to exit at each checkpoint.

```python
decided = torch.zeros(B, dtype=torch.bool)   # all False initially
for i, exit_logits in enumerate(exit_logits_list):
    probs = F.softmax(exit_logits, 1)
    max_conf, _ = probs.max(1)
    condition = (max_conf >= tau) & (~decided)   # confident AND not yet exited
    idx = condition.nonzero(as_tuple=False).squeeze(1)   # which samples exit here
    logits_out[idx] = exit_logits[idx]
    exit_ids[idx] = i
    decided[idx] = True
```

### Dynamic Policy (`inference_dynamic`)

Uses the trained GateMLP to decide. The gate takes 5 features and outputs a probability of "should exit here." Exit if `sigmoid(gate_output) >= gate_threshold`.

`torch.sigmoid(x) = 1 / (1 + e^(-x))` — squashes any real number to (0, 1). The gate outputs a raw logit; sigmoid converts it to an "exit probability."

---

## 17. Shannon Entropy

**Entropy** measures uncertainty or randomness. In information theory, Shannon entropy of a probability distribution p is:

```
H(p) = -∑ p_i × log(p_i)
```

Applied to class probabilities:
- **Low entropy** → peaked distribution → model is certain (e.g., [0.95, 0.02, 0.03])
- **High entropy** → flat distribution → model is uncertain (e.g., [0.33, 0.33, 0.34])

In code:
```python
def entropy_from_probs(probs, eps=1e-8):
    return -(probs * (probs + eps).log()).sum(dim=1)
```

The `eps=1e-8` prevents `log(0)` which would be `-infinity`.

### Used in Two Ways

1. **Static gate:** Optionally require `entropy <= tau_entropy` in addition to high confidence
2. **Dynamic gate features:** One of the 5 input features to the GateMLP

---

## 18. FLOPs / MACs Profiling

**FLOP** (Floating Point Operation) and **MAC** (Multiply-Accumulate operation) are units for measuring compute.

For a convolutional layer computing output pixel at position (i, j):
```
MACs = out_channels × (in_channels / groups) × kernel_h × kernel_w
```
Across all output pixels: multiply by `out_h × out_w`.

### Hooks — a PyTorch Mechanism

A **hook** is a callback function that PyTorch calls automatically at certain points during the forward pass.

```python
def conv_hook(module, inp, output):
    # Count MACs when this conv layer runs
    macs[0] += ...

hook = layer.register_forward_hook(conv_hook)
# Now every time layer(x) is called, conv_hook also runs
hook.remove()   # clean up when done
```

The `macs = [0]` trick (using a list instead of a plain integer) is because Python closures can't rebind a plain variable from an outer scope, but they can mutate a mutable container like a list.

### `MACProfile` Dataclass

```python
@dataclass
class MACProfile:
    stem_macs: int
    layer_macs: Dict[int, int]   # layer 1 → MACs, layer 2 → MACs, ...
    fc_macs: int
    head_macs: Dict[int, int]
```

Stores MAC counts for each segment. Used to compute the **FLOPs fraction** for each exit point — what fraction of the full backbone's compute was used before exiting.

---

## 19. Pareto Frontier Sweep

### The Accuracy-Compute Trade-off

By adjusting `tau` (static) or `gate_threshold` (dynamic), you can trade off:
- **High threshold** → harder to exit → more compute used → higher accuracy
- **Low threshold** → easy to exit → less compute → lower accuracy

### Pareto Frontier

A set of operating points where you cannot improve one metric without worsening the other. The **Pareto sweep** runs evaluation at many threshold values and prints the accuracy vs. FLOPs table:

```
tau     acc(%)   flops(%)  lat(ms)
0.500   72.10    45.2      3.21
0.700   78.50    58.7      4.05
0.900   82.30    74.1      5.12
0.990   83.10    98.2      6.88
```

This helps you pick the right operating point for your deployment constraints.

---

## 20. Checkpointing

Saving and loading model weights so training can be interrupted and resumed.

### `state_dict`

A Python dictionary mapping parameter names to their tensors. It captures the entire learned state of the model.

```python
# Save
torch.save(model.state_dict(), "model.pt")

# Load
model.load_state_dict(torch.load("model.pt", map_location=device))
```

`map_location=device` ensures that weights saved on a GPU can be loaded on a CPU, and vice versa.

---

## 21. TensorBoard Logging

TensorBoard is a visualisation tool that shows training curves (loss over time, accuracy over time) in a web browser.

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="./logs")
writer.add_scalar("loss/train", loss_value, epoch)
```

The `Logger` class in this file wraps TensorBoard with a print fallback — if TensorBoard isn't installed, it just prints to console. The `try/except ImportError` pattern handles this gracefully:

```python
try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False   # TensorBoard not installed — that's fine
```

---

## 22. argparse

**argparse** is the standard Python library for building command-line interfaces. It lets users pass options when running the script:

```bash
python selfxit_v2.py --dataset cifar100 --model resnet18 --epochs_backbone 5
```

```python
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="cifar10", choices=["cifar10","cifar100"])
parser.add_argument("--epochs_backbone", type=int, default=0)
parser.add_argument("--sweep", action="store_true")   # flag: present=True, absent=False
args = parser.parse_args()

# Then use:
args.dataset         # "cifar100"
args.epochs_backbone # 5
args.sweep           # True or False
```

`nargs="+"` means "one or more values":
```bash
--exits 2 3 4    → args.exits = [2, 3, 4]
```

---

## 23. Dataclasses

A **dataclass** is a convenient way to create classes that mainly hold data (like a struct in other languages).

```python
@dataclass
class EvalResult:
    policy:             str
    accuracy:           float
    avg_latency_ms:     float
    exit_distribution:  List[float]
    avg_flops_fraction: float = 0.0   # default value
    per_class_exits:    Dict[int, List[float]] = field(default_factory=dict)
```

The `@dataclass` decorator automatically generates `__init__`, `__repr__`, etc. You get a clean data container without writing boilerplate.

`field(default_factory=dict)` is required for mutable defaults — you can't use `= {}` directly in dataclasses (it would be shared across all instances).

---

## 24. Type Hints

Python is dynamically typed, but type hints add documentation and help catch bugs:

```python
def entropy_from_probs(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
#                       ^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^^^
#                       param type          param with default    return type
```

Common types used:
- `torch.Tensor` — a PyTorch tensor
- `int`, `float`, `str`, `bool` — basic Python types
- `List[int]` — a list of integers
- `Dict[int, int]` — a dict mapping int keys to int values
- `Optional[float]` — either a float or None
- `Tuple[int, ...]` — a tuple of integers

---

## 25. Binary Search

Binary search finds a value satisfying a condition in O(log n) steps by halving the search range each time.

Used in `find_budget_threshold` to find the `gate_threshold` that achieves a target FLOPs budget:

```python
lo, hi = 0.05, 0.99
for _ in range(18):          # 18 iterations → precision of ~(0.99-0.05)/2^18 ≈ 0.000004
    mid = (lo + hi) / 2
    f = avg_flops(mid)        # measure actual FLOPs fraction at this threshold
    if f > target_budget:
        hi = mid              # threshold too low (exits too early → too few FLOPs? 
                              # actually: lower threshold → easier to exit → fewer FLOPs)
    else:
        lo = mid
```

The relationship: higher threshold → harder to exit → more FLOPs. So to increase FLOPs, raise the threshold (raise lo), and vice versa.

---

## 26. The Big Picture

Here is how all the pieces work together end-to-end:

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. LOAD DATA                                                         │
│     CIFAR-10/100 or TinyImageNet → augment → batch → DataLoader      │
└────────────────────────┬────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│  2. BUILD MODEL: EarlyExitResNet                                      │
│                                                                       │
│   [Stem] → [Layer1] → [Layer2] → [Layer3] → [Layer4] → [FC]         │
│                          ↓            ↓           ↓                   │
│                      [ExitHead2] [ExitHead3] [ExitHead4]             │
│                      [Gate2]     [Gate3]     [Gate4]                 │
└────────────────────────┬────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│  3. TRAIN                                                             │
│                                                                       │
│   Option A (Sequential):                                             │
│     Phase 1: Train backbone with CrossEntropyLoss                    │
│     Phase 2: Freeze backbone, train ExitHeads via distillation       │
│     Phase 3: Freeze all, train Gates as binary classifiers           │
│                                                                       │
│   Option B (Joint):                                                  │
│     Train backbone + exit heads together with combined loss          │
│     Then train gates                                                  │
└────────────────────────┬────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│  4. PROFILE                                                           │
│     Measure MACs for each layer and exit head using forward hooks     │
│     Compute FLOPs fraction for each exit point                       │
└────────────────────────┬────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│  5. EVALUATE                                                          │
│                                                                       │
│   Static policy:  exit if confidence >= tau (simple threshold)       │
│   Dynamic policy: exit if GateMLP(features) >= gate_threshold        │
│                                                                       │
│   Report: accuracy, latency, FLOPs fraction, exit distribution       │
│                                                                       │
│   Optional Pareto sweep: run at many thresholds to find trade-offs   │
│   Optional per-class analysis: which classes exit early/late?        │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Concepts Summary

| Concept | What It Does |
|---------|-------------|
| ResNet | Deep CNN backbone with skip connections |
| Early Exit | Multiple classifiers at intermediate layers; skip remaining layers when confident |
| ExitHead | Small classifier (pool → FC → FC) attached mid-network |
| GateMLP | Learned binary decision: "exit here?" using 5 features |
| Knowledge Distillation | Train exit heads to mimic teacher's soft outputs |
| Temperature | Controls softness of probability distributions in distillation |
| Static Policy | Exit when max confidence > fixed threshold |
| Dynamic Policy | Exit based on learned GateMLP decision |
| MACs/FLOPs | Measure of computational cost |
| Pareto Sweep | Find accuracy vs. compute trade-off curve |
| Checkpointing | Save and resume training from disk |
| Dataclass | Clean Python data containers |
| Hooks | PyTorch callbacks to inspect/measure layer outputs |

---

*Document generated for `selfxit_v2.py` — SelfXit Extended Early-Exit ResNet*
