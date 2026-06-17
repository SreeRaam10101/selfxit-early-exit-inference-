"""
channels_last memory format experiment.
Nsight profiling found nchwToNhwc conversion = 14.1% of GPU time.
This script measures whether pinning the model + inputs to channels_last
eliminates that overhead and improves throughput.
"""
import time
import torch
import torch.nn as nn
import sys
sys.path.insert(0, "/home/raam/selfxit")
from selfxit_v2 import EarlyExitResNet, load_checkpoint
import torchvision
import torchvision.transforms as T

CKPT_DIR   = "./ckpts"
DATASET    = "cifar10"
BATCH_SIZE = 128
N_WARMUP   = 10
N_BENCH    = 50
DEVICE     = torch.device("cuda")

def get_testloader():
    tf = T.Compose([T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010))])
    ds = torchvision.datasets.CIFAR10("./data", train=False,
                                      download=True, transform=tf)
    return torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE,
                                       num_workers=2, pin_memory=True)

def bench(model, loader, fmt_name, channels_last=False):
    model.eval()
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    # Warmup
    images, _ = next(iter(loader))
    images = images.to(DEVICE)
    if channels_last:
        images = images.to(memory_format=torch.channels_last)

    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = model(images)
    torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            if i >= N_BENCH:
                break
            imgs = imgs.to(DEVICE)
            if channels_last:
                imgs = imgs.to(memory_format=torch.channels_last)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(imgs)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

    import numpy as np
    lats = torch.tensor(latencies)
    throughput = BATCH_SIZE / (lats.mean().item() / 1000)
    print(f"\n[{fmt_name}]")
    print(f"  Throughput : {throughput:.0f} samples/sec")
    print(f"  Latency    : P50={torch.quantile(lats, 0.50).item():.2f}ms  "
          f"P95={torch.quantile(lats, 0.95).item():.2f}ms  "
          f"P99={torch.quantile(lats, 0.99).item():.2f}ms")
    return throughput

def main():
    loader = get_testloader()

    print("Loading model...")
    model = EarlyExitResNet("resnet18", num_classes=10, cifar_stem=True)
    load_checkpoint(model, CKPT_DIR, device=DEVICE)
    model = model.to(DEVICE)

    print(f"\n{'='*50}")
    print("channels_last Memory Format Benchmark")
    print(f"{'='*50}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {BATCH_SIZE}, Warmup: {N_WARMUP}, Bench batches: {N_BENCH}")

    t_default = bench(model, loader, "NCHW (default)",   channels_last=False)
    # Reload clean model for fair comparison
    model2 = EarlyExitResNet("resnet18", num_classes=10, cifar_stem=True)
    load_checkpoint(model2, CKPT_DIR, device=DEVICE)
    model2 = model2.to(DEVICE)
    t_cl = bench(model2, loader, "NHWC (channels_last)", channels_last=True)

    speedup = t_cl / t_default
    print(f"\n{'='*50}")
    print(f"Speedup from channels_last: {speedup:.3f}x  "
          f"({'faster' if speedup > 1 else 'slower'})")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
