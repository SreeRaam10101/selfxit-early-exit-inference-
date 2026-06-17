"""
selfxit_kernels.py — Triton kernel for gate feature computation.

Fuses the 5-op _gate_features() into a single kernel:
  [max_conf, entropy, logit_margin, depth_norm, logits_l2]

One Triton program per batch row. Reads probs + logits ONCE from HBM,
computes all 5 features in registers, writes a 5-wide output row.

Usage:
  python selfxit_kernels.py --num_classes 10   # benchmark CIFAR-10
  python selfxit_kernels.py --num_classes 100  # benchmark CIFAR-100

  # Nsight Compute profile:
  ncu --set full python selfxit_kernels.py --num_classes 10 --ncu_mode
"""

import argparse
import time
import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
#  Triton kernel
# ---------------------------------------------------------------------------

@triton.jit
def _gate_features_kernel(
    probs_ptr,   # [B, C] float32 — softmax probabilities
    logits_ptr,  # [B, C] float32 — raw logits
    out_ptr,     # [B, 5] float32 — output feature vector
    depth_norm,  # float32 scalar — normalised exit depth
    C,           # int — number of classes
    stride_pb, stride_pc,
    stride_lb, stride_lc,
    BLOCK_C: tl.constexpr,
):
    """
    Each program handles one row (one batch element).

    Features computed (in order, matching _gate_features output):
      0: max_conf   = max(probs)
      1: entropy    = -sum(p * log(p + eps))
      2: margin     = top1_logit - top2_logit
      3: depth_norm = scalar (same for every row)
      4: l2         = ||logits||_2
    """
    row = tl.program_id(0)

    offs = tl.arange(0, BLOCK_C)
    mask = offs < C

    # Load probs and logits for this row
    p = tl.load(probs_ptr + row * stride_pb + offs * stride_pc,
                 mask=mask, other=0.0)
    l = tl.load(logits_ptr + row * stride_lb + offs * stride_lc,
                 mask=mask, other=-1e30)

    # --- Feature 0: max_conf -------------------------------------------------
    max_conf = tl.max(p, axis=0)

    # --- Feature 1: entropy --------------------------------------------------
    # H = -sum(p * log(p + eps)),  eps = 1e-8 matches entropy_from_probs()
    log_p = tl.log(p + 1e-8)
    entropy = -tl.sum(p * log_p, axis=0)

    # --- Feature 2: logit margin (top1 - top2) --------------------------------
    # Step 1: find top-1
    top1 = tl.max(l, axis=0)
    # Step 2: find argmax of top-1, then hide that position to get top-2
    top1_idx = tl.argmax(l, axis=0)
    l_top2 = tl.where(offs == top1_idx, -1e30, l)
    top2 = tl.max(l_top2, axis=0)
    margin = top1 - top2

    # --- Feature 4: L2 norm of logits ----------------------------------------
    l_sq = tl.where(mask, l * l, 0.0)
    l2 = tl.sqrt(tl.sum(l_sq, axis=0))

    # --- Write output [5 scalars] --------------------------------------------
    base = out_ptr + row * 5
    tl.store(base + 0, max_conf)
    tl.store(base + 1, entropy)
    tl.store(base + 2, margin)
    tl.store(base + 3, depth_norm)
    tl.store(base + 4, l2)


# ---------------------------------------------------------------------------
#  Python wrapper (drop-in for EarlyExitResNet._gate_features)
# ---------------------------------------------------------------------------

def gate_features_triton(probs: torch.Tensor,
                         logits: torch.Tensor,
                         depth_norm: float) -> torch.Tensor:
    """
    Drop-in replacement for EarlyExitResNet._gate_features().

    Args:
        probs:      [B, C] softmax probabilities (float32, CUDA)
        logits:     [B, C] raw logits           (float32, CUDA)
        depth_norm: scalar float, normalised exit depth

    Returns:
        [B, 5] feature tensor matching _gate_features() output order:
        [max_conf, entropy, logit_margin, depth_norm, logits_l2]
    """
    assert probs.is_cuda and logits.is_cuda, "gate_features_triton requires CUDA tensors"
    assert probs.dtype == torch.float32, "gate_features_triton expects float32"
    B, C = probs.shape
    out = torch.empty(B, 5, device=probs.device, dtype=torch.float32)
    BLOCK_C = triton.next_power_of_2(C)
    num_warps = max(1, BLOCK_C // 16)

    _gate_features_kernel[(B,)](
        probs, logits, out,
        float(depth_norm),
        C,
        probs.stride(0), probs.stride(1),
        logits.stride(0), logits.stride(1),
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
    )
    return out


# ---------------------------------------------------------------------------
#  Correctness check
# ---------------------------------------------------------------------------

def _gate_features_pytorch(probs: torch.Tensor,
                            logits: torch.Tensor,
                            depth_norm: float) -> torch.Tensor:
    """Reference PyTorch implementation matching EarlyExitResNet._gate_features."""
    with torch.no_grad():
        max_conf, _ = probs.max(1)
        entropy = -(probs * (probs + 1e-8).log()).sum(1)
        top2_vals, _ = torch.topk(logits, k=2, dim=1)
        margin = top2_vals[:, 0] - top2_vals[:, 1]
        l2 = logits.norm(p=2, dim=1)
    return torch.stack([
        max_conf, entropy, margin,
        torch.full_like(max_conf, depth_norm),
        l2,
    ], dim=1)


def check_correctness(num_classes: int = 10, batch_size: int = 32,
                      atol: float = 1e-4) -> bool:
    """Verify Triton kernel matches PyTorch reference."""
    device = torch.device("cuda")
    logits = torch.randn(batch_size, num_classes, device=device)
    probs  = F.softmax(logits, dim=1)
    depth_norm = 0.5

    ref = _gate_features_pytorch(probs, logits, depth_norm)
    out = gate_features_triton(probs, logits, depth_norm)
    torch.cuda.synchronize()

    ok = torch.allclose(ref, out, atol=atol)
    if not ok:
        diff = (ref - out).abs()
        print(f"  [FAIL] max diff = {diff.max().item():.2e}  "
              f"mean diff = {diff.mean().item():.2e}")
    return ok


# ---------------------------------------------------------------------------
#  Benchmark
# ---------------------------------------------------------------------------

def benchmark(num_classes: int,
              batch_sizes: list,
              n_warmup: int = 50,
              n_bench:  int = 500,
              ncu_mode: bool = False) -> None:
    """
    Compare latency of PyTorch vs Triton gate feature computation.
    ncu_mode=True reduces iterations for clean Nsight Compute profiling.
    """
    device = torch.device("cuda")
    if ncu_mode:
        n_warmup, n_bench = 3, 10

    print(f"\n{'='*65}")
    print(f"Gate Feature Benchmark  C={num_classes}  "
          f"(warmup={n_warmup}, bench={n_bench})")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"{'='*65}")
    print(f"{'B':>6}  {'PyTorch µs':>12}  {'Triton µs':>12}  {'Speedup':>9}")
    print(f"{'-'*6}  {'-'*12}  {'-'*12}  {'-'*9}")

    for B in batch_sizes:
        logits = torch.randn(B, num_classes, device=device)
        probs  = F.softmax(logits, dim=1)
        depth_norm = 0.33

        def time_fn(fn):
            for _ in range(n_warmup):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_bench):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / n_bench * 1e6  # µs

        t_pt = time_fn(lambda: _gate_features_pytorch(probs, logits, depth_norm))
        t_tr = time_fn(lambda: gate_features_triton(probs, logits, depth_norm))
        speedup = t_pt / t_tr

        print(f"{B:>6}  {t_pt:>11.2f}µ  {t_tr:>11.2f}µ  {speedup:>8.2f}×")

    print(f"{'='*65}")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Triton gate feature kernel — correctness check + benchmark"
    )
    parser.add_argument("--num_classes", type=int, default=10,
                        help="Number of output classes (10=CIFAR-10, 100=CIFAR-100)")
    parser.add_argument("--batch_sizes", type=int, nargs="+",
                        default=[1, 8, 32, 64, 128, 256],
                        help="Batch sizes to benchmark")
    parser.add_argument("--ncu_mode", action="store_true",
                        help="Reduce iterations for clean Nsight Compute profiling")
    parser.add_argument("--skip_correctness", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available — cannot run Triton kernel.")
        return

    if not args.skip_correctness:
        print("\nCorrectness check...")
        for C in [args.num_classes]:
            for B in [1, 32, 128]:
                ok = check_correctness(C, B)
                status = "PASS" if ok else "FAIL"
                print(f"  C={C:>4}  B={B:>4}  [{status}]")

    benchmark(
        num_classes=args.num_classes,
        batch_sizes=args.batch_sizes,
        ncu_mode=args.ncu_mode,
    )


if __name__ == "__main__":
    main()
