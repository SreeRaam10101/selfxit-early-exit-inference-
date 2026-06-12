# Cascade Inference (plan.md #1) — Design

## Motivation

Today, `EarlyExitResNet._backbone_features()` always runs the **entire**
backbone (`layer1`..`layer4` + `fc`) regardless of where a sample's exit
policy decides to exit. `inference_static` / `inference_dynamic` compute all
exit-head logits up front via `forward_with_exits()`, then pick which one to
report. Exit-id bookkeeping and FLOPs-fraction accounting are correct in
theory, but no compute is actually skipped — `PROFILING.md` and
`benchmark_single_sample` measure latency for a path that does strictly more
work than necessary.

This change makes batch-size-1 inference **actually stop** running backbone
layers once an exit policy fires, and adds a benchmark that shows the
resulting latency delta.

## Scope

- **In scope**: `inference_static(..., cascade=True)` and
  `inference_dynamic(..., cascade=True)` for `batch_size == 1`;
  `benchmark_single_sample` updated to compare `cascade=False` vs
  `cascade=True`.
- **Out of scope**: batched (`B > 1`) cascade via gather/scatter (plan.md #1
  mentions this as a "fascinating tension" but it's a separate, larger
  change). Training, calibration, and `forward_with_exits` are untouched —
  they need every exit's logits for every sample and cannot skip layers.
- **No new CLI flags.** The existing `--benchmark_single` flag now produces
  both cascade and non-cascade results.

## Architecture

### `_cascade_steps` generator (new private method on `EarlyExitResNet`)

```python
def _cascade_steps(self, x: torch.Tensor):
    """Lazily run the backbone layer-by-layer for a single sample (B == 1).

    Yields (exit_idx, logits, feat) for each configured exit head, in order,
    then a final (num_exits, final_logits, None) for the backbone head.
    Because this is a generator, layers after the one a consumer stops at
    are never executed.
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
```

Generator laziness *is* the cascade: a `for` loop that `return`s as soon as
an exit fires means `getattr(self.backbone, f"layer{layer_idx}")(out)` for
later layers is never called.

### `inference_static(x, tau=0.9, tau_entropy=None, cascade=False)`

When `cascade=True`:

```python
assert x.size(0) == 1, "cascade=True only supports batch size 1"
for exit_idx, logits, _ in self._cascade_steps(x):
    if exit_idx == self.num_exits:
        return logits, torch.tensor([exit_idx], device=x.device)
    probs = F.softmax(logits, 1)
    max_conf, _ = probs.max(1)
    condition = max_conf >= tau
    if tau_entropy is not None:
        ent = entropy_from_probs(probs)
        condition = condition & (ent <= tau_entropy)
    if condition.item():
        return logits, torch.tensor([exit_idx], device=x.device)
```

This mirrors the per-exit decision logic already in the `cascade=False` loop
(same `tau` / `tau_entropy` formulas), just driven by the generator instead
of a precomputed `exit_logits_list`.

### `inference_dynamic(x, gate_threshold=0.8, cascade=False)`

Same structure, but the per-exit decision uses the gate MLP:

```python
assert x.size(0) == 1, "cascade=True only supports batch size 1"
for exit_idx, logits, feat in self._cascade_steps(x):
    if exit_idx == self.num_exits:
        return logits, torch.tensor([exit_idx], device=x.device)
    probs = F.softmax(logits, 1)
    dnorm = (exit_idx + 1) / self.num_exits
    feats = self._gate_features(probs, logits, dnorm)
    gate_prob = torch.sigmoid(self.gates[exit_idx](feats))
    if (gate_prob >= gate_threshold).item():
        return logits, torch.tensor([exit_idx], device=x.device)
```

`dnorm` matches the existing convention from the batched loop
(`depth_norms = [i / self.num_exits for i in range(1, self.num_exits + 1)]`).

The `cascade=False` branches of both methods are **unchanged** — existing
batched behavior, used by eval/sweep/per-class analysis, is untouched.

## `benchmark_single_sample` changes

Current signature/behavior: for `n_runs` random single images, call
`inference_static`/`inference_dynamic` (always `cascade=False`), time with
`time.perf_counter()`, report overall + per-exit P50/P95/P99.

New behavior, for each sampled image:

1. Run with `cascade=False` → record time, logits, exit_id.
2. Run with `cascade=True` → record time, logits, exit_id.
3. **Equivalence check**: `assert exit_id_cascade == exit_id_nocascade` and
   `torch.allclose(logits_cascade, logits_nocascade, atol=1e-5)`. Any
   mismatch raises immediately — this is the correctness guard for the new
   lazy path, exercised on every benchmark run.

Report two sets of P50/P95/P99 tables (overall + per-exit breakdown), one
for each mode, labeled "full backbone (no cascade)" and "cascade
(early-stop)". This is the before/after comparison plan.md #1 calls for.

## Testing / Validation

- The equivalence assertion inside `benchmark_single_sample` is the primary
  correctness check (runs on real data, all exit paths, every benchmark
  invocation).
- Manual smoke test: run
  `python selfxit_v2.py --dataset cifar10 --model resnet18 --policy both --benchmark_single --benchmark_n_runs 100`
  on the existing `ckpts/model_30ep.pt` checkpoint and confirm:
  - no assertion errors
  - cascade P50/P95/P99 <= no-cascade P50/P95/P99 for exit_ids 0 and 1
    (exit_id == num_exits, i.e. full backbone, should be ~equal between the
    two modes since no layers are skipped)

## Out of Scope / Follow-ups

- Batched gather/scatter cascade (plan.md #1's harder variant).
- Updating `MACProfile` / `flops_fraction` to report *measured* (vs.
  theoretical) MACs saved by cascade — could be a natural follow-up once
  this lands.
- `channels_last` memory format experiment (separate profiling follow-up,
  unrelated to this change).
