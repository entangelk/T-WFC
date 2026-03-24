# T-WFC Experiment Results and Conclusions

> **Last experiment date**: 2026-03-18
> **Reproduce**: `PYTHONPATH=src python3 -m t_wfc.cli --dataset <name> --compare-sgd --sgd-epochs 300 --seed 7 --show-steps 0`
> **All results measured with seed=7, NumPy CPU ops, shared dataset objects**

---

## 1. Objective

Verify whether WFC (Wave Function Collapse) concepts can train a toy MLP
**without backpropagation, using only discrete 5-value weights (`{-1, -0.5, 0, 0.5, 1}`)**.

Baseline: SGD+Momentum(0.9), continuous real-valued weights, identical architecture.

---

## 2. Accuracy Comparison

| Dataset | Nonlinearity | T-WFC test acc | SGD+Mom test acc | Model | T-WFC time | SGD time | Notes |
|---------|-------------|----------------|------------------|-------|-----------|----------|-------|
| **linear_binary** | None | **0.967** | **0.967** | 2-6-2 | 0.10s | 0.10s | Equal |
| **blobs_binary** | None | **1.000** | **1.000** | 2-6-2 | 0.10s | 0.11s | Equal |
| **make_blobs** | None (linearly separable) | **1.000** | **1.000** | 2-8-3 | 0.20s | 0.10s | All 5 seeds: 1.000 |
| **iris** | Weak | **0.972** | 0.944 | 4-8-3 | 0.27s | 0.13s | T-WFC slightly better |
| **make_moons** | Moderate | 0.933 | **1.000** | 2-6-2 | 0.11s | 0.11s | Gap begins |
| **xor** | Moderate | 0.660 | **1.000** | 2-6-2 | 0.12s | 0.15s | Effective failure |
| **circles** | Moderate | 0.620 | **1.000** | 2-6-2 | 0.12s | 0.15s | Effective failure |
| **spiral** | Strong | 0.433 | **0.987** | 2-24-24-3 | 21.55s | 0.82s | Effective failure |

### Pattern

- **Linearly separable**: T-WFC = SGD. Discrete 5 values can construct a perfect decision boundary.
- **Weak nonlinearity (iris)**: T-WFC slightly outperforms SGD. The discrete weight space is sufficient for iris-level separation.
- **Moderate nonlinearity and above**: T-WFC degrades sharply. The search capacity of single-weight collapse with 5 discrete values is fundamentally insufficient.

---

## 3. Speed and Memory Comparison

| Dataset | T-WFC time | SGD time | T-WFC memory | SGD memory | T-WFC forwards | SGD fwd+bwd |
|---------|-----------|----------|-------------|-----------|----------------|-------------|
| make_blobs | 0.39s | 0.26s | 400K | 83K | ~2,040 | ~900 |
| make_moons | 0.20s | 0.24s | 186K | 75K | ~1,280 | ~900 |
| xor | 0.26s | 0.39s | 191K | 83K | ~1,280 | ~1,500 |
| circles | 0.23s | 0.37s | 191K | 83K | ~1,280 | ~1,500 |
| iris | 0.54s | 0.30s | 601K | 85K | ~2,680 | ~1,200 |
| spiral | **26.25s** | 1.57s | **56,459K** | 483K | ~29,880 | ~4,500 |

### Analysis

- **Small models (32–67 params)**: Comparable wall-clock time. T-WFC is occasionally faster (xor, circles, make_moons).
- **Memory**: T-WFC always uses more memory — maintaining per-weight probability distributions over the domain.
- **Scaling limit**: At 747 parameters (spiral), T-WFC is **16.7× slower and uses 117× more memory**. Complexity grows as O(n × budget × domain_size), making it impractical beyond a few hundred parameters.
- **No speed advantage**: No scenario was found where T-WFC consistently outperforms SGD in wall-clock time.

---

## 4. Conclusions

### What was proven (PoC success)

1. **WFC → weight training mapping works**: The observe → collapse → propagate → backtrack loop genuinely trains a toy MLP.
2. **Matches SGD on linearly separable problems**: make_blobs achieves 100% across all 5 seeds; iris slightly outperforms SGD.
3. **Gradient-free learning is possible**: No autograd, no backpropagation, no optimizer — pure forward passes and discrete state search.

### Limitations

1. **Fundamentally insufficient search capacity for nonlinear problems**: XOR, circles, spiral — the combinatorial space of 5 discrete values cannot represent the required decision boundaries.
2. **Does not scale**: Time and memory grow sharply with parameter count. Impractical beyond a few hundred parameters.
3. **Memory overhead**: Probability distribution maintenance cost always exceeds SGD.
4. **No speed advantage**: Even on small models, no consistent speed benefit over SGD.

### Root Cause

WFC was designed for **spatially local discrete constraint satisfaction problems** like 2D tilemap generation. Neural network weight spaces are:
- High-dimensional,
- Non-local in interaction (one layer's weights affect all subsequent layers through nonlinear activations),
- Unable to express smooth decision boundaries through discrete value combinations.

T-WFC succeeds on linearly separable problems because the decision boundary is a simple hyperplane — expressible with 5-value combinations. Once nonlinearity is required, this breaks down.

### Retrospective on algorithmic identity

In practice, T-WFC behaves more like greedy coordinate descent over a discrete domain than like WFC proper. Without WFC's core mechanism — hard adjacency constraints between tiles — loss-based propagation alone cannot achieve the global consistency that WFC was designed to enforce. The current implementation is more accurately described as a gradient-free discrete search combining entropy-based observation with neighbor propagation heuristics.

### Unexplored directions

- **Larger domain**: 21 values at 0.1 spacing → more expressiveness, but search time explodes
- **Simultaneous collapse**: Collapsing multiple weights per step → better search efficiency, but conflicts with core WFC mechanics
- **Spatially local architectures**: CNN kernels with local connectivity may fit WFC's propagation model more naturally
- **Hardware-constrained settings**: Edge devices requiring low-bit weights — T-WFC as an alternative to post-training quantization

---

## 5. Fairness Declaration

- T-WFC: Discrete 5-value constraint (`{-1, -0.5, 0, 0.5, 1}`), gradient-free
- SGD baseline: Continuous real-valued weights, momentum=0.9, lr=0.08, decay=0.01, batch_size=32
- Same model architecture, same data split (shared `DatasetSplit` object), same evaluation function
- SGD epochs (300) verified to be sufficient for convergence on all datasets
- All numbers reproducible with `seed=7` and `np.random.default_rng`
