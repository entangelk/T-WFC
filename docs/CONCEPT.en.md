<p align="center">
  <a href="./CONCEPT.en.md"><img src="https://img.shields.io/badge/Language-EN-111111?style=for-the-badge" alt="English"></a>
  <a href="./CONCEPT.md"><img src="https://img.shields.io/badge/Language-KO-6B7280?style=for-the-badge" alt="한국어"></a>
</p>
<p align="center"><sub>Switch language / 언어 전환</sub></p>

# Mini Project Concept: WFC-Based Neural Network Weight Collapse Experiment

## 1. Project Overview

* **Project name:** Tensor Wave Function Collapse (T-WFC)
* **Goal:** Verify whether neural-network weights can be optimized without gradient descent by mapping WFC concepts such as `superposition` and `collapse` onto network training.
* **Target environment:** Personal machine, CPU or single GPU
* **Core scope:** Tiny MLPs with fewer than 100 parameters

## 2. Core Algorithm Mapping (WFC -> Neural Network)

The procedural map-generation logic of WFC is mapped onto neural-network weights as follows.

| WFC Concept | T-WFC Mapping |
| --- | --- |
| **Tile** | Individual neural-network weight parameter (`w_1, w_2, ...`) |
| **State space** | The discrete set of values each weight may take, for example `D = {-1, 0, 1}` |
| **Superposition** | A weight is not fixed yet and simultaneously carries probabilities over multiple candidate values |
| **Observation** | Run a minibatch, temporarily clamp one weight, and measure the loss change to estimate that weight's entropy |
| **Collapse** | Permanently fix the single weight with the lowest uncertainty |
| **Propagation** | Recompute or adjust the distributions of neighboring weights after one weight becomes fixed |

## 3. Environment and Datasets

* **Language and libraries:** Python 3.x, NumPy (or PyTorch used only for tensor math, without `autograd`)
* **Datasets**
  * Priority 1: **Iris** (input dimension 4, output classes 3) because it stays very small.
  * Priority 2: **Scikit-learn make_moons** because a 2D boundary is easy to visualize.
  * Current implementation note: the codebase supports both an internal `make_moons` generator and a vendored `iris.csv`, so no external network access is required.
  * Current experiment note: because decision surfaces are easy to inspect, `make_moons` is the preferred path for visualization work.

* **Model structure (Toy MLP)**
  * Iris expansion target:
    * Input: 4 nodes
    * Hidden: 8 nodes (`4 x 8 = 32` weights)
    * Output: 3 nodes (`8 x 3 = 24` weights)
    * Total: about 60 parameters including biases
  * Current prototype paths:
    * `make_moons`: `2-6-2` MLP with 32 parameters including biases
    * `iris`: `4-8-3` MLP with 67 parameters including biases

## 4. Planned Pipeline

### Step 1: Initialization

* Define all weights over a discrete pool such as `W in {-1, -0.5, 0, 0.5, 1}`.
* Start from a uniform distribution across those candidate values.

### Step 2: Observation and Shadow Forwarding

* **Shadow forwarding:** unresolved weights use the expectation of their current probability distributions as temporary values.
* For each unresolved weight, force it to each candidate value and measure the minibatch loss.
* Convert those losses into Shannon entropy. If one candidate clearly dominates, entropy becomes low.
* The current prototype also records both `shadow_weights` (expectation-based) and `hard_weights` (argmax-based) so partially collapsed states can be compared directly.

### Step 3: Collapse

* Select the single weight with the lowest entropy.
* Permanently fix it to one value. Its distribution becomes a one-hot state.

### Step 4: Propagation

* Once one weight collapses, the network's information flow changes.
* Recompute or adjust the distributions of neighboring weights that share nodes with the collapsed weight.
* **Stop condition:** repeat until every weight has collapsed to a single value.

## 5. Expected Issues and Mitigations

* **Contradictions:** A bad collapse path can lead to a dead end where loss no longer improves.
* **Mitigation:** Introduce backtracking that rewinds recent collapse steps when the loss moves outside an allowed range.
* **Current prototype implementation:** when only unacceptable collapses remain, it rewinds a fixed number of recent steps and temporarily bans the oldest reverted choice.
* **Current prototype safeguard:** when rollback keeps repeating at the same frontier, a forced-commit safety valve prevents zero-progress oscillation.
* **Current prototype safeguard 2:** forced commits do not just take a local fallback for the current observation; they search unresolved weights globally and choose the best collapse under a hybrid `shadow loss + hard loss + gap penalty` score.

* **Computation cost:** testing all weights against all candidate values scales poorly.
* **Mitigation:** observe only a subset of unresolved weights per step as a heuristic.

## 6. Expected Value

1. **Learning without backpropagation:** test whether gradient descent is not the only path.
2. **Extreme quantization:** final models can remain in low-precision discrete states such as `-1, 0, 1`.
3. **Traceability:** collapse logs make it easier to inspect why a specific weight value was chosen.
