<p align="center">
  <a href="./VERIFICATION.en.md"><img src="https://img.shields.io/badge/Language-EN-111111?style=for-the-badge" alt="English"></a>
  <a href="./VERIFICATION.md"><img src="https://img.shields.io/badge/Language-KO-6B7280?style=for-the-badge" alt="한국어"></a>
</p>
<p align="center"><sub>Switch language / 언어 전환</sub></p>

# Verification Guide

This document explains why the current checks exist, what each test family is trying to protect, and how generated reports should be organized as the project grows.

## Why We Test This Way

`T-WFC` is a research prototype, not a conventional gradient-based training stack. That changes the testing priorities.

- We care more about protecting the collapse loop semantics than about micro-optimizing performance.
- We want to catch silent regressions in rollback, forced commits, and snapshot bookkeeping.
- We need file-output checks because the project produces visual artifacts and Markdown reports as part of the research workflow.
- We keep tests small and cheap so they can run often while the algorithm is still changing quickly.

## What The Current Checks Target

### 1. Dataset Integrity

These tests verify that `make_moons`, `spiral`, and Iris loading return the expected shapes, labels, and split behavior.

What this protects:
- broken dataset generation
- invalid label coverage
- accidental changes to train/test split assumptions

Why this matters now:
The project has moved beyond the original tiny `make_moons` / Iris-only scope. `spiral` is not a large modern benchmark, but it is large enough to expose search-space growth and multi-class behavior while keeping 2D visual inspection available.

### 2. Multi-Layer Model / Gradient Checks

These tests verify that deeper MLP configurations can:
- build a valid parameter layout
- run a forward pass
- produce a gradient vector with the right shape

What this protects:
- single-layer assumptions leaking into multi-layer code
- broken parameter packing/unpacking
- future regressions in the SGD baseline path
### 3. Trainer Smoke Tests

These tests run short end-to-end experiments and assert:
- finite losses
- bounded accuracies
- expected step counts
- snapshot alignment with final counters

What this protects:
- collapse loop regressions
- metric calculation bugs
- broken snapshot/result summaries

### 4. Rollback / Contradiction Stress Tests

These tests intentionally use harsh settings such as negative tolerance so the trainer is forced into rollback-heavy behavior.

What this protects:
- rollback paths that rarely trigger in normal runs
- forced-commit fallback behavior
- forbidden-value ban tracking
- frontier pressure accounting

Why this matters:
The hardest bugs in this project usually appear when the search stalls or contradicts itself, not during clean runs.

### 5. Visualization Output Tests

These tests write PNG and GIF files and confirm they exist and are non-empty.

What this protects:
- plotting regressions
- serialization failures
- broken figure/export wiring

What it does not protect:
- visual quality
- exact pixel correctness
- whether a plot is "good" for presentation

Those still need human review.

The comparison path now also checks that `T-WFC vs SGD` metrics boards, 2D boundary boards, and comparison GIFs are written successfully.

### 6. Batch Reporting Tests

These tests verify:
- seed gallery generation
- per-seed artifact export
- Markdown report generation
- best/worst highlighting
- peak-ban and latest-ban summaries

What this protects:
- research reporting workflow
- drill-down links between gallery, report, and per-seed artifacts
- summary logic for multi-seed comparison

### 7. SGD Baseline Smoke Tests

These tests verify that a conventional `numpy` SGD path improves on at least one stable dataset with a deeper MLP.

What this protects:
- the comparison baseline itself
- backprop wiring in the generalized MLP
- the claim that T-WFC is being compared against a real optimization path instead of only against random initialization

What it does not guarantee:
- that T-WFC beats SGD
- that the comparison is already fair at larger scale
- that hard-weight quality tracks shadow-weight quality on deeper models

## How To Run Verification

### Main smoke suite

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

### Syntax sanity check

```bash
python3 -m py_compile src/t_wfc/*.py tests/test_smoke.py
```

### Packaging / CLI sanity check

If you want to verify the installable CLI path:

```bash
python3 -m pip install -e .
t-wfc --help
```

This confirms that `pyproject.toml` metadata and the console entry point are wired correctly.

### Multi-layer comparison sanity check

```bash
t-wfc --dataset iris --hidden-layers 16,16 --max-steps 18 --compare-sgd --sgd-epochs 160 --sgd-batch-size 24
```

This is a useful “does the scaled path still make sense?” check because it exercises:
- the deeper MLP path
- automatic initial jitter for T-WFC
- the SGD baseline path
- the side-by-side CLI summary

## What The Current Tests Do Not Guarantee

- They do not prove the algorithm is numerically optimal.
- They do not lock down exact accuracy targets across every seed.
- They do not compare images against golden references.
- They do not yet benchmark runtime or memory.
- They do not replace manual review of storyboard, GIF, and report readability.

That is intentional. The project is still in concept-validation mode.

## Report Growth And Artifact Layout

Reports will continue to grow as more datasets and seed sweeps are added. To keep that manageable, generated outputs should be grouped by dataset.

Recommended layout:

```text
artifacts/
  make_moons/
    plots/
    animations/
    frames/
    reports/
  iris/
    plots/
    reports/
  spiral/
    plots/
    reports/
```

Recommended practice:
- keep generated research outputs in `artifacts/<dataset>/...`
- keep top-level docs in `docs/`
- avoid mixing permanent reference docs with generated run outputs

## Why This Matters For Public Publishing

For a public repository, a good verification story should answer three questions quickly:

1. Can someone run the code?
2. Can they trust that the basic research loop still works?
3. Can they understand what the visual/report artifacts are showing?

This guide exists to make those answers visible without forcing a new reader to reverse-engineer the test suite first.
