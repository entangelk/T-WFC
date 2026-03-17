# Work Log - 2026-03-17

## Goal

- Address external review feedback on README structure (remove meta section, trim Current Status, reduce Quickstart commands).
- Audit the fairness of T-WFC vs SGD comparisons: same conditions, proper baseline tuning, asymmetry disclosure.
- Fix the SGD baseline to use momentum and re-tune comparison settings so the baseline represents a properly configured optimizer.
- Add baseline comparison fairness guidelines to CLAUDE.md and AGENTS.md for future work.

## Completed Work

- Trimmed `README.md` and `README.ko.md`.
  - removed the "What This README Is Trying To Show" / "이 README에서 보여주려는 것" section — the section titles already convey the same information
  - condensed "Current Status" from 15+ nested bullets to 6 lines with a CHANGELOG link
  - reduced Quickstart from 8 command blocks to 3 (install + 2 basic runs) plus `--help` and docs pointers
- Audited the SGD baseline fairness.
  - discovered that the SGD baseline was using vanilla SGD (no momentum) while being presented as "SGD baseline" — this significantly understated SGD's true capability
  - ran hyperparameter sweeps on all three datasets (make_moons, spiral, iris) across epochs, learning rates, and momentum values
  - key finding: spiral vanilla SGD reached 0.422, but SGD+momentum(0.9) reached 0.911 — the old comparison was deeply misleading
  - make_moons: vanilla 0.975 → momentum 1.000; iris: unchanged at 0.944
- Updated `src/t_wfc/baseline.py`.
  - added `momentum` parameter to `SGDBaselineConfig` (default 0.9)
  - added velocity tracking and momentum update in `train_sgd_classifier`
  - added validation for momentum range [0.0, 1.0)
- Updated `src/t_wfc/cli.py`.
  - added `--sgd-momentum` CLI option (default 0.9)
  - wired it into `SGDBaselineConfig` construction
- Updated `README.md` and `README.ko.md` comparison sections.
  - renamed section from "T-WFC vs SGD" to "T-WFC vs SGD+Momentum"
  - added opening paragraph explaining the discrete-vs-continuous weight asymmetry
  - updated all comparison numbers: make_moons 0.925 vs 1.000, spiral 0.367 vs 0.911, iris 0.556 vs 0.944
  - changed caption tone from "presentation-friendly" to "closest case"
- Regenerated public comparison media under `docs/media/`.
  - refreshed `make_moons_twfc_vs_sgd.gif` and `make_moons_twfc_vs_sgd_boundaries.png` (SGD+momentum, epochs=200)
  - refreshed `spiral_twfc_vs_sgd.gif` (SGD+momentum, epochs=400)
  - refreshed `iris_twfc_vs_sgd_metrics.png` (SGD+momentum, epochs=200)
- Added baseline comparison fairness guidelines to `CLAUDE.md` and `AGENTS.md`.
  - new section 6.6 "Baseline 비교 공정성 규칙" covering: equal conditions, asymmetry disclosure, verification perspective, test/comparison function review
  - also fixed a truncated sentence in the existing 6.5 section of both files
  - renumbered 6.6 → 6.7 for "현재 단계에서 피해야 할 것"
- Updated `HANDOFF.md`.
  - updated comparison runbook commands with new epoch counts
  - recorded the momentum addition and README trimming in Current State
- Updated `CHANGELOG.md` with a 2026-03-17 entry.
- Full codebase re-review of trainer.py, test_smoke.py, reporting.py, visualization.py.
  - trainer.py: scoring, propagation, trial_score logic verified as correct; `_trial_score` weights are already configurable via `config.hard_loss_weight` and `config.hard_gap_weight`
  - visualization.py: all three comparison functions (`save_baseline_metrics_comparison_plot`, `save_baseline_comparison_plot`, `save_baseline_comparison_gif`) use identical axis scales, shared grids, and consistent color coding — no visual bias
  - reporting.py: divergence calculations (`shadow - hard` for accuracy, `hard - shadow` for loss) are directionally correct
  - test_smoke.py: identified that existing 26 tests only check structural properties (shapes, finiteness, "runs without crashing"), not algorithmic correctness
- Added 10 algorithmic correctness tests to `tests/test_smoke.py` (total: 36 tests).
  - `ObservationDirectionalityTest`: verifies observation assigns highest probability to lowest-loss value; verifies `_loss_to_distribution` monotonicity with known loss vector
  - `CollapseSelectionTest`: verifies `_select_next_weight` picks the weight with lowest entropy across all candidates
  - `PropagationEffectTest`: verifies propagation modifies neighbor distributions after collapse; verifies stronger `propagation_blend` causes larger distribution shift
  - `TrialScoreTest`: verifies exact `_trial_score` calculation against formula; verifies hard_gap penalty activates only when hard_loss > shadow_loss
  - `SGDMomentumCorrectnessTest`: verifies momentum=0.9 produces different weights from momentum=0.0; verifies momentum improves convergence on make_moons
- Verified determinism: all random sources use `np.random.default_rng(seed)` with no global state dependency — same seed produces identical results across environments.

## Technical Decisions

- Chose momentum=0.9 as the default because it is the most standard setting and dramatically improves SGD convergence on spiral without requiring Adam complexity.
- Did not add Adam — momentum SGD is a fair, well-understood baseline. Adam could be added later if needed.
- Kept the same default learning rate (0.08) and decay (0.01) since momentum=0.9 works well with these on all three datasets.
- Updated comparison epoch counts to ensure SGD convergence:
  - make_moons: 140 → 200 (already converged at 140 with momentum, but 200 gives margin)
  - spiral: 220 → 400 (needed more epochs to converge with momentum on a harder dataset)
  - iris: 160 → 200 (already at ceiling accuracy regardless)
- The README now explicitly states that T-WFC uses 5 discrete values while SGD uses continuous weights — this is the most important asymmetry and was previously missing.

## Verification

- `python3 -m py_compile src/t_wfc/*.py tests/test_smoke.py`
  - Result: passed
- `PYTHONPATH=src python3 -m unittest discover -s tests`
  - Result: passed (Ran 36 tests in 20.834s)
- SGD momentum sweep (seed=7, all three datasets)
  - make_moons: momentum SGD converges to 1.000 test acc across all tested lr/epochs combinations
  - iris: 0.944 test acc across all combinations (saturated)
  - spiral: best momentum SGD reaches 0.911 (epochs=400, lr=0.08, mom=0.9) vs old vanilla 0.422
- CLI comparison runs verified:
  - make_moons: hard 0.925 vs SGD+mom 1.000 ✓
  - spiral: hard 0.367 vs SGD+mom 0.911 ✓
  - iris: hard 0.556, shadow 0.750 vs SGD+mom 0.944 ✓
- All 12 docs/media/ files referenced by README exist and are non-empty.

## Next Steps

- Consider whether the spiral comparison merits a different experimental approach (more collapse steps, wider domain, etc.) now that the honest gap is much larger.
- Continue tuning T-WFC behavior on deeper models — the gap against a proper SGD baseline is now honestly visible.
- The new fairness guidelines in CLAUDE.md/AGENTS.md should be applied retroactively to any future baseline additions.
