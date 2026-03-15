<p align="center">
  <a href="./README.md"><img src="https://img.shields.io/badge/Language-EN-111111?style=for-the-badge" alt="English"></a>
  <a href="./README.ko.md"><img src="https://img.shields.io/badge/Language-KO-6B7280?style=for-the-badge" alt="한국어"></a>
</p>
<p align="center"><sub>Switch language / 언어 전환</sub></p>

# T-WFC

Tensor Wave Function Collapse (`T-WFC`) is a research prototype that tests whether a tiny neural network can be trained without gradient descent by borrowing the `superposition -> observation -> collapse -> propagation` loop from Wave Function Collapse.

## Current Status

- `make_moons` and vendored `iris.csv` are both supported.
- The trainer already supports observation, single-weight collapse, propagation, rollback-aware backtracking, and hybrid-scored forced commits.
- `make_moons` runs can save:
  - an overview plot with `initial shadow / final shadow / final hard`
  - a progress timeline plot across committed collapse steps
  - a per-snapshot frame sequence for step-by-step inspection
  - a combined storyboard with metrics, selected snapshots, and event highlights
  - an animated GIF built from committed snapshots with event badges and cumulative counters
  - rollback bursts, alt-choice retries, and forced commits are rendered as separate signals instead of one blended event tag
  - forbidden-value bans and frontier pressure are also rendered as separate overlay signals
  - ban overlays now identify which weights accumulated bans, not only how many bans existed
- Any dataset run can save a metrics timeline plot for shadow/hard loss and accuracy.
- Multi-seed runs can save a comparison gallery, per-seed drill-down artifacts, and a Markdown report with best/worst seed highlights, peak-ban summaries, and direct links to each seed's metrics/storyboard/GIF outputs.
- The package now exposes an installable `t-wfc` CLI via `pyproject.toml`.

## Quickstart

```bash
python3 -m pip install -e .
t-wfc --help
PYTHONPATH=src python3 -m unittest discover -s tests
t-wfc --dataset make_moons --max-steps 8 --show-steps 6
t-wfc --dataset make_moons --max-steps 8 --show-steps 2 --save-plot artifacts/make_moons/plots/overview.png --save-progress-plot artifacts/make_moons/plots/progress.png --progress-panels 5 --save-metrics-plot artifacts/make_moons/plots/metrics.png --save-frames-dir artifacts/make_moons/frames/steps --max-frame-count 6
t-wfc --dataset make_moons --max-steps 8 --show-steps 2 --save-storyboard artifacts/make_moons/plots/storyboard.png --storyboard-panels 5 --save-gif artifacts/make_moons/animations/steps.gif --max-frame-count 6 --gif-frame-duration-ms 350
t-wfc --dataset make_moons --max-steps 8 --seed-list 7,11,17,23,31 --save-seed-gallery artifacts/make_moons/plots/seed_gallery.png --gallery-columns 3 --save-seed-artifacts-dir artifacts/make_moons/reports/seed_runs --save-md-report artifacts/make_moons/reports/seed_report.md --report-title "T-WFC make_moons Seed Sweep"
```

## Documentation

- Concept, English: [docs/CONCEPT.en.md](./docs/CONCEPT.en.md)
- Concept, Korean: [docs/CONCEPT.md](./docs/CONCEPT.md)
- Verification, English: [docs/VERIFICATION.en.md](./docs/VERIFICATION.en.md)
- Verification, Korean: [docs/VERIFICATION.md](./docs/VERIFICATION.md)
- Project handoff: [HANDOFF.md](./HANDOFF.md)
- Change history: [CHANGELOG.md](./CHANGELOG.md)

## Repository Map

- `src/t_wfc/data.py`: dataset loading and splits
- `src/t_wfc/model.py`: toy MLP definition
- `src/t_wfc/state.py`: discrete probability state
- `src/t_wfc/trainer.py`: collapse loop, rollback logic, metrics, snapshots
- `src/t_wfc/batch.py`: repeated experiment runs across seed lists and per-seed artifact export
- `src/t_wfc/reporting.py`: Markdown seed-sweep report generation with drill-down links
- `src/t_wfc/visualization.py`: overview, progress, metrics, storyboard, GIF, frame-sequence, and seed-gallery plots
- `src/t_wfc/cli.py`: command-line entry point
- `pyproject.toml`: package metadata, dependencies, and the `t-wfc` console script

## Notes

- This is still a research prototype, not a polished training framework.
- `numpy` is the main runtime dependency.
- `matplotlib` is used for visualization output.
- `Pillow` is used for GIF export.
