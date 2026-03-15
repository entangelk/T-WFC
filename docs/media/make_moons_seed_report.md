# T-WFC make_moons Seed Sweep

- Generated: 2026-03-15T12:11:43
- Dataset: `make_moons`
- Model: `2-6-2` with `32` parameters
- Seeds: `7, 11, 17, 23, 31`

## Configuration

- `samples`: `120`
- `noise`: `0.08`
- `hidden_dim`: `6`
- `observation_budget`: `8`
- `propagation_budget`: `6`
- `max_steps`: `8`
- `backtrack_tolerance`: `0.03`
- `rollback_depth`: `2`
- `max_frontier_rollbacks`: `3`
- `max_attempt_multiplier`: `8`

## Aggregate Summary

- Mean hard test accuracy: `0.780`
- Std hard test accuracy: `0.065`
- Mean hard test loss: `0.4876`
- Mean rollback count: `0.00`
- Mean forced-commit count: `0.00`
- Max active forbidden values: `0`
- Max frontier pressure: `0`
- Best seed: `7` (hard test acc `0.900`)
- Worst seed: `23` (hard test acc `0.700`)

## Highlights

### Best Seed: `7`

- Hard test accuracy/loss: `0.900` / `0.3219`
- Search pressure: `rb=0` `alt=0` `forced=0` `max_bans=0` `max_pressure=0`
- Peak ban focus: `clean`
- Latest ban delta: `none`
- Drilldown: [metrics](make_moons_seed_runs/seed_007/make_moons_seed_007_metrics.png) | [storyboard](make_moons_seed_runs/seed_007/make_moons_seed_007_storyboard.png) | [gif](make_moons_seed_runs/seed_007/make_moons_seed_007_steps.gif)

### Worst Seed: `23`

- Hard test accuracy/loss: `0.700` / `0.6281`
- Search pressure: `rb=0` `alt=0` `forced=0` `max_bans=0` `max_pressure=0`
- Peak ban focus: `clean`
- Latest ban delta: `none`
- Drilldown: [metrics](make_moons_seed_runs/seed_023/make_moons_seed_023_metrics.png) | [storyboard](make_moons_seed_runs/seed_023/make_moons_seed_023_storyboard.png) | [gif](make_moons_seed_runs/seed_023/make_moons_seed_023_steps.gif)

## Gallery

- Plot: [make_moons_seed_gallery.png](make_moons_seed_gallery.png)

![Seed gallery](make_moons_seed_gallery.png)

## Seed Table

| Seed | Tag | Collapsed | Shadow Test Acc | Hard Test Acc | Hard Test Loss | Rollbacks | Alt Choices | Forced | Max Bans | Max Pressure | Peak Ban Focus | Latest Ban Delta | Metrics | Storyboard | GIF |
|------|-----|-----------|-----------------|---------------|----------------|-----------|-------------|--------|----------|--------------|----------------|------------------|---------|------------|-----|
| 7 | BEST | 8/32 | 0.900 | 0.900 | 0.3219 | 0 | 0 | 0 | 0 | 0 | clean | none | [metrics](make_moons_seed_runs/seed_007/make_moons_seed_007_metrics.png) | [storyboard](make_moons_seed_runs/seed_007/make_moons_seed_007_storyboard.png) | [gif](make_moons_seed_runs/seed_007/make_moons_seed_007_steps.gif) |
| 11 | - | 8/32 | 0.767 | 0.767 | 0.5484 | 0 | 0 | 0 | 0 | 0 | clean | none | [metrics](make_moons_seed_runs/seed_011/make_moons_seed_011_metrics.png) | [storyboard](make_moons_seed_runs/seed_011/make_moons_seed_011_storyboard.png) | [gif](make_moons_seed_runs/seed_011/make_moons_seed_011_steps.gif) |
| 17 | - | 8/32 | 0.767 | 0.767 | 0.4730 | 0 | 0 | 0 | 0 | 0 | clean | none | [metrics](make_moons_seed_runs/seed_017/make_moons_seed_017_metrics.png) | [storyboard](make_moons_seed_runs/seed_017/make_moons_seed_017_storyboard.png) | [gif](make_moons_seed_runs/seed_017/make_moons_seed_017_steps.gif) |
| 23 | WORST | 8/32 | 0.700 | 0.700 | 0.6281 | 0 | 0 | 0 | 0 | 0 | clean | none | [metrics](make_moons_seed_runs/seed_023/make_moons_seed_023_metrics.png) | [storyboard](make_moons_seed_runs/seed_023/make_moons_seed_023_storyboard.png) | [gif](make_moons_seed_runs/seed_023/make_moons_seed_023_steps.gif) |
| 31 | - | 8/32 | 0.800 | 0.767 | 0.4664 | 0 | 0 | 0 | 0 | 0 | clean | none | [metrics](make_moons_seed_runs/seed_031/make_moons_seed_031_metrics.png) | [storyboard](make_moons_seed_runs/seed_031/make_moons_seed_031_storyboard.png) | [gif](make_moons_seed_runs/seed_031/make_moons_seed_031_steps.gif) |

## Notes

- `Max Bans` counts the largest number of active forbidden candidate values seen during the run.
- `Max Pressure` counts the highest rollback pressure observed at a single search frontier before a commit.
- `Storyboard` and `GIF` links appear only for 2D datasets where decision-surface rendering is available.

## Seed Drilldown

### Seed 7 [BEST]

- Peak Ban Focus: `clean`
- Latest Ban Delta: `none`
- Metrics: [open](make_moons_seed_runs/seed_007/make_moons_seed_007_metrics.png)
- Storyboard: [open](make_moons_seed_runs/seed_007/make_moons_seed_007_storyboard.png)
- GIF: [open](make_moons_seed_runs/seed_007/make_moons_seed_007_steps.gif)

### Seed 23 [WORST]

- Peak Ban Focus: `clean`
- Latest Ban Delta: `none`
- Metrics: [open](make_moons_seed_runs/seed_023/make_moons_seed_023_metrics.png)
- Storyboard: [open](make_moons_seed_runs/seed_023/make_moons_seed_023_storyboard.png)
- GIF: [open](make_moons_seed_runs/seed_023/make_moons_seed_023_steps.gif)

### Seed 11

- Peak Ban Focus: `clean`
- Latest Ban Delta: `none`
- Metrics: [open](make_moons_seed_runs/seed_011/make_moons_seed_011_metrics.png)
- Storyboard: [open](make_moons_seed_runs/seed_011/make_moons_seed_011_storyboard.png)
- GIF: [open](make_moons_seed_runs/seed_011/make_moons_seed_011_steps.gif)

### Seed 17

- Peak Ban Focus: `clean`
- Latest Ban Delta: `none`
- Metrics: [open](make_moons_seed_runs/seed_017/make_moons_seed_017_metrics.png)
- Storyboard: [open](make_moons_seed_runs/seed_017/make_moons_seed_017_storyboard.png)
- GIF: [open](make_moons_seed_runs/seed_017/make_moons_seed_017_steps.gif)

### Seed 31

- Peak Ban Focus: `clean`
- Latest Ban Delta: `none`
- Metrics: [open](make_moons_seed_runs/seed_031/make_moons_seed_031_metrics.png)
- Storyboard: [open](make_moons_seed_runs/seed_031/make_moons_seed_031_storyboard.png)
- GIF: [open](make_moons_seed_runs/seed_031/make_moons_seed_031_steps.gif)

