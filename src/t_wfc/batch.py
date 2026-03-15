from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from .data import DatasetSplit, load_dataset
from .model import MLPConfig, ToyMLP
from .trainer import ExperimentResult, TWFCConfig, TWFCTrainer


@dataclass(frozen=True)
class SeedExperiment:
    seed: int
    dataset_name: str
    dataset: DatasetSplit
    model: ToyMLP
    result: ExperimentResult


@dataclass(frozen=True)
class SeedArtifacts:
    seed: int
    output_dir: Path
    metrics_plot: Path
    storyboard_plot: Path | None
    gif_path: Path | None


def run_seed_batch(
    dataset_name: str,
    seeds: tuple[int, ...] | list[int],
    *,
    samples: int = 120,
    noise: float = 0.08,
    hidden_dim: int = 0,
    hidden_layers: tuple[int, ...] | list[int] | None = None,
    config_template: TWFCConfig,
) -> tuple[SeedExperiment, ...]:
    experiments: list[SeedExperiment] = []

    for seed in seeds:
        dataset = load_dataset(
            dataset_name,
            n_samples=samples,
            noise=noise,
            seed=seed,
        )
        input_dim = dataset.x_train.shape[1]
        output_dim = int(max(dataset.y_train.max(), dataset.y_test.max()) + 1)
        resolved_hidden_layers = _resolve_hidden_layers(
            dataset_name=dataset_name,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
        )
        model = ToyMLP(
            MLPConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_layers=resolved_hidden_layers,
            )
        )
        trainer = TWFCTrainer(
            model=model,
            config=replace(config_template, seed=seed),
        )
        experiments.append(
            SeedExperiment(
                seed=seed,
                dataset_name=dataset_name,
                dataset=dataset,
                model=model,
                result=trainer.fit(dataset),
            )
        )

    return tuple(experiments)


def _resolve_hidden_layers(
    *,
    dataset_name: str,
    hidden_dim: int,
    hidden_layers: tuple[int, ...] | list[int] | None,
) -> tuple[int, ...]:
    if hidden_layers:
        return tuple(int(width) for width in hidden_layers)
    if hidden_dim > 0:
        return (hidden_dim,)
    if dataset_name == "spiral":
        return (24, 24)
    if dataset_name == "iris":
        return (8,)
    return (6,)


def export_seed_artifacts(
    experiments: tuple[SeedExperiment, ...] | list[SeedExperiment],
    output_dir: str | Path,
    *,
    storyboard_panels: int = 6,
    max_frames: int = 0,
    gif_frame_duration_ms: int = 450,
) -> tuple[SeedArtifacts, ...]:
    if not experiments:
        raise ValueError("At least one experiment is required to export seed artifacts")
    if storyboard_panels < 2:
        raise ValueError("storyboard_panels must be at least 2")
    if gif_frame_duration_ms <= 0:
        raise ValueError("gif_frame_duration_ms must be positive")

    from .visualization import save_metrics_plot, save_snapshot_gif, save_storyboard_plot

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    artifacts: list[SeedArtifacts] = []

    for experiment in experiments:
        seed_root = output_root / f"seed_{experiment.seed:03d}"
        seed_root.mkdir(parents=True, exist_ok=True)

        metrics_plot = save_metrics_plot(
            experiment.result,
            seed_root / f"{experiment.dataset_name}_seed_{experiment.seed:03d}_metrics.png",
            title=(
                f"T-WFC {experiment.dataset_name} seed={experiment.seed} metrics "
                f"({experiment.result.collapsed_count}/{experiment.result.parameter_count} collapsed)"
            ),
        )

        storyboard_plot: Path | None = None
        gif_path: Path | None = None
        if experiment.dataset.x_train.shape[1] == 2:
            storyboard_plot = save_storyboard_plot(
                model=experiment.model,
                dataset=experiment.dataset,
                result=experiment.result,
                output_path=seed_root / f"{experiment.dataset_name}_seed_{experiment.seed:03d}_storyboard.png",
                max_panels=storyboard_panels,
                title=(
                    f"T-WFC {experiment.dataset_name} seed={experiment.seed} storyboard "
                    f"({experiment.result.collapsed_count}/{experiment.result.parameter_count} collapsed)"
                ),
            )
            gif_path = save_snapshot_gif(
                model=experiment.model,
                dataset=experiment.dataset,
                result=experiment.result,
                output_path=seed_root / f"{experiment.dataset_name}_seed_{experiment.seed:03d}_steps.gif",
                max_frames=max_frames,
                frame_duration_ms=gif_frame_duration_ms,
                title_prefix=f"T-WFC {experiment.dataset_name} seed={experiment.seed}",
            )

        artifacts.append(
            SeedArtifacts(
                seed=experiment.seed,
                output_dir=seed_root,
                metrics_plot=metrics_plot,
                storyboard_plot=storyboard_plot,
                gif_path=gif_path,
            )
        )

    return tuple(artifacts)
