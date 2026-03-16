from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path

import numpy as np

from .batch import SeedArtifacts, SeedExperiment


def save_seed_markdown_report(
    experiments: tuple[SeedExperiment, ...] | list[SeedExperiment],
    output_path: str | Path,
    *,
    title: str | None = None,
    gallery_path: str | Path | None = None,
    seed_artifacts: dict[int, SeedArtifacts] | tuple[SeedArtifacts, ...] | list[SeedArtifacts] | None = None,
    config_summary: dict[str, object] | None = None,
) -> Path:
    if not experiments:
        raise ValueError("At least one experiment is required to write a report")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    dataset_name = experiments[0].dataset_name
    model = experiments[0].model
    shadow_test_accuracy = np.array(
        [experiment.result.final_shadow_metrics.test_accuracy for experiment in experiments],
        dtype=np.float64,
    )
    hard_test_accuracy = np.array([experiment.result.final_hard_metrics.test_accuracy for experiment in experiments], dtype=np.float64)
    shadow_test_loss = np.array(
        [experiment.result.final_shadow_metrics.test_loss for experiment in experiments],
        dtype=np.float64,
    )
    hard_test_loss = np.array([experiment.result.final_hard_metrics.test_loss for experiment in experiments], dtype=np.float64)
    accuracy_gaps = shadow_test_accuracy - hard_test_accuracy
    loss_gaps = hard_test_loss - shadow_test_loss
    rollback_counts = np.array([experiment.result.rollback_count for experiment in experiments], dtype=np.int64)
    forced_commit_counts = np.array([experiment.result.forced_commit_count for experiment in experiments], dtype=np.int64)
    max_forbidden_counts = np.array(
        [
            max(snapshot.forbidden_value_count for snapshot in experiment.result.snapshots)
            for experiment in experiments
        ],
        dtype=np.int64,
    )
    max_frontier_pressures = np.array(
        [
            max(snapshot.frontier_pressure for snapshot in experiment.result.snapshots)
            for experiment in experiments
        ],
        dtype=np.int64,
    )
    artifacts_by_seed = _index_seed_artifacts(seed_artifacts)

    best_index = int(hard_test_accuracy.argmax())
    worst_index = int(hard_test_accuracy.argmin())
    config_lines = _format_config_summary(config_summary or {})

    lines: list[str] = []
    lines.append(f"# {title or f'T-WFC {dataset_name} Seed Report'}")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Dataset: `{dataset_name}`")
    lines.append(
        f"- Model: `{model.architecture_label}` "
        f"with `{model.parameter_count}` parameters"
    )
    lines.append(f"- Seeds: `{', '.join(str(experiment.seed) for experiment in experiments)}`")
    lines.append("")
    if config_lines:
        lines.append("## Configuration")
        lines.append("")
        lines.extend(config_lines)
        lines.append("")
    lines.append("## Aggregate Summary")
    lines.append("")
    lines.append(f"- Mean shadow test accuracy: `{shadow_test_accuracy.mean():.3f}`")
    lines.append(f"- Mean hard test accuracy: `{hard_test_accuracy.mean():.3f}`")
    lines.append(f"- Std hard test accuracy: `{hard_test_accuracy.std():.3f}`")
    lines.append(f"- Mean shadow-hard test accuracy gap: `{accuracy_gaps.mean():+.3f}`")
    lines.append(f"- Mean shadow test loss: `{shadow_test_loss.mean():.4f}`")
    lines.append(f"- Mean hard test loss: `{hard_test_loss.mean():.4f}`")
    lines.append(f"- Mean hard-shadow test loss gap: `{loss_gaps.mean():+.4f}`")
    lines.append(f"- Mean rollback count: `{rollback_counts.mean():.2f}`")
    lines.append(f"- Mean forced-commit count: `{forced_commit_counts.mean():.2f}`")
    lines.append(f"- Max active forbidden values: `{int(max_forbidden_counts.max())}`")
    lines.append(f"- Max frontier pressure: `{int(max_frontier_pressures.max())}`")
    lines.append(
        f"- Best seed: `{experiments[best_index].seed}` "
        f"(hard test acc `{hard_test_accuracy[best_index]:.3f}`)"
    )
    lines.append(
        f"- Worst seed: `{experiments[worst_index].seed}` "
        f"(hard test acc `{hard_test_accuracy[worst_index]:.3f}`)"
    )
    lines.append("")
    lines.extend(
        _divergence_sections(
            experiments=experiments,
            accuracy_gaps=accuracy_gaps,
            loss_gaps=loss_gaps,
        )
    )
    lines.extend(
        _highlight_sections(
            experiments=experiments,
            best_index=best_index,
            worst_index=worst_index,
            artifacts_by_seed=artifacts_by_seed,
            output_dir=output.parent,
        )
    )

    if gallery_path is not None:
        gallery = Path(gallery_path)
        relative_gallery = _relative_path(gallery, output.parent)
        lines.append("## Gallery")
        lines.append("")
        lines.append(f"- Plot: [{relative_gallery}]({relative_gallery})")
        if gallery.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            lines.append("")
            lines.append(f"![Seed gallery]({relative_gallery})")
        lines.append("")

    lines.append("## Seed Table")
    lines.append("")
    if artifacts_by_seed:
        lines.append(
            "| Seed | Tag | Collapsed | Shadow Test Acc | Hard Test Acc | Acc Gap (S-H) | "
            "Hard Test Loss | Loss Gap (H-S) | Rollbacks | Alt Choices | Forced | Max Bans | "
            "Max Pressure | Peak Ban Focus | Latest Ban Delta | Metrics | Storyboard | GIF |"
        )
        lines.append(
            "|------|-----|-----------|-----------------|---------------|---------------|----------------|----------------|"
            "-----------|-------------|--------|----------|--------------|----------------|------------------|---------|------------|-----|"
        )
    else:
        lines.append("| Seed | Tag | Collapsed | Shadow Test Acc | Hard Test Acc | Acc Gap (S-H) | Hard Test Loss | Loss Gap (H-S) | Rollbacks | Alt Choices | Forced | Max Bans | Max Pressure | Peak Ban Focus | Latest Ban Delta |")
        lines.append("|------|-----|-----------|-----------------|---------------|---------------|----------------|----------------|-----------|-------------|--------|----------|--------------|----------------|------------------|")
    for experiment in experiments:
        result = experiment.result
        accuracy_gap = _shadow_hard_accuracy_gap(experiment)
        loss_gap = _shadow_hard_loss_gap(experiment)
        seed_tag = _seed_tag(experiment=experiment, experiments=experiments, best_index=best_index, worst_index=worst_index)
        row = (
            "| "
            f"{experiment.seed} | "
            f"{seed_tag} | "
            f"{result.collapsed_count}/{result.parameter_count} | "
            f"{result.final_shadow_metrics.test_accuracy:.3f} | "
            f"{result.final_hard_metrics.test_accuracy:.3f} | "
            f"{accuracy_gap:+.3f} | "
            f"{result.final_hard_metrics.test_loss:.4f} | "
            f"{loss_gap:+.4f} | "
            f"{result.rollback_count} | "
            f"{result.backtrack_count} | "
            f"{result.forced_commit_count} | "
            f"{max(snapshot.forbidden_value_count for snapshot in result.snapshots)} | "
            f"{max(snapshot.frontier_pressure for snapshot in result.snapshots)} | "
            f"{_experiment_peak_ban_summary(experiment, include_values=False)} | "
            f"{_experiment_latest_ban_delta_summary(experiment)} |"
        )
        if artifacts_by_seed:
            artifact = artifacts_by_seed.get(experiment.seed)
            row += (
                f" {_artifact_link(artifact.metrics_plot if artifact else None, output.parent, 'metrics')} |"
                f" {_artifact_link(artifact.storyboard_plot if artifact else None, output.parent, 'storyboard')} |"
                f" {_artifact_link(artifact.gif_path if artifact else None, output.parent, 'gif')} |"
            )
        lines.append(row)

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `Acc Gap (S-H)` is `shadow test accuracy - hard test accuracy`, so positive values mean shadow stayed ahead.")
    lines.append("- `Loss Gap (H-S)` is `hard test loss - shadow test loss`, so positive values mean hard stayed worse.")
    lines.append("- `Max Bans` counts the largest number of active forbidden candidate values seen during the run.")
    lines.append("- `Max Pressure` counts the highest rollback pressure observed at a single search frontier before a commit.")
    if artifacts_by_seed:
        lines.append("- `Storyboard` and `GIF` links appear only for 2D datasets where decision-surface rendering is available.")

    if artifacts_by_seed:
        lines.append("")
        lines.append("## Seed Drilldown")
        lines.append("")
        for experiment in _ordered_drilldown_experiments(experiments, best_index=best_index, worst_index=worst_index):
            artifact = artifacts_by_seed.get(experiment.seed)
            if artifact is None:
                continue

            seed_tag = _seed_tag(experiment=experiment, experiments=experiments, best_index=best_index, worst_index=worst_index)
            heading_suffix = f" [{seed_tag}]" if seed_tag != "-" else ""
            lines.append(f"### Seed {experiment.seed}{heading_suffix}")
            lines.append("")
            lines.append(f"- Shadow vs hard: `{_experiment_divergence_summary(experiment)}`")
            lines.append(f"- Peak Ban Focus: `{_experiment_peak_ban_summary(experiment, include_values=True)}`")
            lines.append(f"- Latest Ban Delta: `{_experiment_latest_ban_delta_summary(experiment, short=False)}`")
            lines.append(f"- Metrics: {_artifact_link(artifact.metrics_plot, output.parent, 'open')}")
            lines.append(f"- Storyboard: {_artifact_link(artifact.storyboard_plot, output.parent, 'open')}")
            lines.append(f"- GIF: {_artifact_link(artifact.gif_path, output.parent, 'open')}")
            lines.append("")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _format_config_summary(config_summary: dict[str, object]) -> list[str]:
    if not config_summary:
        return []

    lines: list[str] = []
    for key, value in config_summary.items():
        lines.append(f"- `{key}`: `{value}`")
    return lines


def _relative_path(target: Path, base_dir: Path) -> str:
    return os.path.relpath(target, start=base_dir)


def _index_seed_artifacts(
    seed_artifacts: dict[int, SeedArtifacts] | tuple[SeedArtifacts, ...] | list[SeedArtifacts] | None,
) -> dict[int, SeedArtifacts]:
    if seed_artifacts is None:
        return {}
    if isinstance(seed_artifacts, dict):
        return dict(seed_artifacts)
    return {artifact.seed: artifact for artifact in seed_artifacts}


def _artifact_link(path: Path | None, base_dir: Path, label: str) -> str:
    if path is None:
        return "-"
    relative = _relative_path(path, base_dir)
    return f"[{label}]({relative})"


def _highlight_sections(
    *,
    experiments: tuple[SeedExperiment, ...] | list[SeedExperiment],
    best_index: int,
    worst_index: int,
    artifacts_by_seed: dict[int, SeedArtifacts],
    output_dir: Path,
) -> list[str]:
    sections: list[str] = ["## Highlights", ""]
    highlight_specs: list[tuple[str, SeedExperiment]] = [("Best Seed", experiments[best_index])]
    if worst_index != best_index:
        highlight_specs.append(("Worst Seed", experiments[worst_index]))

    for title, experiment in highlight_specs:
        sections.append(f"### {title}: `{experiment.seed}`")
        sections.append("")
        sections.append(
            f"- Hard test accuracy/loss: `{experiment.result.final_hard_metrics.test_accuracy:.3f}` / "
            f"`{experiment.result.final_hard_metrics.test_loss:.4f}`"
        )
        sections.append(f"- Shadow vs hard: `{_experiment_divergence_summary(experiment)}`")
        sections.append(
            f"- Search pressure: `rb={experiment.result.rollback_count}` `alt={experiment.result.backtrack_count}` "
            f"`forced={experiment.result.forced_commit_count}` "
            f"`max_bans={max(snapshot.forbidden_value_count for snapshot in experiment.result.snapshots)}` "
            f"`max_pressure={max(snapshot.frontier_pressure for snapshot in experiment.result.snapshots)}`"
        )
        sections.append(f"- Peak ban focus: `{_experiment_peak_ban_summary(experiment, include_values=True)}`")
        sections.append(f"- Latest ban delta: `{_experiment_latest_ban_delta_summary(experiment, short=False)}`")
        artifact = artifacts_by_seed.get(experiment.seed)
        if artifact is not None:
            sections.append(
                "- Drilldown: "
                f"{_artifact_link(artifact.metrics_plot, output_dir, 'metrics')} | "
                f"{_artifact_link(artifact.storyboard_plot, output_dir, 'storyboard')} | "
                f"{_artifact_link(artifact.gif_path, output_dir, 'gif')}"
            )
            sections.extend(
                _highlight_preview_lines(
                    title=title,
                    artifact=artifact,
                    output_dir=output_dir,
                )
            )
        sections.append("")

    return sections


def _highlight_preview_lines(
    *,
    title: str,
    artifact: SeedArtifacts,
    output_dir: Path,
) -> list[str]:
    lines: list[str] = []
    if artifact.storyboard_plot is not None:
        relative_storyboard = _relative_path(artifact.storyboard_plot, output_dir)
        lines.append("")
        lines.append(f"![{title} preview: storyboard]({relative_storyboard})")

    relative_metrics = _relative_path(artifact.metrics_plot, output_dir)
    lines.append("")
    lines.append(f"![{title} preview: metrics]({relative_metrics})")
    return lines


def _divergence_sections(
    *,
    experiments: tuple[SeedExperiment, ...] | list[SeedExperiment],
    accuracy_gaps: np.ndarray,
    loss_gaps: np.ndarray,
) -> list[str]:
    sections: list[str] = ["## Shadow vs Hard Divergence", ""]
    matched_or_exceeded_count = int(np.count_nonzero(accuracy_gaps <= 1e-12))
    largest_shadow_lead_index = int(accuracy_gaps.argmax())
    highest_loss_penalty_index = int(loss_gaps.argmax())
    most_aligned_index = int(np.abs(accuracy_gaps).argmin())

    sections.append(
        f"- Seeds where hard test accuracy matched or exceeded shadow: "
        f"`{matched_or_exceeded_count}/{len(experiments)}`"
    )
    sections.append(
        f"- Largest shadow accuracy lead: "
        f"`{_experiment_accuracy_gap_summary(experiments[largest_shadow_lead_index])}`"
    )
    if float(accuracy_gaps.min()) < -1e-12:
        sections.append(
            f"- Largest hard accuracy lead: "
            f"`{_experiment_accuracy_gap_summary(experiments[int(accuracy_gaps.argmin())])}`"
        )
    else:
        sections.append(
            "- Largest hard accuracy lead: "
            "`none` (hard never exceeded shadow on test accuracy in this batch)"
        )
    sections.append(
        f"- Largest hard loss penalty: "
        f"`{_experiment_loss_gap_summary(experiments[highest_loss_penalty_index])}`"
    )
    if float(loss_gaps.min()) < -1e-12:
        sections.append(
            f"- Largest hard loss gain: "
            f"`{_experiment_loss_gap_summary(experiments[int(loss_gaps.argmin())])}`"
        )
    else:
        sections.append(
            "- Largest hard loss gain: "
            "`none` (hard never beat shadow on test loss in this batch)"
        )
    sections.append(
        f"- Most aligned seed: "
        f"`{_experiment_divergence_summary(experiments[most_aligned_index])}`"
    )
    sections.append("")
    return sections


def _seed_tag(
    *,
    experiment: SeedExperiment,
    experiments: tuple[SeedExperiment, ...] | list[SeedExperiment],
    best_index: int,
    worst_index: int,
) -> str:
    best_seed = experiments[best_index].seed
    worst_seed = experiments[worst_index].seed
    if experiment.seed == best_seed and experiment.seed == worst_seed:
        return "BEST/WORST"
    if experiment.seed == best_seed:
        return "BEST"
    if experiment.seed == worst_seed:
        return "WORST"
    return "-"


def _ordered_drilldown_experiments(
    experiments: tuple[SeedExperiment, ...] | list[SeedExperiment],
    *,
    best_index: int,
    worst_index: int,
) -> tuple[SeedExperiment, ...]:
    priority: list[SeedExperiment] = []
    seen: set[int] = set()
    for index in (best_index, worst_index):
        experiment = experiments[index]
        if experiment.seed in seen:
            continue
        priority.append(experiment)
        seen.add(experiment.seed)

    for experiment in experiments:
        if experiment.seed in seen:
            continue
        priority.append(experiment)
        seen.add(experiment.seed)

    return tuple(priority)


def _shadow_hard_accuracy_gap(experiment: SeedExperiment) -> float:
    return experiment.result.final_shadow_metrics.test_accuracy - experiment.result.final_hard_metrics.test_accuracy


def _shadow_hard_loss_gap(experiment: SeedExperiment) -> float:
    return experiment.result.final_hard_metrics.test_loss - experiment.result.final_shadow_metrics.test_loss


def _experiment_accuracy_gap_summary(experiment: SeedExperiment) -> str:
    return (
        f"seed {experiment.seed}: shadow {experiment.result.final_shadow_metrics.test_accuracy:.3f} "
        f"vs hard {experiment.result.final_hard_metrics.test_accuracy:.3f} "
        f"(gap {_shadow_hard_accuracy_gap(experiment):+.3f})"
    )


def _experiment_loss_gap_summary(experiment: SeedExperiment) -> str:
    return (
        f"seed {experiment.seed}: shadow {experiment.result.final_shadow_metrics.test_loss:.4f} "
        f"vs hard {experiment.result.final_hard_metrics.test_loss:.4f} "
        f"(gap {_shadow_hard_loss_gap(experiment):+.4f})"
    )


def _experiment_divergence_summary(experiment: SeedExperiment) -> str:
    return (
        f"acc gap {_shadow_hard_accuracy_gap(experiment):+.3f} "
        f"(shadow {experiment.result.final_shadow_metrics.test_accuracy:.3f}, "
        f"hard {experiment.result.final_hard_metrics.test_accuracy:.3f}); "
        f"loss gap {_shadow_hard_loss_gap(experiment):+.4f} "
        f"(shadow {experiment.result.final_shadow_metrics.test_loss:.4f}, "
        f"hard {experiment.result.final_hard_metrics.test_loss:.4f})"
    )


def _experiment_peak_ban_summary(
    experiment: SeedExperiment,
    *,
    include_values: bool,
) -> str:
    peak_snapshot = None
    peak_entry = None

    for snapshot in experiment.result.snapshots:
        for entry in snapshot.forbidden_entries:
            if peak_entry is None or entry.ban_count > peak_entry.ban_count or (
                entry.ban_count == peak_entry.ban_count and snapshot.step < peak_snapshot.step
            ):
                peak_snapshot = snapshot
                peak_entry = entry

    if peak_entry is None or peak_snapshot is None:
        return "clean"

    if include_values:
        value_text = ",".join(f"{value:.1f}" for value in peak_entry.banned_values)
        return f"{peak_entry.parameter_name}=[{value_text}] @ s{peak_snapshot.step:02d}"
    return f"{peak_entry.parameter_name}({peak_entry.ban_count}) @ s{peak_snapshot.step:02d}"


def _experiment_latest_ban_delta_summary(
    experiment: SeedExperiment,
    *,
    short: bool = True,
) -> str:
    for snapshot in reversed(experiment.result.snapshots):
        if not snapshot.forbidden_delta_labels:
            continue
        prefix = f"s{snapshot.step:02d}: "
        summary = "; ".join(snapshot.forbidden_delta_labels)
        text = prefix + summary
        return _truncate_text(text, limit=40 if short else 96)
    return "none"


def _truncate_text(value: str, *, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(limit - 3, 0)] + "..."
