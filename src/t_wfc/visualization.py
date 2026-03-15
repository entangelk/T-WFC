from __future__ import annotations

from io import BytesIO
import os
import tempfile
from pathlib import Path

import numpy as np

from .baseline import SGDBaselineResult
from .batch import SeedExperiment
from .data import DatasetSplit
from .model import ToyMLP
from .trainer import ExperimentResult, ExperimentSnapshot


def save_metrics_plot(
    result: ExperimentResult,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    plt, _ = _load_matplotlib()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    snapshots = result.snapshots
    steps = np.array([snapshot.step for snapshot in snapshots], dtype=np.int64)
    collapsed_counts = np.array([snapshot.collapsed_count for snapshot in snapshots], dtype=np.int64)

    shadow_train_loss = np.array([snapshot.shadow_metrics.train_loss for snapshot in snapshots], dtype=np.float64)
    shadow_test_loss = np.array([snapshot.shadow_metrics.test_loss for snapshot in snapshots], dtype=np.float64)
    hard_train_loss = np.array([snapshot.hard_metrics.train_loss for snapshot in snapshots], dtype=np.float64)
    hard_test_loss = np.array([snapshot.hard_metrics.test_loss for snapshot in snapshots], dtype=np.float64)

    shadow_train_accuracy = np.array([snapshot.shadow_metrics.train_accuracy for snapshot in snapshots], dtype=np.float64)
    shadow_test_accuracy = np.array([snapshot.shadow_metrics.test_accuracy for snapshot in snapshots], dtype=np.float64)
    hard_train_accuracy = np.array([snapshot.hard_metrics.train_accuracy for snapshot in snapshots], dtype=np.float64)
    hard_test_accuracy = np.array([snapshot.hard_metrics.test_accuracy for snapshot in snapshots], dtype=np.float64)

    figure, axes = plt.subplots(3, 1, figsize=(10.5, 9.2), sharex=True, constrained_layout=True)

    axes[0].plot(steps, shadow_train_loss, color="#1f77b4", linewidth=2.0, label="shadow train")
    axes[0].plot(steps, shadow_test_loss, color="#1f77b4", linewidth=2.0, linestyle="--", label="shadow test")
    axes[0].plot(steps, hard_train_loss, color="#d62728", linewidth=2.0, label="hard train")
    axes[0].plot(steps, hard_test_loss, color="#d62728", linewidth=2.0, linestyle="--", label="hard test")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right", ncol=2)

    axes[1].plot(steps, shadow_train_accuracy, color="#1f77b4", linewidth=2.0, label="shadow train")
    axes[1].plot(steps, shadow_test_accuracy, color="#1f77b4", linewidth=2.0, linestyle="--", label="shadow test")
    axes[1].plot(steps, hard_train_accuracy, color="#d62728", linewidth=2.0, label="hard train")
    axes[1].plot(steps, hard_test_accuracy, color="#d62728", linewidth=2.0, linestyle="--", label="hard test")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="lower right", ncol=2)

    axes[2].plot(steps, collapsed_counts, color="#111111", linewidth=2.2, marker="o", markersize=4.5, label="collapsed")
    axes[2].set_ylabel("Collapsed")
    axes[2].set_xlabel("Committed step")
    axes[2].grid(alpha=0.25)

    _plot_event_markers(
        axes=(axes[0], axes[1], axes[2]),
        result=result,
        y_values=(
            shadow_test_loss,
            shadow_test_accuracy,
            collapsed_counts,
        ),
    )
    _annotate_ban_identity(axis=axes[2], snapshots=snapshots[1:])

    collapsed_ratio = f"{result.collapsed_count}/{result.parameter_count}"
    figure.suptitle(title or f"T-WFC metrics timeline ({collapsed_ratio} collapsed)")
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def save_baseline_metrics_comparison_plot(
    result: ExperimentResult,
    baseline_result: SGDBaselineResult,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    plt, _ = _load_matplotlib()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    snapshots = result.snapshots
    steps = np.array([snapshot.step for snapshot in snapshots], dtype=np.int64)
    baseline_epochs = np.arange(len(baseline_result.history), dtype=np.int64)

    shadow_test_loss = np.array([snapshot.shadow_metrics.test_loss for snapshot in snapshots], dtype=np.float64)
    hard_test_loss = np.array([snapshot.hard_metrics.test_loss for snapshot in snapshots], dtype=np.float64)
    shadow_test_accuracy = np.array([snapshot.shadow_metrics.test_accuracy for snapshot in snapshots], dtype=np.float64)
    hard_test_accuracy = np.array([snapshot.hard_metrics.test_accuracy for snapshot in snapshots], dtype=np.float64)

    baseline_train_loss = np.array([metrics.train_loss for metrics in baseline_result.history], dtype=np.float64)
    baseline_test_loss = np.array([metrics.test_loss for metrics in baseline_result.history], dtype=np.float64)
    baseline_train_accuracy = np.array([metrics.train_accuracy for metrics in baseline_result.history], dtype=np.float64)
    baseline_test_accuracy = np.array([metrics.test_accuracy for metrics in baseline_result.history], dtype=np.float64)

    figure, axes = plt.subplots(2, 2, figsize=(14.5, 9.0), constrained_layout=True)

    axes[0, 0].plot(steps, shadow_test_loss, color="#1f77b4", linewidth=2.2, label="T-WFC shadow test")
    axes[0, 0].plot(steps, hard_test_loss, color="#d62728", linewidth=2.2, label="T-WFC hard test")
    axes[0, 0].axhline(
        baseline_result.final_metrics.test_loss,
        color="#0f766e",
        linewidth=2.0,
        linestyle="--",
        label="SGD final test",
    )
    axes[0, 0].set_title("Loss: T-WFC timeline vs SGD endpoint")
    axes[0, 0].set_xlabel("Committed step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(loc="upper right")

    axes[0, 1].plot(steps, shadow_test_accuracy, color="#1f77b4", linewidth=2.2, label="T-WFC shadow test")
    axes[0, 1].plot(steps, hard_test_accuracy, color="#d62728", linewidth=2.2, label="T-WFC hard test")
    axes[0, 1].axhline(
        baseline_result.final_metrics.test_accuracy,
        color="#0f766e",
        linewidth=2.0,
        linestyle="--",
        label="SGD final test",
    )
    axes[0, 1].set_title("Accuracy: T-WFC timeline vs SGD endpoint")
    axes[0, 1].set_xlabel("Committed step")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(loc="lower right")

    axes[1, 0].plot(baseline_epochs, baseline_train_loss, color="#0f766e", linewidth=2.2, label="SGD train")
    axes[1, 0].plot(baseline_epochs, baseline_test_loss, color="#14b8a6", linewidth=2.2, linestyle="--", label="SGD test")
    axes[1, 0].axhline(result.final_shadow_metrics.test_loss, color="#1f77b4", linewidth=1.8, linestyle=":", label="T-WFC shadow final")
    axes[1, 0].axhline(result.final_hard_metrics.test_loss, color="#d62728", linewidth=1.8, linestyle=":", label="T-WFC hard final")
    axes[1, 0].set_title("Loss: SGD timeline vs T-WFC endpoint")
    axes[1, 0].set_xlabel("SGD epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(loc="upper right")

    axes[1, 1].plot(baseline_epochs, baseline_train_accuracy, color="#0f766e", linewidth=2.2, label="SGD train")
    axes[1, 1].plot(baseline_epochs, baseline_test_accuracy, color="#14b8a6", linewidth=2.2, linestyle="--", label="SGD test")
    axes[1, 1].axhline(result.final_shadow_metrics.test_accuracy, color="#1f77b4", linewidth=1.8, linestyle=":", label="T-WFC shadow final")
    axes[1, 1].axhline(result.final_hard_metrics.test_accuracy, color="#d62728", linewidth=1.8, linestyle=":", label="T-WFC hard final")
    axes[1, 1].set_title("Accuracy: SGD timeline vs T-WFC endpoint")
    axes[1, 1].set_xlabel("SGD epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].set_ylim(-0.02, 1.02)
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(loc="lower right")

    _plot_event_markers(
        axes=(axes[0, 0], axes[0, 1]),
        result=result,
        y_values=(shadow_test_loss, shadow_test_accuracy),
    )
    _annotate_ban_identity(axis=axes[0, 1], snapshots=snapshots[1:])

    figure.suptitle(
        title
        or (
            "T-WFC vs SGD metrics comparison "
            f"(hard acc {result.final_hard_metrics.test_accuracy:.3f} vs {baseline_result.final_metrics.test_accuracy:.3f})"
        )
    )
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def save_baseline_comparison_plot(
    model: ToyMLP,
    dataset: DatasetSplit,
    result: ExperimentResult,
    baseline_result: SGDBaselineResult,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    if dataset.x_train.shape[1] != 2:
        raise ValueError("Baseline comparison surfaces currently support only 2D datasets")

    plt, ListedColormap = _load_matplotlib()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(19.2, 8.8), constrained_layout=True)
    grid = figure.add_gridspec(2, 4, height_ratios=(1.0, 1.55))

    loss_axis = figure.add_subplot(grid[0, :2])
    accuracy_axis = figure.add_subplot(grid[0, 2:])

    steps = np.array([snapshot.step for snapshot in result.snapshots], dtype=np.int64)
    shadow_test_loss = np.array([snapshot.shadow_metrics.test_loss for snapshot in result.snapshots], dtype=np.float64)
    hard_test_loss = np.array([snapshot.hard_metrics.test_loss for snapshot in result.snapshots], dtype=np.float64)
    shadow_test_accuracy = np.array([snapshot.shadow_metrics.test_accuracy for snapshot in result.snapshots], dtype=np.float64)
    hard_test_accuracy = np.array([snapshot.hard_metrics.test_accuracy for snapshot in result.snapshots], dtype=np.float64)

    loss_axis.plot(steps, shadow_test_loss, color="#1f77b4", linewidth=2.2, label="T-WFC shadow")
    loss_axis.plot(steps, hard_test_loss, color="#d62728", linewidth=2.2, label="T-WFC hard")
    loss_axis.axhline(baseline_result.final_metrics.test_loss, color="#0f766e", linewidth=2.0, linestyle="--", label="SGD final")
    loss_axis.set_title("Test Loss")
    loss_axis.set_xlabel("Committed step")
    loss_axis.set_ylabel("Loss")
    loss_axis.grid(alpha=0.25)
    loss_axis.legend(loc="upper right")

    accuracy_axis.plot(steps, shadow_test_accuracy, color="#1f77b4", linewidth=2.2, label="T-WFC shadow")
    accuracy_axis.plot(steps, hard_test_accuracy, color="#d62728", linewidth=2.2, label="T-WFC hard")
    accuracy_axis.axhline(
        baseline_result.final_metrics.test_accuracy,
        color="#0f766e",
        linewidth=2.0,
        linestyle="--",
        label="SGD final",
    )
    accuracy_axis.set_title("Test Accuracy")
    accuracy_axis.set_xlabel("Committed step")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.set_ylim(-0.02, 1.02)
    accuracy_axis.grid(alpha=0.25)
    accuracy_axis.legend(loc="lower right")

    _plot_event_markers(
        axes=(loss_axis, accuracy_axis),
        result=result,
        y_values=(shadow_test_loss, shadow_test_accuracy),
    )

    grid_x, grid_y, grid_features = _build_grid(dataset)
    surface_cmap = ListedColormap(["#dceaf7", "#fde2d0", "#e3f2df", "#f7e1f3"])
    point_colors = np.array(["#1f77b4", "#d62728", "#2ca02c", "#9467bd"])
    panel_specs = (
        ("Initial Shadow", result.initial_shadow_weights, result.initial_shadow_metrics.test_accuracy, result.initial_shadow_metrics.test_loss, "#4b5563"),
        ("Final Shadow", result.shadow_weights, result.final_shadow_metrics.test_accuracy, result.final_shadow_metrics.test_loss, "#1d4ed8"),
        ("Final Hard", result.hard_weights, result.final_hard_metrics.test_accuracy, result.final_hard_metrics.test_loss, "#dc2626"),
        ("SGD Final", baseline_result.weights, baseline_result.final_metrics.test_accuracy, baseline_result.final_metrics.test_loss, "#0f766e"),
    )

    for column_index, (panel_title, weights, test_accuracy, test_loss, accent_color) in enumerate(panel_specs):
        axis = figure.add_subplot(grid[1, column_index])
        _render_panel(
            axis=axis,
            model=model,
            dataset=dataset,
            weights=weights,
            grid_x=grid_x,
            grid_y=grid_y,
            grid_features=grid_features,
            surface_cmap=surface_cmap,
            point_colors=point_colors,
        )
        axis.set_title(f"{panel_title}\nacc={test_accuracy:.3f} loss={test_loss:.4f}")
        axis.set_xlabel("x0")
        axis.set_ylabel("x1")
        for spine in axis.spines.values():
            spine.set_color(accent_color)
            spine.set_linewidth(2.1)
        if panel_title == "Final Hard":
            _draw_ban_overlay(axis, result.snapshots[-1], anchor_x=0.98, anchor_y=0.02, fontsize=7.4)

    figure.suptitle(
        title
        or (
            "T-WFC vs SGD boundary comparison "
            f"({result.collapsed_count}/{result.parameter_count} collapsed, "
            f"hard acc {result.final_hard_metrics.test_accuracy:.3f} vs SGD {baseline_result.final_metrics.test_accuracy:.3f})"
        )
    )
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def save_baseline_comparison_gif(
    model: ToyMLP,
    dataset: DatasetSplit,
    result: ExperimentResult,
    baseline_result: SGDBaselineResult,
    output_path: str | Path,
    max_frames: int = 0,
    frame_duration_ms: int = 450,
    title_prefix: str | None = None,
) -> Path:
    if dataset.x_train.shape[1] != 2:
        raise ValueError("Baseline comparison GIF currently supports only 2D datasets")
    if frame_duration_ms <= 0:
        raise ValueError("frame_duration_ms must be positive")

    Image = _load_pillow()
    plt, ListedColormap = _load_matplotlib()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    snapshots = result.snapshots if max_frames <= 0 else _select_snapshots(result.snapshots, max_panels=max_frames)
    grid_x, grid_y, grid_features = _build_grid(dataset)
    surface_cmap = ListedColormap(["#dceaf7", "#fde2d0", "#e3f2df", "#f7e1f3"])
    point_colors = np.array(["#1f77b4", "#d62728", "#2ca02c", "#9467bd"])
    frames = []

    for snapshot in snapshots:
        figure, axes = plt.subplots(1, 3, figsize=(15.8, 5.0), constrained_layout=True)
        panels = (
            ("T-WFC Shadow", snapshot.shadow_weights, snapshot.shadow_metrics.test_accuracy, snapshot.shadow_metrics.test_loss, "#1d4ed8"),
            ("T-WFC Hard", snapshot.hard_weights, snapshot.hard_metrics.test_accuracy, snapshot.hard_metrics.test_loss, "#dc2626"),
            ("SGD Final", baseline_result.weights, baseline_result.final_metrics.test_accuracy, baseline_result.final_metrics.test_loss, "#0f766e"),
        )

        for axis, (panel_title, weights, test_accuracy, test_loss, accent_color) in zip(axes, panels, strict=True):
            _render_panel(
                axis=axis,
                model=model,
                dataset=dataset,
                weights=weights,
                grid_x=grid_x,
                grid_y=grid_y,
                grid_features=grid_features,
                surface_cmap=surface_cmap,
                point_colors=point_colors,
            )
            axis.set_title(f"{panel_title}\nacc={test_accuracy:.3f} loss={test_loss:.4f}")
            axis.set_xlabel("x0")
            axis.set_ylabel("x1")
            for spine in axis.spines.values():
                spine.set_color(accent_color)
                spine.set_linewidth(2.0)

        style = _snapshot_style(snapshot=snapshot, result=result)
        _draw_badge_stack(
            axes[0],
            _snapshot_badges(snapshot=snapshot, result=result),
            anchor_x=0.02,
            anchor_y=0.86,
            vertical_gap=0.12,
            fontsize=8.2,
        )
        axes[2].text(
            0.02,
            0.98,
            "SGD",
            ha="left",
            va="top",
            transform=axes[2].transAxes,
            fontsize=9.0,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#0f766e", edgecolor="none", alpha=0.95),
        )
        _draw_ban_overlay(axes[1], snapshot, anchor_x=0.98, anchor_y=0.02, fontsize=7.4)
        figure.suptitle(
            f"{title_prefix or 'T-WFC vs SGD'} | step {snapshot.step:02d} | "
            f"hard acc {snapshot.hard_metrics.test_accuracy:.3f} vs SGD {baseline_result.final_metrics.test_accuracy:.3f}"
        )
        figure.text(
            0.5,
            0.015,
            (
                f"collapsed={snapshot.collapsed_count}/{result.parameter_count} | "
                f"shadow loss={snapshot.shadow_metrics.test_loss:.4f} | "
                f"hard loss={snapshot.hard_metrics.test_loss:.4f} | "
                f"sgd loss={baseline_result.final_metrics.test_loss:.4f} | "
                f"rb={snapshot.rollback_count} alt={snapshot.backtrack_count} forced={snapshot.forced_commit_count}"
            ),
            ha="center",
            va="bottom",
            fontsize=9.2,
        )

        with BytesIO() as buffer:
            figure.savefig(buffer, format="png", dpi=145)
            buffer.seek(0)
            frame = Image.open(buffer).convert("RGBA").copy()
        plt.close(figure)
        frames.append(frame)

    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        disposal=2,
    )
    for frame in frames:
        frame.close()
    return output


def save_seed_gallery_plot(
    experiments: tuple[SeedExperiment, ...] | list[SeedExperiment],
    output_path: str | Path,
    title: str | None = None,
    columns: int = 3,
) -> Path:
    if not experiments:
        raise ValueError("At least one experiment is required for a seed gallery")
    if columns < 1:
        raise ValueError("columns must be at least 1")

    plt, ListedColormap = _load_matplotlib()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    all_2d = all(experiment.dataset.x_train.shape[1] == 2 for experiment in experiments)
    seeds = [experiment.seed for experiment in experiments]
    x_positions = np.arange(len(experiments), dtype=np.float64)
    shadow_test_accuracy = np.array(
        [experiment.result.final_shadow_metrics.test_accuracy for experiment in experiments],
        dtype=np.float64,
    )
    hard_test_accuracy = np.array(
        [experiment.result.final_hard_metrics.test_accuracy for experiment in experiments],
        dtype=np.float64,
    )
    rollback_counts = np.array([experiment.result.rollback_count for experiment in experiments], dtype=np.int64)
    forced_commit_counts = np.array([experiment.result.forced_commit_count for experiment in experiments], dtype=np.int64)
    max_forbidden_counts = np.array(
        [max(snapshot.forbidden_value_count for snapshot in experiment.result.snapshots) for experiment in experiments],
        dtype=np.int64,
    )
    max_frontier_pressures = np.array(
        [max(snapshot.frontier_pressure for snapshot in experiment.result.snapshots) for experiment in experiments],
        dtype=np.int64,
    )

    gallery_columns = min(max(columns, 1), len(experiments))
    gallery_rows = (len(experiments) + gallery_columns - 1) // gallery_columns

    if all_2d:
        figure = plt.figure(
            figsize=(4.5 * gallery_columns, 5.2 + 4.2 * gallery_rows),
            constrained_layout=True,
        )
        grid = figure.add_gridspec(
            gallery_rows + 2,
            gallery_columns,
            height_ratios=(1.0, 1.2) + tuple(1.65 for _ in range(gallery_rows)),
        )
        panel_row_offset = 2
    else:
        figure = plt.figure(
            figsize=(4.8 * gallery_columns, 6.6),
            constrained_layout=True,
        )
        grid = figure.add_gridspec(3, 1, height_ratios=(1.0, 1.25, 1.1))
        panel_row_offset = 99

    accuracy_axis = figure.add_subplot(grid[0, :])
    pressure_axis = figure.add_subplot(grid[1, :])

    accuracy_axis.bar(x_positions - 0.18, shadow_test_accuracy, width=0.34, color="#93c5fd", label="shadow test acc")
    accuracy_axis.bar(x_positions + 0.18, hard_test_accuracy, width=0.34, color="#1d4ed8", label="hard test acc")
    accuracy_axis.set_ylabel("Test Accuracy")
    accuracy_axis.set_ylim(0.0, 1.05)
    accuracy_axis.set_xticks(x_positions, [f"s{seed}" for seed in seeds])
    accuracy_axis.grid(alpha=0.25, axis="y")
    accuracy_axis.legend(loc="lower right")

    pressure_axis.bar(x_positions - 0.27, rollback_counts, width=0.18, color="#991b1b", label="rollbacks")
    pressure_axis.bar(x_positions - 0.09, forced_commit_counts, width=0.18, color="#dc2626", label="forced")
    pressure_axis.bar(x_positions + 0.09, max_forbidden_counts, width=0.18, color="#7c3aed", label="max bans")
    pressure_axis.bar(x_positions + 0.27, max_frontier_pressures, width=0.18, color="#db2777", label="max pressure")
    pressure_axis.set_ylabel("Search Pressure")
    pressure_axis.set_xticks(x_positions, [f"s{seed}" for seed in seeds])
    pressure_axis.grid(alpha=0.25, axis="y")
    pressure_axis.legend(loc="upper right", ncol=4)

    if all_2d:
        surface_cmap = ListedColormap(["#dceaf7", "#fde2d0", "#e3f2df", "#f7e1f3"])
        point_colors = np.array(["#1f77b4", "#d62728", "#2ca02c", "#9467bd"])

        for experiment_index, experiment in enumerate(experiments):
            row_index = experiment_index // gallery_columns
            column_index = experiment_index % gallery_columns
            axis = figure.add_subplot(grid[panel_row_offset + row_index, column_index])
            grid_x, grid_y, grid_features = _build_grid(experiment.dataset)
            _render_panel(
                axis=axis,
                model=experiment.model,
                dataset=experiment.dataset,
                weights=experiment.result.hard_weights,
                grid_x=grid_x,
                grid_y=grid_y,
                grid_features=grid_features,
                surface_cmap=surface_cmap,
                point_colors=point_colors,
            )

            summary_style = _experiment_summary_style(experiment)
            _style_snapshot_axes((axis,), summary_style)
            axis.set_title(
                f"seed={experiment.seed}\n"
                f"hard acc={experiment.result.final_hard_metrics.test_accuracy:.3f} "
                f"loss={experiment.result.final_hard_metrics.test_loss:.4f}"
            )
            axis.set_xlabel("x0")
            axis.set_ylabel("x1")
            axis.text(
                0.02,
                0.98,
                summary_style["label"],
                ha="left",
                va="top",
                transform=axis.transAxes,
                fontsize=8.8,
                fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=summary_style["badge"], edgecolor="none", alpha=0.95),
            )
            _draw_badge_stack(
                axis,
                _experiment_summary_badges(experiment),
                anchor_x=0.02,
                anchor_y=0.85,
                vertical_gap=0.115,
                fontsize=7.2,
            )
            _draw_experiment_peak_ban_overlay(axis, experiment)
            axis.text(
                0.5,
                -0.2,
                _experiment_panel_footer(experiment),
                ha="center",
                va="top",
                transform=axis.transAxes,
                fontsize=8.0,
            )

        total_panels = gallery_rows * gallery_columns
        for unused_index in range(len(experiments), total_panels):
            row_index = unused_index // gallery_columns
            column_index = unused_index % gallery_columns
            axis = figure.add_subplot(grid[panel_row_offset + row_index, column_index])
            axis.axis("off")
    else:
        summary_axis = figure.add_subplot(grid[2, 0])
        summary_axis.axis("off")
        lines = ["Seed summary"]
        for experiment in experiments:
            lines.append(
                f"seed={experiment.seed} "
                f"hard_acc={experiment.result.final_hard_metrics.test_accuracy:.3f} "
                f"rb={experiment.result.rollback_count} "
                f"forced={experiment.result.forced_commit_count} "
                f"bans<={max(snapshot.forbidden_value_count for snapshot in experiment.result.snapshots)} "
                f"press<={max(snapshot.frontier_pressure for snapshot in experiment.result.snapshots)}"
            )
        summary_axis.text(
            0.01,
            0.98,
            "\n".join(lines),
            ha="left",
            va="top",
            family="monospace",
            fontsize=9.4,
        )

    dataset_name = experiments[0].dataset_name
    figure.suptitle(title or f"T-WFC {dataset_name} seed gallery ({len(experiments)} runs)")
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def save_experiment_plot(
    model: ToyMLP,
    dataset: DatasetSplit,
    result: ExperimentResult,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    if dataset.x_train.shape[1] != 2:
        raise ValueError("Visualization currently supports only 2D datasets")

    plt, ListedColormap = _load_matplotlib()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    panels = (
        (
            "Initial Shadow",
            result.initial_shadow_weights,
            result.initial_shadow_metrics.test_accuracy,
            result.initial_shadow_metrics.test_loss,
        ),
        (
            "Final Shadow",
            result.shadow_weights,
            result.final_shadow_metrics.test_accuracy,
            result.final_shadow_metrics.test_loss,
        ),
        (
            "Final Hard",
            result.hard_weights,
            result.final_hard_metrics.test_accuracy,
            result.final_hard_metrics.test_loss,
        ),
    )

    grid_x, grid_y, grid_features = _build_grid(dataset)
    surface_cmap = ListedColormap(["#dceaf7", "#fde2d0", "#e3f2df", "#f7e1f3"])
    point_colors = np.array(["#1f77b4", "#d62728", "#2ca02c", "#9467bd"])

    for axis, (panel_title, weights, test_accuracy, test_loss) in zip(axes, panels, strict=True):
        _render_panel(
            axis=axis,
            model=model,
            dataset=dataset,
            weights=weights,
            grid_x=grid_x,
            grid_y=grid_y,
            grid_features=grid_features,
            surface_cmap=surface_cmap,
            point_colors=point_colors,
        )
        axis.set_title(f"{panel_title}\nacc={test_accuracy:.3f} loss={test_loss:.4f}")
        axis.set_xlabel("x0")
        axis.set_ylabel("x1")

    axes[0].legend(loc="upper right")
    collapsed_ratio = f"{result.collapsed_count}/{result.parameter_count}"
    figure.suptitle(title or f"T-WFC make_moons decision surfaces ({collapsed_ratio} collapsed)")
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def save_snapshot_frames(
    model: ToyMLP,
    dataset: DatasetSplit,
    result: ExperimentResult,
    output_dir: str | Path,
    max_frames: int = 0,
    title_prefix: str | None = None,
) -> tuple[Path, ...]:
    if dataset.x_train.shape[1] != 2:
        raise ValueError("Visualization currently supports only 2D datasets")

    plt, ListedColormap = _load_matplotlib()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    snapshots = result.snapshots if max_frames <= 0 else _select_snapshots(result.snapshots, max_panels=max_frames)
    grid_x, grid_y, grid_features = _build_grid(dataset)
    surface_cmap = ListedColormap(["#dceaf7", "#fde2d0", "#e3f2df", "#f7e1f3"])
    point_colors = np.array(["#1f77b4", "#d62728", "#2ca02c", "#9467bd"])
    saved_paths: list[Path] = []

    for frame_index, snapshot in enumerate(snapshots):
        figure = _create_snapshot_figure(
            plt=plt,
            model=model,
            dataset=dataset,
            snapshot=snapshot,
            result=result,
            title_prefix=title_prefix or "T-WFC make_moons frame",
            grid_x=grid_x,
            grid_y=grid_y,
            grid_features=grid_features,
            surface_cmap=surface_cmap,
            point_colors=point_colors,
        )

        frame_name = (
            f"frame_{frame_index:03d}_step_{snapshot.step:02d}_"
            f"collapsed_{snapshot.collapsed_count:03d}.png"
        )
        frame_path = output_root / frame_name
        figure.savefig(frame_path, dpi=160)
        plt.close(figure)
        saved_paths.append(frame_path)

    return tuple(saved_paths)


def save_storyboard_plot(
    model: ToyMLP,
    dataset: DatasetSplit,
    result: ExperimentResult,
    output_path: str | Path,
    max_panels: int = 6,
    title: str | None = None,
) -> Path:
    if dataset.x_train.shape[1] != 2:
        raise ValueError("Visualization currently supports only 2D datasets")
    if max_panels < 2:
        raise ValueError("max_panels must be at least 2")

    plt, ListedColormap = _load_matplotlib()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    snapshots = _select_snapshots(result.snapshots, max_panels=max_panels)
    column_count = len(snapshots)
    figure = plt.figure(figsize=(4.2 * column_count, 11.2), constrained_layout=True)
    grid = figure.add_gridspec(4, column_count, height_ratios=(0.95, 0.95, 1.45, 1.45))

    loss_axis = figure.add_subplot(grid[0, :])
    accuracy_axis = figure.add_subplot(grid[1, :], sharex=loss_axis)

    steps = np.array([snapshot.step for snapshot in result.snapshots], dtype=np.int64)
    shadow_test_loss = np.array([snapshot.shadow_metrics.test_loss for snapshot in result.snapshots], dtype=np.float64)
    hard_test_loss = np.array([snapshot.hard_metrics.test_loss for snapshot in result.snapshots], dtype=np.float64)
    shadow_test_accuracy = np.array([snapshot.shadow_metrics.test_accuracy for snapshot in result.snapshots], dtype=np.float64)
    hard_test_accuracy = np.array([snapshot.hard_metrics.test_accuracy for snapshot in result.snapshots], dtype=np.float64)

    loss_axis.plot(steps, shadow_test_loss, color="#1f77b4", linewidth=2.2, label="shadow test loss")
    loss_axis.plot(steps, hard_test_loss, color="#d62728", linewidth=2.2, label="hard test loss")
    loss_axis.set_ylabel("Test Loss")
    loss_axis.grid(alpha=0.25)
    loss_axis.legend(loc="upper right")

    accuracy_axis.plot(steps, shadow_test_accuracy, color="#1f77b4", linewidth=2.2, label="shadow test acc")
    accuracy_axis.plot(steps, hard_test_accuracy, color="#d62728", linewidth=2.2, label="hard test acc")
    accuracy_axis.set_ylabel("Test Accuracy")
    accuracy_axis.set_xlabel("Committed step")
    accuracy_axis.set_ylim(-0.02, 1.02)
    accuracy_axis.grid(alpha=0.25)
    accuracy_axis.legend(loc="lower right")

    grid_x, grid_y, grid_features = _build_grid(dataset)
    surface_cmap = ListedColormap(["#dceaf7", "#fde2d0", "#e3f2df", "#f7e1f3"])
    point_colors = np.array(["#1f77b4", "#d62728", "#2ca02c", "#9467bd"])

    for column_index, snapshot in enumerate(snapshots):
        style = _snapshot_style(snapshot=snapshot, result=result)
        badges = _snapshot_badges(snapshot=snapshot, result=result)
        for metric_axis in (loss_axis, accuracy_axis):
            metric_axis.axvspan(snapshot.step - 0.22, snapshot.step + 0.22, color=style["shade"], alpha=0.32, zorder=0)

        loss_axis.scatter(
            snapshot.step,
            snapshot.shadow_metrics.test_loss,
            color="#1f77b4",
            s=44,
            zorder=4,
            edgecolor=style["accent"],
            linewidth=1.0,
        )
        loss_axis.scatter(
            snapshot.step,
            snapshot.hard_metrics.test_loss,
            color="#d62728",
            s=44,
            zorder=4,
            edgecolor=style["accent"],
            linewidth=1.0,
        )
        accuracy_axis.scatter(
            snapshot.step,
            snapshot.shadow_metrics.test_accuracy,
            color="#1f77b4",
            s=44,
            zorder=4,
            edgecolor=style["accent"],
            linewidth=1.0,
        )
        accuracy_axis.scatter(
            snapshot.step,
            snapshot.hard_metrics.test_accuracy,
            color="#d62728",
            s=44,
            zorder=4,
            edgecolor=style["accent"],
            linewidth=1.0,
        )

        shadow_axis = figure.add_subplot(grid[2, column_index])
        hard_axis = figure.add_subplot(grid[3, column_index])

        _render_panel(
            axis=shadow_axis,
            model=model,
            dataset=dataset,
            weights=snapshot.shadow_weights,
            grid_x=grid_x,
            grid_y=grid_y,
            grid_features=grid_features,
            surface_cmap=surface_cmap,
            point_colors=point_colors,
        )
        _render_panel(
            axis=hard_axis,
            model=model,
            dataset=dataset,
            weights=snapshot.hard_weights,
            grid_x=grid_x,
            grid_y=grid_y,
            grid_features=grid_features,
            surface_cmap=surface_cmap,
            point_colors=point_colors,
        )

        shadow_axis.set_title(
            f"Shadow s{snapshot.step:02d}\n"
            f"c={snapshot.collapsed_count} "
            f"acc={snapshot.shadow_metrics.test_accuracy:.3f} "
            f"loss={snapshot.shadow_metrics.test_loss:.4f}"
        )
        hard_axis.set_title(
            f"Hard s{snapshot.step:02d}\n"
            f"c={snapshot.collapsed_count} "
            f"acc={snapshot.hard_metrics.test_accuracy:.3f} "
            f"loss={snapshot.hard_metrics.test_loss:.4f}"
        )
        shadow_axis.set_xlabel("x0")
        shadow_axis.set_ylabel("x1")
        hard_axis.set_xlabel("x0")
        hard_axis.set_ylabel("x1")
        _style_snapshot_axes((shadow_axis, hard_axis), style)
        shadow_axis.text(
            0.02,
            0.98,
            style["label"],
            ha="left",
            va="top",
            transform=shadow_axis.transAxes,
            fontsize=9.2,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.28", facecolor=style["badge"], edgecolor="none", alpha=0.95),
        )
        _draw_badge_stack(
            shadow_axis,
            badges,
            anchor_x=0.02,
            anchor_y=0.86,
            vertical_gap=0.12,
            fontsize=7.8,
        )
        _draw_ban_overlay(hard_axis, snapshot, anchor_x=0.98, anchor_y=0.02, fontsize=7.0)
        hard_axis.text(
            0.5,
            -0.22,
            _snapshot_meta_label(snapshot=snapshot, result=result),
            ha="center",
            va="top",
            transform=hard_axis.transAxes,
            fontsize=8.3,
        )

    _plot_event_markers(
        axes=(loss_axis, accuracy_axis),
        result=result,
        y_values=(
            shadow_test_loss,
            shadow_test_accuracy,
        ),
    )
    figure.suptitle(title or f"T-WFC storyboard ({result.collapsed_count}/{result.parameter_count} collapsed)")
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def save_snapshot_gif(
    model: ToyMLP,
    dataset: DatasetSplit,
    result: ExperimentResult,
    output_path: str | Path,
    max_frames: int = 0,
    frame_duration_ms: int = 450,
    title_prefix: str | None = None,
) -> Path:
    if dataset.x_train.shape[1] != 2:
        raise ValueError("Visualization currently supports only 2D datasets")
    if frame_duration_ms <= 0:
        raise ValueError("frame_duration_ms must be positive")

    Image = _load_pillow()
    plt, ListedColormap = _load_matplotlib()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    snapshots = result.snapshots if max_frames <= 0 else _select_snapshots(result.snapshots, max_panels=max_frames)
    grid_x, grid_y, grid_features = _build_grid(dataset)
    surface_cmap = ListedColormap(["#dceaf7", "#fde2d0", "#e3f2df", "#f7e1f3"])
    point_colors = np.array(["#1f77b4", "#d62728", "#2ca02c", "#9467bd"])
    frames = []

    for snapshot in snapshots:
        figure = _create_snapshot_figure(
            plt=plt,
            model=model,
            dataset=dataset,
            snapshot=snapshot,
            result=result,
            title_prefix=title_prefix or "T-WFC make_moons animation",
            grid_x=grid_x,
            grid_y=grid_y,
            grid_features=grid_features,
            surface_cmap=surface_cmap,
            point_colors=point_colors,
        )
        with BytesIO() as buffer:
            figure.savefig(buffer, format="png", dpi=145)
            buffer.seek(0)
            frame = Image.open(buffer).convert("RGBA").copy()
        plt.close(figure)
        frames.append(frame)

    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        disposal=2,
    )
    for frame in frames:
        frame.close()
    return output


def save_progress_plot(
    model: ToyMLP,
    dataset: DatasetSplit,
    result: ExperimentResult,
    output_path: str | Path,
    max_panels: int = 6,
    title: str | None = None,
) -> Path:
    if dataset.x_train.shape[1] != 2:
        raise ValueError("Visualization currently supports only 2D datasets")
    if max_panels < 2:
        raise ValueError("max_panels must be at least 2")

    plt, ListedColormap = _load_matplotlib()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    snapshots = _select_snapshots(result.snapshots, max_panels=max_panels)
    column_count = len(snapshots)
    figure, axes = plt.subplots(2, column_count, figsize=(4.1 * column_count, 8.2), constrained_layout=True)
    axes = np.atleast_2d(axes)

    grid_x, grid_y, grid_features = _build_grid(dataset)
    surface_cmap = ListedColormap(["#dceaf7", "#fde2d0", "#e3f2df", "#f7e1f3"])
    point_colors = np.array(["#1f77b4", "#d62728", "#2ca02c", "#9467bd"])

    for column_index, snapshot in enumerate(snapshots):
        shadow_axis = axes[0, column_index]
        hard_axis = axes[1, column_index]

        _render_panel(
            axis=shadow_axis,
            model=model,
            dataset=dataset,
            weights=snapshot.shadow_weights,
            grid_x=grid_x,
            grid_y=grid_y,
            grid_features=grid_features,
            surface_cmap=surface_cmap,
            point_colors=point_colors,
        )
        _render_panel(
            axis=hard_axis,
            model=model,
            dataset=dataset,
            weights=snapshot.hard_weights,
            grid_x=grid_x,
            grid_y=grid_y,
            grid_features=grid_features,
            surface_cmap=surface_cmap,
            point_colors=point_colors,
        )

        shadow_axis.set_title(
            "Shadow "
            f"s{snapshot.step:02d}\n"
            f"c={snapshot.collapsed_count} "
            f"acc={snapshot.shadow_metrics.test_accuracy:.3f} "
            f"loss={snapshot.shadow_metrics.test_loss:.4f}"
        )
        hard_axis.set_title(
            "Hard "
            f"s{snapshot.step:02d}\n"
            f"c={snapshot.collapsed_count} "
            f"acc={snapshot.hard_metrics.test_accuracy:.3f} "
            f"loss={snapshot.hard_metrics.test_loss:.4f}"
        )

        shadow_axis.set_xlabel("x0")
        shadow_axis.set_ylabel("x1")
        hard_axis.set_xlabel("x0")
        hard_axis.set_ylabel("x1")
        _draw_ban_overlay(hard_axis, snapshot, anchor_x=0.98, anchor_y=0.02, fontsize=7.0)

    axes[0, 0].legend(loc="upper right")
    collapsed_ratio = f"{result.collapsed_count}/{result.parameter_count}"
    figure.suptitle(title or f"T-WFC make_moons progress ({collapsed_ratio} collapsed)")
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def _build_grid(dataset: DatasetSplit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_points = np.vstack((dataset.x_train, dataset.x_test))
    x_min, y_min = all_points.min(axis=0) - 0.75
    x_max, y_max = all_points.max(axis=0) + 0.75
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 240),
        np.linspace(y_min, y_max, 240),
    )
    grid_features = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    return grid_x, grid_y, grid_features


def _render_panel(
    axis,
    model: ToyMLP,
    dataset: DatasetSplit,
    weights: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_features: np.ndarray,
    surface_cmap,
    point_colors: np.ndarray,
) -> None:
    predictions = model.predict(weights, grid_features).reshape(grid_x.shape)
    axis.contourf(
        grid_x,
        grid_y,
        predictions,
        levels=np.arange(predictions.max() + 2) - 0.5,
        cmap=surface_cmap,
        alpha=0.95,
    )

    axis.scatter(
        dataset.x_train[:, 0],
        dataset.x_train[:, 1],
        c=point_colors[dataset.y_train],
        edgecolor="#111111",
        linewidth=0.4,
        s=34,
        marker="o",
        label="train",
    )
    axis.scatter(
        dataset.x_test[:, 0],
        dataset.x_test[:, 1],
        c=point_colors[dataset.y_test],
        edgecolor="#111111",
        linewidth=0.8,
        s=54,
        marker="^",
        label="test",
    )


def _create_snapshot_figure(
    plt,
    model: ToyMLP,
    dataset: DatasetSplit,
    snapshot: ExperimentSnapshot,
    result: ExperimentResult,
    title_prefix: str,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_features: np.ndarray,
    surface_cmap,
    point_colors: np.ndarray,
):
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.9), constrained_layout=True)
    style = _snapshot_style(snapshot=snapshot, result=result)
    badges = _snapshot_badges(snapshot=snapshot, result=result)
    panels = (
        ("Shadow", snapshot.shadow_weights, snapshot.shadow_metrics.test_accuracy, snapshot.shadow_metrics.test_loss),
        ("Hard", snapshot.hard_weights, snapshot.hard_metrics.test_accuracy, snapshot.hard_metrics.test_loss),
    )

    for axis, (panel_name, weights, test_accuracy, test_loss) in zip(axes, panels, strict=True):
        _render_panel(
            axis=axis,
            model=model,
            dataset=dataset,
            weights=weights,
            grid_x=grid_x,
            grid_y=grid_y,
            grid_features=grid_features,
            surface_cmap=surface_cmap,
            point_colors=point_colors,
        )
        axis.set_title(f"{panel_name}\nacc={test_accuracy:.3f} loss={test_loss:.4f}")
        axis.set_xlabel("x0")
        axis.set_ylabel("x1")

    axes[0].legend(loc="upper right")
    _style_snapshot_axes(axes, style)
    figure.suptitle(
        _snapshot_title(
            snapshot=snapshot,
            result=result,
            title_prefix=title_prefix,
        )
    )
    figure.text(
        0.018,
        0.965,
        style["label"],
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=style["badge"], edgecolor="none", alpha=0.96),
    )
    _draw_badge_stack(
        axes[0],
        badges,
        anchor_x=0.02,
        anchor_y=0.86,
        vertical_gap=0.12,
        fontsize=8.6,
    )
    _draw_ban_overlay(axes[1], snapshot, anchor_x=0.98, anchor_y=0.02, fontsize=8.0)
    figure.text(
        0.982,
        0.965,
        _snapshot_counter_text(snapshot),
        ha="right",
        va="top",
        fontsize=9.2,
        color="#111827",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=style["accent"], linewidth=1.2, alpha=0.93),
    )
    figure.text(
        0.5,
        0.015,
        _snapshot_footer(snapshot=snapshot, result=result),
        ha="center",
        va="bottom",
        fontsize=9.5,
    )
    return figure


def _snapshot_title(
    snapshot: ExperimentSnapshot,
    result: ExperimentResult,
    title_prefix: str,
) -> str:
    collapsed_ratio = f"{snapshot.collapsed_count}/{result.parameter_count}"
    return f"{title_prefix} | step {snapshot.step:02d} | collapsed {collapsed_ratio}"


def _snapshot_footer(
    snapshot: ExperimentSnapshot,
    result: ExperimentResult,
) -> str:
    ban_focus = _ban_focus_summary(snapshot, limit=1, include_values=False)
    if snapshot.step == 0:
        text = (
            "Initial superposition state before any committed collapse. "
            f"rollbacks={snapshot.rollback_count} alt={snapshot.backtrack_count} forced={snapshot.forced_commit_count} "
            f"bans={snapshot.forbidden_value_count} pressure={snapshot.frontier_pressure}"
        )
        if ban_focus:
            text += f" | focus={ban_focus}"
        return text

    step_log = result.step_logs[snapshot.step - 1]
    flags: list[str] = []
    if step_log.backtracked:
        flags.append("alt-choice")
    if step_log.forced_commit:
        flags.append("forced-commit")
    flag_text = ", ".join(flags) if flags else "direct-commit"
    text = (
        f"{step_log.parameter_name}={step_log.chosen_value:.1f} | "
        f"entropy={step_log.entropy:.4f} | "
        f"shadow={step_log.loss_before:.4f}->{step_log.loss_after:.4f} | "
        f"hard={step_log.hard_loss_after:.4f} | "
        f"score={step_log.score_after:.4f} | "
        f"{flag_text} | "
        f"rollbacks={snapshot.rollback_count}(+{snapshot.rollback_delta}) "
        f"alt={snapshot.backtrack_count}(+{snapshot.backtrack_delta}) "
        f"forced={snapshot.forced_commit_count}(+{snapshot.forced_commit_delta}) | "
        f"bans={snapshot.forbidden_value_count}(delta {snapshot.forbidden_value_delta:+d}) "
        f"pressure={snapshot.frontier_pressure}"
    )
    if ban_focus:
        text += f" | focus={ban_focus}"
    return text


def _snapshot_meta_label(
    snapshot: ExperimentSnapshot,
    result: ExperimentResult,
) -> str:
    ban_focus = _ban_focus_summary(snapshot, limit=2, include_values=False)
    if snapshot.step == 0:
        text = (
            "init\n"
            f"rb={snapshot.rollback_count}(+{snapshot.rollback_delta}) "
            f"alt={snapshot.backtrack_count}(+{snapshot.backtrack_delta}) "
            f"forced={snapshot.forced_commit_count}(+{snapshot.forced_commit_delta})\n"
            f"bans={snapshot.forbidden_value_count}(delta {snapshot.forbidden_value_delta:+d}) "
            f"pressure={snapshot.frontier_pressure}"
        )
        if ban_focus:
            text += f"\nfocus={ban_focus}"
        return text

    step_log = result.step_logs[snapshot.step - 1]
    suffix = []
    if step_log.backtracked:
        suffix.append("alt")
    if step_log.forced_commit:
        suffix.append("forced")
    suffix_text = f" ({'/'.join(suffix)})" if suffix else ""
    text = (
        f"{step_log.parameter_name}={step_log.chosen_value:.1f}{suffix_text}\n"
        f"rb={snapshot.rollback_count}(+{snapshot.rollback_delta}) "
        f"alt={snapshot.backtrack_count}(+{snapshot.backtrack_delta}) "
        f"forced={snapshot.forced_commit_count}(+{snapshot.forced_commit_delta})\n"
        f"bans={snapshot.forbidden_value_count}(delta {snapshot.forbidden_value_delta:+d}) "
        f"pressure={snapshot.frontier_pressure}"
    )
    if ban_focus:
        text += f"\nfocus={ban_focus}"
    return text


def _snapshot_counter_text(snapshot: ExperimentSnapshot) -> str:
    text = (
        f"rollbacks {snapshot.rollback_count} (+{snapshot.rollback_delta}) | "
        f"alt {snapshot.backtrack_count} (+{snapshot.backtrack_delta}) | "
        f"forced {snapshot.forced_commit_count} (+{snapshot.forced_commit_delta}) | "
        f"bans {snapshot.forbidden_value_count} ({snapshot.forbidden_weight_count}w, {snapshot.forbidden_value_delta:+d}) | "
        f"pressure {snapshot.frontier_pressure}"
    )
    ban_focus = _ban_focus_summary(snapshot, limit=1, include_values=False)
    if ban_focus:
        text += f" | focus {ban_focus}"
    return text


def _snapshot_style(
    snapshot: ExperimentSnapshot,
    result: ExperimentResult,
) -> dict[str, str]:
    if snapshot.step == 0:
        return {
            "label": "INITIAL",
            "accent": "#4b5563",
            "badge": "#111827",
            "shade": "#d1d5db",
        }

    step_log = result.step_logs[snapshot.step - 1]
    if snapshot.rollback_delta > 0 and step_log.forced_commit:
        return {
            "label": "ROLLBACK RECOVERY",
            "accent": "#be123c",
            "badge": "#9f1239",
            "shade": "#fbcfe8",
        }
    if snapshot.rollback_delta > 0:
        return {
            "label": "ROLLBACK PRESSURE",
            "accent": "#b91c1c",
            "badge": "#991b1b",
            "shade": "#fecaca",
        }
    if step_log.forced_commit and step_log.backtracked:
        return {
            "label": "FORCED + ALT",
            "accent": "#b45309",
            "badge": "#9a3412",
            "shade": "#fed7aa",
        }
    if step_log.forced_commit:
        return {
            "label": "FORCED COMMIT",
            "accent": "#dc2626",
            "badge": "#b91c1c",
            "shade": "#fecaca",
        }
    if step_log.backtracked:
        return {
            "label": "ALT CHOICE",
            "accent": "#d97706",
            "badge": "#b45309",
            "shade": "#fde68a",
        }
    return {
        "label": "DIRECT COMMIT",
        "accent": "#059669",
        "badge": "#047857",
        "shade": "#bbf7d0",
    }


def _snapshot_badges(
    snapshot: ExperimentSnapshot,
    result: ExperimentResult,
) -> tuple[tuple[str, str], ...]:
    if snapshot.step == 0:
        return (("INITIAL", "#111827"),)

    badges: list[tuple[str, str]] = []
    if snapshot.rollback_delta > 0:
        badges.append((f"ROLLBACK +{snapshot.rollback_delta}", "#991b1b"))
    if snapshot.backtrack_delta > 0:
        badges.append((f"ALT +{snapshot.backtrack_delta}", "#b45309"))
    if snapshot.forced_commit_delta > 0:
        badges.append((f"FORCED +{snapshot.forced_commit_delta}", "#b91c1c"))
    if snapshot.forbidden_value_count > 0:
        badges.append((f"BAN {snapshot.forbidden_value_count}", "#7c3aed"))
    if snapshot.frontier_pressure > 0:
        badges.append((f"PRESS {snapshot.frontier_pressure}", "#db2777"))
    if not badges:
        badges.append(("DIRECT", "#047857"))
    return tuple(badges)


def _experiment_summary_style(experiment: SeedExperiment) -> dict[str, str]:
    max_pressure = max(snapshot.frontier_pressure for snapshot in experiment.result.snapshots)
    max_bans = max(snapshot.forbidden_value_count for snapshot in experiment.result.snapshots)
    if max_pressure > 0:
        return {
            "label": "PRESSURED",
            "accent": "#be123c",
            "badge": "#9f1239",
            "shade": "#fbcfe8",
        }
    if max_bans > 0:
        return {
            "label": "BANNED PATHS",
            "accent": "#7c3aed",
            "badge": "#6d28d9",
            "shade": "#ede9fe",
        }
    if experiment.result.forced_commit_count > 0:
        return {
            "label": "FORCED SEARCH",
            "accent": "#dc2626",
            "badge": "#b91c1c",
            "shade": "#fecaca",
        }
    return {
        "label": "STABLE",
        "accent": "#059669",
        "badge": "#047857",
        "shade": "#bbf7d0",
    }


def _experiment_summary_badges(experiment: SeedExperiment) -> tuple[tuple[str, str], ...]:
    max_bans = max(snapshot.forbidden_value_count for snapshot in experiment.result.snapshots)
    max_pressure = max(snapshot.frontier_pressure for snapshot in experiment.result.snapshots)
    badges: list[tuple[str, str]] = []
    if experiment.result.rollback_count > 0:
        badges.append((f"ROLLBACK {experiment.result.rollback_count}", "#991b1b"))
    if experiment.result.backtrack_count > 0:
        badges.append((f"ALT {experiment.result.backtrack_count}", "#b45309"))
    if experiment.result.forced_commit_count > 0:
        badges.append((f"FORCED {experiment.result.forced_commit_count}", "#b91c1c"))
    if max_bans > 0:
        badges.append((f"BAN<={max_bans}", "#7c3aed"))
    if max_pressure > 0:
        badges.append((f"PRESS<={max_pressure}", "#db2777"))
    if not badges:
        badges.append(("CLEAN", "#047857"))
    return tuple(badges)


def _experiment_panel_footer(experiment: SeedExperiment) -> str:
    peak_ban = _experiment_peak_ban_text(experiment)
    text = (
        f"shadow acc={experiment.result.final_shadow_metrics.test_accuracy:.3f} | "
        f"rb={experiment.result.rollback_count} alt={experiment.result.backtrack_count} "
        f"forced={experiment.result.forced_commit_count} | "
        f"max bans={max(snapshot.forbidden_value_count for snapshot in experiment.result.snapshots)} "
        f"max pressure={max(snapshot.frontier_pressure for snapshot in experiment.result.snapshots)}"
    )
    if peak_ban:
        text += f"\npeak ban={peak_ban}"
    return text


def _style_snapshot_axes(axes, style: dict[str, str]) -> None:
    for axis in axes:
        for spine in axis.spines.values():
            spine.set_color(style["accent"])
            spine.set_linewidth(2.2)


def _draw_badge_stack(
    axis,
    badges: tuple[tuple[str, str], ...],
    anchor_x: float,
    anchor_y: float,
    vertical_gap: float,
    fontsize: float,
) -> None:
    for index, (label, color) in enumerate(badges):
        axis.text(
            anchor_x,
            anchor_y - index * vertical_gap,
            label,
            ha="left",
            va="top",
            transform=axis.transAxes,
            fontsize=fontsize,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=color, edgecolor="none", alpha=0.94),
        )


def _draw_ban_overlay(
    axis,
    snapshot: ExperimentSnapshot,
    anchor_x: float,
    anchor_y: float,
    fontsize: float,
) -> None:
    overlay_text = _snapshot_ban_overlay_text(snapshot)
    if overlay_text is None:
        return

    axis.text(
        anchor_x,
        anchor_y,
        overlay_text,
        ha="right",
        va="bottom",
        transform=axis.transAxes,
        fontsize=fontsize,
        color="#4c1d95",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#f5f3ff", edgecolor="#7c3aed", linewidth=1.2, alpha=0.95),
    )


def _draw_experiment_peak_ban_overlay(axis, experiment: SeedExperiment) -> None:
    peak_ban = _experiment_peak_ban_text(experiment)
    if not peak_ban:
        return

    axis.text(
        0.98,
        0.02,
        f"peak ban\n{peak_ban}",
        ha="right",
        va="bottom",
        transform=axis.transAxes,
        fontsize=7.0,
        color="#4c1d95",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#f5f3ff", edgecolor="#7c3aed", linewidth=1.0, alpha=0.94),
    )


def _annotate_ban_identity(axis, snapshots: tuple[ExperimentSnapshot, ...] | list[ExperimentSnapshot]) -> None:
    annotated = 0
    for snapshot in snapshots:
        if not snapshot.forbidden_delta_labels:
            continue
        axis.annotate(
            _truncate_text(snapshot.forbidden_delta_labels[0], limit=18),
            xy=(snapshot.step, snapshot.collapsed_count),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.3,
            rotation=24,
            color="#6d28d9",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#7c3aed", linewidth=0.9, alpha=0.92),
        )
        annotated += 1
        if annotated >= 6:
            break


def _plot_event_markers(
    axes,
    result: ExperimentResult,
    y_values: tuple[np.ndarray, ...],
) -> None:
    if not result.step_logs:
        return

    snapshots = result.snapshots[1:]
    steps = np.array([snapshot.step for snapshot in snapshots], dtype=np.int64)
    rollback_mask = np.array([snapshot.rollback_delta > 0 for snapshot in snapshots], dtype=bool)
    forced_mask = np.array([snapshot.forced_commit_delta > 0 for snapshot in snapshots], dtype=bool)
    backtracked_mask = np.array([snapshot.backtrack_delta > 0 for snapshot in snapshots], dtype=bool)
    ban_mask = np.array([snapshot.forbidden_value_delta > 0 for snapshot in snapshots], dtype=bool)
    pressure_mask = np.array([snapshot.frontier_pressure > 0 for snapshot in snapshots], dtype=bool)
    direct_mask = ~(rollback_mask | forced_mask | backtracked_mask | ban_mask | pressure_mask)

    for axis, values in zip(axes, y_values, strict=True):
        if direct_mask.any():
            axis.scatter(
                steps[direct_mask],
                values[steps[direct_mask]],
                color="#059669",
                marker="o",
                s=26,
                alpha=0.45,
                label="direct commit" if axis is axes[0] else None,
                zorder=3,
            )
        if rollback_mask.any():
            axis.scatter(
                steps[rollback_mask],
                values[steps[rollback_mask]],
                color="#991b1b",
                marker="s",
                s=58,
                alpha=0.82,
                label="rollback burst" if axis is axes[0] else None,
                zorder=6,
            )
        if ban_mask.any():
            axis.scatter(
                steps[ban_mask],
                values[steps[ban_mask]],
                color="#7c3aed",
                marker="P",
                s=48,
                alpha=0.82,
                label="ban growth" if axis is axes[0] else None,
                zorder=5,
            )
        if backtracked_mask.any():
            axis.scatter(
                steps[backtracked_mask],
                values[steps[backtracked_mask]],
                color="#d97706",
                marker="^",
                s=42,
                alpha=0.8,
                label="alt choice" if axis is axes[0] else None,
                zorder=4,
            )
        if forced_mask.any():
            axis.scatter(
                steps[forced_mask],
                values[steps[forced_mask]],
                color="#dc2626",
                marker="X",
                s=52,
                alpha=0.85,
                label="forced commit" if axis is axes[0] else None,
                zorder=5,
            )
        if pressure_mask.any():
            axis.scatter(
                steps[pressure_mask],
                values[steps[pressure_mask]],
                color="#db2777",
                marker="D",
                s=44,
                alpha=0.78,
                label="frontier pressure" if axis is axes[0] else None,
                zorder=4.5,
            )

    if rollback_mask.any() or backtracked_mask.any() or forced_mask.any() or ban_mask.any() or pressure_mask.any() or direct_mask.any():
        axes[0].legend(loc="upper right", ncol=4)


def _snapshot_ban_overlay_text(snapshot: ExperimentSnapshot) -> str | None:
    lines: list[str] = []
    ban_focus = _ban_focus_summary(snapshot, limit=2, include_values=True)
    if ban_focus:
        lines.append(f"ban focus: {ban_focus}")
    if snapshot.forbidden_delta_labels:
        lines.append(f"delta: {'; '.join(snapshot.forbidden_delta_labels[:2])}")
    if not lines:
        return None
    return "\n".join(lines)


def _ban_focus_summary(
    snapshot: ExperimentSnapshot,
    limit: int,
    include_values: bool,
) -> str:
    if not snapshot.forbidden_entries:
        return ""
    return "; ".join(
        _format_forbidden_entry(entry, include_values=include_values)
        for entry in snapshot.forbidden_entries[:limit]
    )


def _format_forbidden_entry(entry, include_values: bool) -> str:
    if include_values:
        values = ",".join(f"{value:.1f}" for value in entry.banned_values)
        return f"{entry.parameter_name}=[{values}]"
    return f"{entry.parameter_name}({entry.ban_count})"


def _experiment_peak_ban_text(experiment: SeedExperiment) -> str:
    peak_entry = None
    for snapshot in experiment.result.snapshots:
        for entry in snapshot.forbidden_entries:
            if peak_entry is None or entry.ban_count > peak_entry.ban_count or (
                entry.ban_count == peak_entry.ban_count and entry.weight_index < peak_entry.weight_index
            ):
                peak_entry = entry

    if peak_entry is None:
        return ""
    return _format_forbidden_entry(peak_entry, include_values=False)


def _truncate_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(limit - 1, 0)] + "..."


def _select_snapshots(
    snapshots: tuple[ExperimentSnapshot, ...],
    max_panels: int,
) -> tuple[ExperimentSnapshot, ...]:
    if len(snapshots) <= max_panels:
        return snapshots

    candidate_indices = np.linspace(0, len(snapshots) - 1, num=max_panels)
    deduped_indices: list[int] = []
    for index in candidate_indices.round().astype(int).tolist():
        if not deduped_indices or deduped_indices[-1] != index:
            deduped_indices.append(index)

    if deduped_indices[-1] != len(snapshots) - 1:
        deduped_indices[-1] = len(snapshots) - 1

    return tuple(snapshots[index] for index in deduped_indices)


def _load_matplotlib():
    cache_dir = Path(tempfile.gettempdir()) / "t_wfc_mplconfig"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    return plt, ListedColormap


def _load_pillow():
    from PIL import Image

    return Image
