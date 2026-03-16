from __future__ import annotations

import argparse
from pathlib import Path

from .baseline import SGDBaselineConfig, train_sgd_classifier
from .batch import _resolve_hidden_layers, export_seed_artifacts, run_seed_batch
from .data import load_dataset
from .model import MLPConfig, ToyMLP
from .reporting import save_seed_markdown_report
from .trainer import TWFCConfig, TWFCTrainer
from .visualization import (
    save_baseline_comparison_gif,
    save_baseline_comparison_plot,
    save_baseline_metrics_comparison_plot,
    save_experiment_plot,
    save_metrics_plot,
    save_progress_plot,
    save_seed_gallery_plot,
    save_snapshot_frames,
    save_snapshot_gif,
    save_storyboard_plot,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the T-WFC toy prototype.")
    parser.add_argument("--dataset", choices=("make_moons", "iris", "spiral"), default="make_moons", help="Dataset to run the prototype on.")
    parser.add_argument("--samples", type=int, default=0, help="Dataset sample count. Use 0 to select a dataset-specific default.")
    parser.add_argument("--noise", type=float, default=-1.0, help="Dataset noise level. Use a negative value to select a dataset-specific default.")
    parser.add_argument("--hidden-dim", type=int, default=0, help="Legacy single hidden-layer width. Use 0 to select a dataset-specific default.")
    parser.add_argument("--hidden-layers", type=str, default="", help="Optional comma-separated hidden layer widths, for example '16,16'.")
    parser.add_argument("--initial-jitter", type=float, default=-1.0, help="Initial symmetry-breaking prior strength. Use a negative value to select a model-specific default.")
    parser.add_argument("--observation-budget", type=int, default=8, help="How many unresolved weights to observe per step.")
    parser.add_argument("--propagation-budget", type=int, default=6, help="How many neighboring weights to update after each collapse.")
    parser.add_argument("--max-steps", type=int, default=0, help="How many collapse steps to run. Use 0 for a full collapse.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=-1.0,
        help="Softmax temperature used to turn losses into pseudo-probabilities. Use a negative value to select a model-specific default.",
    )
    parser.add_argument("--backtrack-tolerance", type=float, default=0.03, help="Allowed loss increase before trying an alternate collapse value.")
    parser.add_argument("--hard-loss-weight", type=float, default=0.35, help="Weight applied to hard-state loss when ranking forced-commit candidates.")
    parser.add_argument("--hard-gap-weight", type=float, default=0.2, help="Extra penalty for cases where hard-state loss is worse than shadow loss.")
    parser.add_argument("--rollback-depth", type=int, default=2, help="How many committed collapse steps to rewind when rollback is triggered.")
    parser.add_argument("--rollback-depth-growth", type=int, default=0, help="How many extra committed steps to rewind for each repeated rollback at the same frontier.")
    parser.add_argument("--rollback-ban-count", type=int, default=1, help="How many reverted decisions to ban after a rollback. The oldest reverted choices are banned first.")
    parser.add_argument("--max-frontier-rollbacks", type=int, default=3, help="How many rollbacks to allow at the same search frontier before force-committing the best available collapse.")
    parser.add_argument("--max-attempt-multiplier", type=int, default=8, help="Safety multiplier that caps total collapse attempts when rollbacks happen.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for dataset generation and observation sampling.")
    parser.add_argument("--seed-list", type=str, default="", help="Optional comma-separated seed list for multi-seed gallery/report runs.")
    parser.add_argument("--compare-sgd", action="store_true", help="Train the same MLP with a numpy SGD baseline and print a side-by-side summary.")
    parser.add_argument("--sgd-epochs", type=int, default=240, help="Epoch count for the SGD baseline.")
    parser.add_argument("--sgd-learning-rate", type=float, default=0.08, help="Initial learning rate for the SGD baseline.")
    parser.add_argument("--sgd-learning-rate-decay", type=float, default=0.01, help="Per-epoch learning-rate decay for the SGD baseline.")
    parser.add_argument("--sgd-batch-size", type=int, default=32, help="Mini-batch size for the SGD baseline. Use -1 for full-batch training.")
    parser.add_argument("--sgd-weight-scale", type=float, default=1.0, help="Initialization scale for the SGD baseline.")
    parser.add_argument("--show-steps", type=int, default=8, help="How many collapse steps to print in the summary.")
    parser.add_argument("--save-plot", type=Path, default=None, help="Optional PNG output path for a 2D decision-surface plot.")
    parser.add_argument("--save-progress-plot", type=Path, default=None, help="Optional PNG output path for a multi-step progress visualization.")
    parser.add_argument("--progress-panels", type=int, default=6, help="How many timeline snapshots to include in the progress plot.")
    parser.add_argument("--save-metrics-plot", type=Path, default=None, help="Optional PNG output path for a metrics timeline plot.")
    parser.add_argument("--save-storyboard", type=Path, default=None, help="Optional PNG output path for a combined metrics-plus-snapshots storyboard.")
    parser.add_argument("--storyboard-panels", type=int, default=6, help="How many snapshots to include in the storyboard plot.")
    parser.add_argument("--save-frames-dir", type=Path, default=None, help="Optional directory for per-snapshot frame PNG exports.")
    parser.add_argument("--max-frame-count", type=int, default=0, help="How many snapshot frames to export. Use 0 to export every committed snapshot.")
    parser.add_argument("--save-gif", type=Path, default=None, help="Optional GIF output path for an animated snapshot sequence.")
    parser.add_argument("--gif-frame-duration-ms", type=int, default=450, help="Frame duration used for the GIF animation.")
    parser.add_argument("--save-baseline-metrics-plot", type=Path, default=None, help="Optional PNG output path for a T-WFC-vs-SGD metrics comparison.")
    parser.add_argument("--save-baseline-comparison-plot", type=Path, default=None, help="Optional PNG output path for a 2D T-WFC-vs-SGD boundary comparison.")
    parser.add_argument("--save-baseline-comparison-gif", type=Path, default=None, help="Optional GIF output path for a 2D T-WFC-vs-SGD comparison animation.")
    parser.add_argument("--save-seed-gallery", type=Path, default=None, help="Optional PNG output path for a multi-seed comparison gallery.")
    parser.add_argument("--save-seed-artifacts-dir", type=Path, default=None, help="Optional directory for per-seed metrics/storyboard/GIF exports in batch mode.")
    parser.add_argument("--gallery-columns", type=int, default=3, help="How many columns to use for the multi-seed gallery.")
    parser.add_argument("--save-md-report", type=Path, default=None, help="Optional Markdown output path for a multi-seed summary report.")
    parser.add_argument("--report-title", type=str, default="", help="Optional custom title for the Markdown report or multi-seed gallery.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed_list = _parse_seed_list(args.seed_list)
    if len(seed_list) == 1:
        args.seed = seed_list[0]

    parsed_hidden_layers = _parse_hidden_layers(args.hidden_layers)
    resolved_hidden_layers = _resolve_hidden_layers(
        dataset_name=args.dataset,
        hidden_dim=args.hidden_dim,
        hidden_layers=parsed_hidden_layers,
    )
    resolved_initial_jitter = _resolve_initial_jitter(resolved_hidden_layers, args.initial_jitter)

    config = TWFCConfig(
        initial_jitter=resolved_initial_jitter,
        observation_budget=args.observation_budget,
        propagation_budget=args.propagation_budget,
        max_steps=None if args.max_steps == 0 else args.max_steps,
        temperature=args.temperature,
        backtrack_tolerance=args.backtrack_tolerance,
        hard_loss_weight=args.hard_loss_weight,
        hard_gap_weight=args.hard_gap_weight,
        rollback_depth=args.rollback_depth,
        rollback_depth_growth=args.rollback_depth_growth,
        rollback_ban_count=args.rollback_ban_count,
        max_frontier_rollbacks=args.max_frontier_rollbacks,
        max_attempt_multiplier=args.max_attempt_multiplier,
        seed=args.seed,
    )

    if len(seed_list) > 1:
        _run_seed_batch_mode(args, seed_list, config)
        return

    samples = _resolve_dataset_samples(args.dataset, args.samples)
    noise = _resolve_dataset_noise(args.dataset, args.noise)
    dataset = load_dataset(
        args.dataset,
        n_samples=samples,
        noise=noise,
        seed=args.seed,
    )
    input_dim = dataset.x_train.shape[1]
    output_dim = int(max(dataset.y_train.max(), dataset.y_test.max()) + 1)
    model = ToyMLP(MLPConfig(input_dim=input_dim, output_dim=output_dim, hidden_layers=resolved_hidden_layers))
    trainer = TWFCTrainer(
        model=model,
        config=config,
    )

    result = trainer.fit(dataset)

    print("T-WFC prototype run")
    print(f"Dataset/model: {args.dataset}, {model.architecture_label}")
    print(f"Collapsed weights: {result.collapsed_count}/{result.parameter_count}")
    print(f"Local backtracks used: {result.backtrack_count}")
    print(f"Rollbacks used: {result.rollback_count}")
    print(f"Forced commits used: {result.forced_commit_count}")
    print(
        "Shadow train loss/acc: "
        f"{result.initial_shadow_metrics.train_loss:.4f} -> {result.final_shadow_metrics.train_loss:.4f}, "
        f"{result.initial_shadow_metrics.train_accuracy:.3f} -> {result.final_shadow_metrics.train_accuracy:.3f}"
    )
    print(
        "Shadow test loss/acc:  "
        f"{result.initial_shadow_metrics.test_loss:.4f} -> {result.final_shadow_metrics.test_loss:.4f}, "
        f"{result.initial_shadow_metrics.test_accuracy:.3f} -> {result.final_shadow_metrics.test_accuracy:.3f}"
    )
    print(
        "Hard train/test acc:   "
        f"{result.final_hard_metrics.train_accuracy:.3f} / {result.final_hard_metrics.test_accuracy:.3f}"
    )
    print(
        "Hard train/test loss:  "
        f"{result.final_hard_metrics.train_loss:.4f} / {result.final_hard_metrics.test_loss:.4f}"
    )

    baseline_result = None
    if args.compare_sgd:
        baseline_result = train_sgd_classifier(
            model=model,
            dataset=dataset,
            config=SGDBaselineConfig(
                epochs=args.sgd_epochs,
                learning_rate=args.sgd_learning_rate,
                learning_rate_decay=args.sgd_learning_rate_decay,
                batch_size=args.sgd_batch_size,
                weight_scale=args.sgd_weight_scale,
                seed=args.seed,
            ),
        )
        print("")
        print("SGD baseline")
        print(
            "SGD train loss/acc:   "
            f"{baseline_result.initial_metrics.train_loss:.4f} -> {baseline_result.final_metrics.train_loss:.4f}, "
            f"{baseline_result.initial_metrics.train_accuracy:.3f} -> {baseline_result.final_metrics.train_accuracy:.3f}"
        )
        print(
            "SGD test loss/acc:    "
            f"{baseline_result.initial_metrics.test_loss:.4f} -> {baseline_result.final_metrics.test_loss:.4f}, "
            f"{baseline_result.initial_metrics.test_accuracy:.3f} -> {baseline_result.final_metrics.test_accuracy:.3f}"
        )
        print(
            "Hard-vs-SGD test acc: "
            f"{result.final_hard_metrics.test_accuracy:.3f} vs {baseline_result.final_metrics.test_accuracy:.3f}"
        )
        print(
            "Hard-vs-SGD test loss:"
            f" {result.final_hard_metrics.test_loss:.4f} vs {baseline_result.final_metrics.test_loss:.4f}"
        )

    if any(
        path is not None
        for path in (
            args.save_baseline_metrics_plot,
            args.save_baseline_comparison_plot,
            args.save_baseline_comparison_gif,
        )
    ) and baseline_result is None:
        raise ValueError("Baseline comparison outputs require --compare-sgd")

    if args.save_plot is not None:
        saved_path = save_experiment_plot(
            model=model,
            dataset=dataset,
            result=result,
            output_path=args.save_plot,
            title=f"T-WFC {args.dataset} ({result.collapsed_count}/{result.parameter_count} collapsed)",
        )
        print(f"Saved plot: {saved_path}")

    if args.save_progress_plot is not None:
        saved_progress_path = save_progress_plot(
            model=model,
            dataset=dataset,
            result=result,
            output_path=args.save_progress_plot,
            max_panels=args.progress_panels,
            title=f"T-WFC {args.dataset} progress ({result.collapsed_count}/{result.parameter_count} collapsed)",
        )
        print(f"Saved progress plot: {saved_progress_path}")

    if args.save_metrics_plot is not None:
        saved_metrics_path = save_metrics_plot(
            result=result,
            output_path=args.save_metrics_plot,
            title=f"T-WFC {args.dataset} metrics ({result.collapsed_count}/{result.parameter_count} collapsed)",
        )
        print(f"Saved metrics plot: {saved_metrics_path}")

    if args.save_storyboard is not None:
        saved_storyboard_path = save_storyboard_plot(
            model=model,
            dataset=dataset,
            result=result,
            output_path=args.save_storyboard,
            max_panels=args.storyboard_panels,
            title=f"T-WFC {args.dataset} storyboard ({result.collapsed_count}/{result.parameter_count} collapsed)",
        )
        print(f"Saved storyboard: {saved_storyboard_path}")

    if args.save_frames_dir is not None:
        saved_frames = save_snapshot_frames(
            model=model,
            dataset=dataset,
            result=result,
            output_dir=args.save_frames_dir,
            max_frames=args.max_frame_count,
            title_prefix=f"T-WFC {args.dataset}",
        )
        print(f"Saved snapshot frames: {len(saved_frames)} into {args.save_frames_dir}")

    if args.save_gif is not None:
        saved_gif_path = save_snapshot_gif(
            model=model,
            dataset=dataset,
            result=result,
            output_path=args.save_gif,
            max_frames=args.max_frame_count,
            frame_duration_ms=args.gif_frame_duration_ms,
            title_prefix=f"T-WFC {args.dataset}",
        )
        print(f"Saved GIF: {saved_gif_path}")

    if args.save_baseline_metrics_plot is not None:
        saved_baseline_metrics_path = save_baseline_metrics_comparison_plot(
            result=result,
            baseline_result=baseline_result,
            output_path=args.save_baseline_metrics_plot,
            title=f"T-WFC vs SGD metrics ({args.dataset}, {result.collapsed_count}/{result.parameter_count} collapsed)",
        )
        print(f"Saved baseline metrics comparison: {saved_baseline_metrics_path}")

    if args.save_baseline_comparison_plot is not None:
        saved_baseline_plot_path = save_baseline_comparison_plot(
            model=model,
            dataset=dataset,
            result=result,
            baseline_result=baseline_result,
            output_path=args.save_baseline_comparison_plot,
            title=f"T-WFC vs SGD boundaries ({args.dataset}, {result.collapsed_count}/{result.parameter_count} collapsed)",
        )
        print(f"Saved baseline boundary comparison: {saved_baseline_plot_path}")

    if args.save_baseline_comparison_gif is not None:
        saved_baseline_gif_path = save_baseline_comparison_gif(
            model=model,
            dataset=dataset,
            result=result,
            baseline_result=baseline_result,
            output_path=args.save_baseline_comparison_gif,
            max_frames=args.max_frame_count,
            frame_duration_ms=args.gif_frame_duration_ms,
            title_prefix=f"T-WFC vs SGD {args.dataset}",
        )
        print(f"Saved baseline comparison GIF: {saved_baseline_gif_path}")

    if args.show_steps <= 0:
        return

    print("")
    print("Sample collapse steps:")
    for step in result.step_logs[: args.show_steps]:
        print(
            f"  step={step.step:02d} "
            f"param={step.parameter_name:<8} "
            f"value={step.chosen_value:>4.1f} "
            f"entropy={step.entropy:.4f} "
            f"shadow={step.loss_before:.4f}->{step.loss_after:.4f} "
            f"hard={step.hard_loss_after:.4f} "
            f"score={step.score_after:.4f} "
            f"neighbors={len(step.propagated_indices)} "
            f"backtracked={step.backtracked} "
            f"forced={step.forced_commit}"
        )


def _parse_seed_list(raw_value: str) -> tuple[int, ...]:
    if not raw_value.strip():
        return ()
    return tuple(int(token.strip()) for token in raw_value.split(",") if token.strip())


def _parse_hidden_layers(raw_value: str) -> tuple[int, ...]:
    if not raw_value.strip():
        return ()
    return tuple(int(token.strip()) for token in raw_value.split(",") if token.strip())


def _resolve_dataset_samples(dataset_name: str, requested_samples: int) -> int:
    if requested_samples > 0:
        return requested_samples
    if dataset_name == "spiral":
        return 600
    return 120


def _resolve_dataset_noise(dataset_name: str, requested_noise: float) -> float:
    if requested_noise >= 0.0:
        return requested_noise
    if dataset_name == "spiral":
        return 0.16
    return 0.08


def _resolve_initial_jitter(hidden_layers: tuple[int, ...], requested_jitter: float) -> float:
    if requested_jitter >= 0.0:
        return requested_jitter
    if len(hidden_layers) > 1:
        return 0.08
    return 0.0


def _run_seed_batch_mode(args, seed_list: tuple[int, ...], config: TWFCConfig) -> None:
    if args.compare_sgd:
        raise ValueError("--compare-sgd currently supports only single-run mode")
    incompatible_single_run_flags = (
        args.save_plot,
        args.save_progress_plot,
        args.save_metrics_plot,
        args.save_storyboard,
        args.save_frames_dir,
        args.save_gif,
        args.save_baseline_metrics_plot,
        args.save_baseline_comparison_plot,
        args.save_baseline_comparison_gif,
    )
    if any(flag is not None for flag in incompatible_single_run_flags):
        raise ValueError("Single-run plot options cannot be combined with --seed-list batch mode")

    experiments = run_seed_batch(
        args.dataset,
        seed_list,
        samples=_resolve_dataset_samples(args.dataset, args.samples),
        noise=_resolve_dataset_noise(args.dataset, args.noise),
        hidden_dim=args.hidden_dim,
        hidden_layers=_parse_hidden_layers(args.hidden_layers),
        config_template=config,
    )

    hard_test_accuracy = [experiment.result.final_hard_metrics.test_accuracy for experiment in experiments]
    print("T-WFC seed batch run")
    print(f"Dataset: {args.dataset}")
    print(f"Seeds: {', '.join(str(seed) for seed in seed_list)}")
    print(f"Mean hard test acc: {sum(hard_test_accuracy) / len(hard_test_accuracy):.3f}")

    for experiment in experiments:
        max_bans = max(snapshot.forbidden_value_count for snapshot in experiment.result.snapshots)
        max_pressure = max(snapshot.frontier_pressure for snapshot in experiment.result.snapshots)
        print(
            f"  seed={experiment.seed:>3d} "
            f"hard_acc={experiment.result.final_hard_metrics.test_accuracy:.3f} "
            f"hard_loss={experiment.result.final_hard_metrics.test_loss:.4f} "
            f"rb={experiment.result.rollback_count} "
            f"alt={experiment.result.backtrack_count} "
            f"forced={experiment.result.forced_commit_count} "
            f"bans<={max_bans} "
            f"pressure<={max_pressure}"
        )

    gallery_path = None
    if args.save_seed_gallery is not None:
        gallery_path = save_seed_gallery_plot(
            experiments=experiments,
            output_path=args.save_seed_gallery,
            title=args.report_title or f"T-WFC {args.dataset} seed gallery",
            columns=args.gallery_columns,
        )
        print(f"Saved seed gallery: {gallery_path}")

    seed_artifacts = ()
    if args.save_seed_artifacts_dir is not None:
        seed_artifacts = export_seed_artifacts(
            experiments=experiments,
            output_dir=args.save_seed_artifacts_dir,
            storyboard_panels=args.storyboard_panels,
            max_frames=args.max_frame_count,
            gif_frame_duration_ms=args.gif_frame_duration_ms,
        )
        print(f"Saved per-seed artifacts: {len(seed_artifacts)} runs into {args.save_seed_artifacts_dir}")

    if args.save_md_report is not None:
        report_path = save_seed_markdown_report(
            experiments=experiments,
            output_path=args.save_md_report,
            title=args.report_title or f"T-WFC {args.dataset} seed report",
            gallery_path=gallery_path,
            seed_artifacts=seed_artifacts,
            config_summary={
                "samples": _resolve_dataset_samples(args.dataset, args.samples),
                "noise": _resolve_dataset_noise(args.dataset, args.noise),
                "hidden_layers": ",".join(
                    str(width)
                    for width in _resolve_hidden_layers(
                        dataset_name=args.dataset,
                        hidden_dim=args.hidden_dim,
                        hidden_layers=_parse_hidden_layers(args.hidden_layers),
                    )
                ),
                "initial_jitter": _resolve_initial_jitter(
                    _resolve_hidden_layers(
                        dataset_name=args.dataset,
                        hidden_dim=args.hidden_dim,
                        hidden_layers=_parse_hidden_layers(args.hidden_layers),
                    ),
                    args.initial_jitter,
                ),
                "observation_budget": args.observation_budget,
                "propagation_budget": args.propagation_budget,
                "max_steps": "full" if args.max_steps == 0 else args.max_steps,
                "backtrack_tolerance": args.backtrack_tolerance,
                "rollback_depth": args.rollback_depth,
                "rollback_depth_growth": args.rollback_depth_growth,
                "rollback_ban_count": args.rollback_ban_count,
                "max_frontier_rollbacks": args.max_frontier_rollbacks,
                "max_attempt_multiplier": args.max_attempt_multiplier,
            },
        )
        print(f"Saved Markdown report: {report_path}")


if __name__ == "__main__":
    main()
