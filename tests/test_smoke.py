from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from t_wfc.batch import export_seed_artifacts, run_seed_batch
from t_wfc.data import load_dataset, load_iris_dataset, make_moons_dataset
from t_wfc.model import MLPConfig, ToyMLP
from t_wfc.reporting import save_seed_markdown_report
from t_wfc.trainer import TWFCConfig, TWFCTrainer
from t_wfc.visualization import (
    save_experiment_plot,
    save_metrics_plot,
    save_progress_plot,
    save_seed_gallery_plot,
    save_snapshot_frames,
    save_snapshot_gif,
    save_storyboard_plot,
)


class MakeMoonsDatasetTest(unittest.TestCase):
    def test_dataset_shapes_and_labels(self) -> None:
        dataset = load_dataset("make_moons", n_samples=40, noise=0.05, seed=3)

        self.assertEqual(dataset.x_train.shape[1], 2)
        self.assertEqual(dataset.x_test.shape[1], 2)
        self.assertEqual(dataset.y_train.ndim, 1)
        self.assertEqual(dataset.y_test.ndim, 1)
        self.assertSetEqual(set(dataset.y_train.tolist()) | set(dataset.y_test.tolist()), {0, 1})


class IrisDatasetTest(unittest.TestCase):
    def test_dataset_shapes_and_labels(self) -> None:
        dataset = load_iris_dataset(seed=5)

        self.assertEqual(dataset.x_train.shape[1], 4)
        self.assertEqual(dataset.x_test.shape[1], 4)
        self.assertSetEqual(set(dataset.y_train.tolist()) | set(dataset.y_test.tolist()), {0, 1, 2})
        self.assertGreaterEqual(int((dataset.y_test == 0).sum()), 1)
        self.assertGreaterEqual(int((dataset.y_test == 1).sum()), 1)
        self.assertGreaterEqual(int((dataset.y_test == 2).sum()), 1)


class TWFCTrainerSmokeTest(unittest.TestCase):
    def test_make_moons_end_to_end_run(self) -> None:
        dataset = make_moons_dataset(n_samples=60, noise=0.04, seed=11)
        model = ToyMLP(MLPConfig(input_dim=2, hidden_dim=4, output_dim=2))
        trainer = TWFCTrainer(
            model=model,
            config=TWFCConfig(
                observation_budget=4,
                propagation_budget=3,
                max_steps=8,
                seed=11,
            ),
        )

        result = trainer.fit(dataset)

        self.assertEqual(result.collapsed_count, 8)
        self.assertEqual(len(result.step_logs), 8)
        self.assertTrue(math.isfinite(result.initial_shadow_metrics.train_loss))
        self.assertTrue(math.isfinite(result.final_shadow_metrics.train_loss))
        self.assertTrue(math.isfinite(result.final_hard_metrics.test_loss))
        self.assertGreaterEqual(result.final_shadow_metrics.train_accuracy, 0.0)
        self.assertLessEqual(result.final_shadow_metrics.train_accuracy, 1.0)
        self.assertGreaterEqual(result.rollback_count, 0)
        self.assertGreaterEqual(result.forced_commit_count, 0)
        self.assertGreater(len(result.step_logs[0].propagated_indices), 0)
        self.assertTrue(math.isfinite(result.step_logs[0].hard_loss_after))
        self.assertTrue(math.isfinite(result.step_logs[0].score_after))
        self.assertEqual(len(result.snapshots), result.collapsed_count + 1)
        self.assertEqual(result.snapshots[0].rollback_count, 0)
        self.assertEqual(result.snapshots[0].backtrack_count, 0)
        self.assertEqual(result.snapshots[0].forced_commit_count, 0)
        self.assertEqual(result.snapshots[0].rollback_delta, 0)
        self.assertEqual(result.snapshots[0].backtrack_delta, 0)
        self.assertEqual(result.snapshots[0].forced_commit_delta, 0)
        self.assertEqual(result.snapshots[0].forbidden_value_count, 0)
        self.assertEqual(result.snapshots[0].forbidden_value_delta, 0)
        self.assertEqual(result.snapshots[0].frontier_pressure, 0)
        self.assertEqual(result.snapshots[-1].rollback_count, result.rollback_count)
        self.assertEqual(result.snapshots[-1].forced_commit_count, result.forced_commit_count)

    def test_iris_partial_run_reports_shadow_and_hard_metrics(self) -> None:
        dataset = load_iris_dataset(seed=13)
        model = ToyMLP(MLPConfig(input_dim=4, hidden_dim=8, output_dim=3))
        trainer = TWFCTrainer(
            model=model,
            config=TWFCConfig(
                observation_budget=6,
                propagation_budget=4,
                max_steps=12,
                seed=13,
            ),
        )

        result = trainer.fit(dataset)

        self.assertEqual(result.collapsed_count, 12)
        self.assertEqual(len(result.step_logs), 12)
        self.assertTrue(math.isfinite(result.final_shadow_metrics.test_loss))
        self.assertTrue(math.isfinite(result.final_hard_metrics.test_loss))
        self.assertGreaterEqual(result.final_hard_metrics.test_accuracy, 0.0)
        self.assertLessEqual(result.final_hard_metrics.test_accuracy, 1.0)

    def test_negative_tolerance_can_trigger_rollbacks(self) -> None:
        dataset = make_moons_dataset(n_samples=60, noise=0.04, seed=17)
        model = ToyMLP(MLPConfig(input_dim=2, hidden_dim=4, output_dim=2))
        trainer = TWFCTrainer(
            model=model,
            config=TWFCConfig(
                observation_budget=4,
                propagation_budget=3,
                max_steps=4,
                backtrack_tolerance=-10.0,
                rollback_depth=1,
                max_frontier_rollbacks=1,
                max_attempt_multiplier=12,
                seed=17,
            ),
        )

        result = trainer.fit(dataset)

        self.assertGreaterEqual(result.rollback_count, 1)
        self.assertGreaterEqual(result.forced_commit_count, 1)
        self.assertGreaterEqual(result.collapsed_count, 1)
        self.assertTrue(math.isfinite(result.final_shadow_metrics.test_loss))
        self.assertTrue(any(step.forced_commit for step in result.step_logs))
        self.assertTrue(any(snapshot.rollback_delta > 0 for snapshot in result.snapshots[1:]))
        self.assertTrue(any(snapshot.forced_commit_delta > 0 for snapshot in result.snapshots[1:]))
        self.assertGreater(max(snapshot.forbidden_value_count for snapshot in result.snapshots), 0)
        self.assertGreater(max(snapshot.frontier_pressure for snapshot in result.snapshots), 0)
        self.assertTrue(any(snapshot.forbidden_entries for snapshot in result.snapshots[1:]))
        self.assertTrue(any(snapshot.forbidden_delta_labels for snapshot in result.snapshots[1:]))
        first_ban_snapshot = next(snapshot for snapshot in result.snapshots[1:] if snapshot.forbidden_entries)
        self.assertGreaterEqual(first_ban_snapshot.forbidden_entries[0].ban_count, 1)
        self.assertIn("[", first_ban_snapshot.forbidden_entries[0].parameter_name)

    def test_make_moons_plot_is_written(self) -> None:
        dataset = make_moons_dataset(n_samples=60, noise=0.04, seed=19)
        model = ToyMLP(MLPConfig(input_dim=2, hidden_dim=4, output_dim=2))
        trainer = TWFCTrainer(
            model=model,
            config=TWFCConfig(
                observation_budget=4,
                propagation_budget=3,
                max_steps=6,
                seed=19,
            ),
        )
        result = trainer.fit(dataset)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "make_moons_plot.png"
            saved_path = save_experiment_plot(model, dataset, result, output_path)

            self.assertEqual(saved_path, output_path)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_make_moons_progress_plot_is_written(self) -> None:
        dataset = make_moons_dataset(n_samples=60, noise=0.04, seed=23)
        model = ToyMLP(MLPConfig(input_dim=2, hidden_dim=4, output_dim=2))
        trainer = TWFCTrainer(
            model=model,
            config=TWFCConfig(
                observation_budget=4,
                propagation_budget=3,
                max_steps=6,
                seed=23,
            ),
        )
        result = trainer.fit(dataset)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "make_moons_progress.png"
            saved_path = save_progress_plot(model, dataset, result, output_path, max_panels=4)

            self.assertEqual(saved_path, output_path)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_iris_metrics_plot_is_written(self) -> None:
        dataset = load_iris_dataset(seed=29)
        model = ToyMLP(MLPConfig(input_dim=4, hidden_dim=8, output_dim=3))
        trainer = TWFCTrainer(
            model=model,
            config=TWFCConfig(
                observation_budget=6,
                propagation_budget=4,
                max_steps=8,
                seed=29,
            ),
        )
        result = trainer.fit(dataset)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "iris_metrics.png"
            saved_path = save_metrics_plot(result, output_path)

            self.assertEqual(saved_path, output_path)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_make_moons_snapshot_frames_are_written(self) -> None:
        dataset = make_moons_dataset(n_samples=60, noise=0.04, seed=31)
        model = ToyMLP(MLPConfig(input_dim=2, hidden_dim=4, output_dim=2))
        trainer = TWFCTrainer(
            model=model,
            config=TWFCConfig(
                observation_budget=4,
                propagation_budget=3,
                max_steps=6,
                seed=31,
            ),
        )
        result = trainer.fit(dataset)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "frames"
            saved_paths = save_snapshot_frames(model, dataset, result, output_dir, max_frames=4)

            self.assertEqual(len(saved_paths), 4)
            for saved_path in saved_paths:
                self.assertTrue(saved_path.exists())
                self.assertGreater(saved_path.stat().st_size, 0)

    def test_make_moons_storyboard_is_written(self) -> None:
        dataset = make_moons_dataset(n_samples=60, noise=0.04, seed=37)
        model = ToyMLP(MLPConfig(input_dim=2, hidden_dim=4, output_dim=2))
        trainer = TWFCTrainer(
            model=model,
            config=TWFCConfig(
                observation_budget=4,
                propagation_budget=3,
                max_steps=6,
                seed=37,
            ),
        )
        result = trainer.fit(dataset)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "make_moons_storyboard.png"
            saved_path = save_storyboard_plot(model, dataset, result, output_path, max_panels=4)

            self.assertEqual(saved_path, output_path)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_make_moons_gif_is_written(self) -> None:
        dataset = make_moons_dataset(n_samples=60, noise=0.04, seed=41)
        model = ToyMLP(MLPConfig(input_dim=2, hidden_dim=4, output_dim=2))
        trainer = TWFCTrainer(
            model=model,
            config=TWFCConfig(
                observation_budget=4,
                propagation_budget=3,
                max_steps=5,
                seed=41,
            ),
        )
        result = trainer.fit(dataset)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "make_moons.gif"
            saved_path = save_snapshot_gif(model, dataset, result, output_path, max_frames=4, frame_duration_ms=250)

            self.assertEqual(saved_path, output_path)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_seed_gallery_and_markdown_report_are_written(self) -> None:
        experiments = run_seed_batch(
            "make_moons",
            (5, 9),
            samples=60,
            noise=0.04,
            hidden_dim=4,
            config_template=TWFCConfig(
                observation_budget=4,
                propagation_budget=3,
                max_steps=4,
                seed=5,
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            gallery_path = temp_root / "seed_gallery.png"
            seed_artifacts_dir = temp_root / "seed_artifacts"
            report_path = temp_root / "seed_report.md"

            saved_gallery = save_seed_gallery_plot(experiments, gallery_path, columns=2)
            saved_artifacts = export_seed_artifacts(
                experiments,
                seed_artifacts_dir,
                storyboard_panels=4,
                max_frames=3,
                gif_frame_duration_ms=220,
            )
            saved_report = save_seed_markdown_report(
                experiments,
                report_path,
                title="Seed Batch Report",
                gallery_path=saved_gallery,
                seed_artifacts=saved_artifacts,
                config_summary={"max_steps": 4, "observation_budget": 4},
            )

            self.assertEqual(saved_gallery, gallery_path)
            self.assertEqual(saved_report, report_path)
            self.assertEqual(len(saved_artifacts), 2)
            self.assertTrue(gallery_path.exists())
            self.assertGreater(gallery_path.stat().st_size, 0)
            for artifact in saved_artifacts:
                self.assertTrue(artifact.metrics_plot.exists())
                self.assertGreater(artifact.metrics_plot.stat().st_size, 0)
                self.assertIsNotNone(artifact.storyboard_plot)
                self.assertIsNotNone(artifact.gif_path)
                self.assertTrue(artifact.storyboard_plot.exists())
                self.assertTrue(artifact.gif_path.exists())
                self.assertGreater(artifact.storyboard_plot.stat().st_size, 0)
                self.assertGreater(artifact.gif_path.stat().st_size, 0)
            self.assertTrue(report_path.exists())
            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn("# Seed Batch Report", report_text)
            self.assertIn("## Highlights", report_text)
            self.assertIn("### Best Seed:", report_text)
            self.assertIn("### Worst Seed:", report_text)
            self.assertIn("| Seed | Tag | Collapsed |", report_text)
            self.assertIn("| Tag |", report_text)
            self.assertIn("Peak Ban Focus", report_text)
            self.assertIn("Latest Ban Delta", report_text)
            self.assertIn("| BEST |", report_text)
            self.assertIn("| WORST |", report_text)
            self.assertIn("seed_gallery.png", report_text)
            self.assertIn("## Seed Drilldown", report_text)
            self.assertIn("[metrics](", report_text)
            self.assertIn("[storyboard](", report_text)
            self.assertIn("[gif](", report_text)
            self.assertIn("make_moons_seed_005_storyboard.png", report_text)

    def test_markdown_report_can_surface_peak_ban_focus(self) -> None:
        experiments = run_seed_batch(
            "make_moons",
            (17,),
            samples=60,
            noise=0.04,
            hidden_dim=4,
            config_template=TWFCConfig(
                observation_budget=4,
                propagation_budget=3,
                max_steps=4,
                backtrack_tolerance=-10.0,
                rollback_depth=1,
                max_frontier_rollbacks=1,
                max_attempt_multiplier=12,
                seed=17,
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "stress_seed_report.md"
            saved_report = save_seed_markdown_report(
                experiments,
                report_path,
                title="Stress Seed Report",
                config_summary={"max_steps": 4, "backtrack_tolerance": -10.0},
            )

            self.assertEqual(saved_report, report_path)
            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn("Peak ban focus:", report_text)
            self.assertIn("Latest ban delta:", report_text)
            self.assertNotIn("Peak ban focus: `clean`", report_text)
            self.assertNotIn("Latest ban delta: `none`", report_text)
            self.assertIn("@ s", report_text)


if __name__ == "__main__":
    unittest.main()
