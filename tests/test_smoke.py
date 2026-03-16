from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from t_wfc.baseline import SGDBaselineConfig, train_sgd_classifier
from t_wfc.batch import export_seed_artifacts, run_seed_batch
from t_wfc.data import load_dataset, load_iris_dataset, make_moons_dataset, make_spiral_dataset
from t_wfc.model import MLPConfig, ToyMLP
from t_wfc.reporting import save_seed_markdown_report
from t_wfc.state import WeightState
from t_wfc.trainer import (
    DEFAULT_OBSERVATION_TEMPERATURE,
    MULTILAYER_OBSERVATION_TEMPERATURE,
    CollapseStep,
    DecisionRecord,
    EvaluationMetrics,
    ExperimentSnapshot,
    TWFCConfig,
    TWFCTrainer,
)
from t_wfc.visualization import (
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


class SpiralDatasetTest(unittest.TestCase):
    def test_dataset_shapes_and_labels(self) -> None:
        dataset = load_dataset("spiral", n_samples=150, noise=0.14, seed=7)

        self.assertEqual(dataset.x_train.shape[1], 2)
        self.assertEqual(dataset.x_test.shape[1], 2)
        self.assertSetEqual(set(dataset.y_train.tolist()) | set(dataset.y_test.tolist()), {0, 1, 2})
        self.assertGreaterEqual(int((dataset.y_test == 0).sum()), 1)
        self.assertGreaterEqual(int((dataset.y_test == 1).sum()), 1)
        self.assertGreaterEqual(int((dataset.y_test == 2).sum()), 1)


class MultiLayerModelSmokeTest(unittest.TestCase):
    def test_multilayer_model_supports_forward_and_gradient(self) -> None:
        model = ToyMLP(MLPConfig(input_dim=2, output_dim=3, hidden_layers=(5, 4)))
        weights = model.random_vector(np.random.default_rng(3))
        features = np.array(
            [[0.1, -0.2], [0.3, 0.4], [-0.5, 0.2]],
            dtype=float,
        )
        labels = np.array([0, 2, 1], dtype=int)

        logits = model.forward(weights, features)
        loss, gradient = model.loss_and_gradient(weights, features, labels)

        self.assertEqual(logits.shape, (3, 3))
        self.assertEqual(gradient.shape, (model.parameter_count,))
        self.assertTrue(math.isfinite(loss))
        self.assertEqual(model.architecture_label, "2-5-4-3")


class ObservationTemperatureResolutionTest(unittest.TestCase):
    def test_single_hidden_layer_auto_temperature_keeps_baseline_default(self) -> None:
        model = ToyMLP(MLPConfig(input_dim=2, hidden_dim=4, output_dim=2))

        auto_trainer = TWFCTrainer(model=model, config=TWFCConfig(seed=3))
        explicit_trainer = TWFCTrainer(
            model=model,
            config=TWFCConfig(
                temperature=DEFAULT_OBSERVATION_TEMPERATURE,
                seed=3,
            ),
        )

        self.assertAlmostEqual(auto_trainer.temperature, DEFAULT_OBSERVATION_TEMPERATURE)
        self.assertAlmostEqual(explicit_trainer.temperature, DEFAULT_OBSERVATION_TEMPERATURE)

    def test_multilayer_auto_temperature_uses_sharper_default_without_overriding_explicit_values(self) -> None:
        model = ToyMLP(MLPConfig(input_dim=4, output_dim=3, hidden_layers=(8, 8)))

        auto_trainer = TWFCTrainer(model=model, config=TWFCConfig(seed=7))
        explicit_auto_trainer = TWFCTrainer(
            model=model,
            config=TWFCConfig(
                temperature=MULTILAYER_OBSERVATION_TEMPERATURE,
                seed=7,
            ),
        )
        explicit_override_trainer = TWFCTrainer(
            model=model,
            config=TWFCConfig(
                temperature=DEFAULT_OBSERVATION_TEMPERATURE,
                seed=7,
            ),
        )

        self.assertAlmostEqual(auto_trainer.temperature, MULTILAYER_OBSERVATION_TEMPERATURE)
        self.assertAlmostEqual(explicit_auto_trainer.temperature, MULTILAYER_OBSERVATION_TEMPERATURE)
        self.assertAlmostEqual(explicit_override_trainer.temperature, DEFAULT_OBSERVATION_TEMPERATURE)

    def test_multilayer_auto_temperature_matches_explicit_run(self) -> None:
        dataset = load_iris_dataset(seed=7)
        model = ToyMLP(MLPConfig(input_dim=4, output_dim=3, hidden_layers=(8, 8)))

        auto_result = TWFCTrainer(
            model=model,
            config=TWFCConfig(
                initial_jitter=0.08,
                observation_budget=6,
                propagation_budget=4,
                max_steps=6,
                seed=7,
            ),
        ).fit(dataset)
        explicit_result = TWFCTrainer(
            model=model,
            config=TWFCConfig(
                initial_jitter=0.08,
                observation_budget=6,
                propagation_budget=4,
                max_steps=6,
                temperature=MULTILAYER_OBSERVATION_TEMPERATURE,
                seed=7,
            ),
        ).fit(dataset)

        np.testing.assert_allclose(auto_result.shadow_weights, explicit_result.shadow_weights)
        np.testing.assert_allclose(auto_result.hard_weights, explicit_result.hard_weights)
        self.assertEqual(
            [
                (step.weight_index, step.chosen_value, step.backtracked, step.forced_commit)
                for step in auto_result.step_logs
            ],
            [
                (step.weight_index, step.chosen_value, step.backtracked, step.forced_commit)
                for step in explicit_result.step_logs
            ],
        )


class RollbackGeneralizationTest(unittest.TestCase):
    @staticmethod
    def _dummy_metrics() -> EvaluationMetrics:
        return EvaluationMetrics(
            train_loss=0.0,
            test_loss=0.0,
            train_accuracy=0.0,
            test_accuracy=0.0,
        )

    @classmethod
    def _dummy_snapshot(cls, *, step: int, parameter_count: int) -> ExperimentSnapshot:
        zero = np.zeros(parameter_count, dtype=float)
        probabilities = np.full((parameter_count, 5), 0.2, dtype=float)
        collapsed_mask = np.zeros(parameter_count, dtype=bool)
        metrics = cls._dummy_metrics()
        return ExperimentSnapshot(
            step=step,
            collapsed_count=step,
            rollback_count=0,
            backtrack_count=0,
            forced_commit_count=0,
            forbidden_weight_count=0,
            forbidden_value_count=0,
            frontier_pressure=0,
            rollback_delta=0,
            backtrack_delta=0,
            forced_commit_delta=0,
            forbidden_value_delta=0,
            forbidden_entries=(),
            forbidden_delta_labels=(),
            event_tags=(),
            collapsed_mask=collapsed_mask,
            distribution_snapshot=probabilities,
            shadow_weights=zero.copy(),
            hard_weights=zero.copy(),
            shadow_metrics=metrics,
            hard_metrics=metrics,
        )

    @staticmethod
    def _dummy_step(*, step: int, weight_index: int, chosen_value: float) -> CollapseStep:
        return CollapseStep(
            step=step,
            weight_index=weight_index,
            parameter_name=f"w[{weight_index}]",
            chosen_value=chosen_value,
            entropy=0.0,
            loss_before=0.0,
            loss_after=0.0,
            hard_loss_after=0.0,
            score_after=0.0,
            propagated_indices=(),
            backtracked=False,
            forced_commit=False,
        )

    def test_rollback_depth_can_grow_with_frontier_pressure(self) -> None:
        model = ToyMLP(MLPConfig(input_dim=2, hidden_dim=2, output_dim=2))
        trainer = TWFCTrainer(
            model=model,
            config=TWFCConfig(
                rollback_depth=1,
                rollback_depth_growth=1,
                seed=3,
            ),
        )

        self.assertEqual(trainer._resolve_rollback_depth(history_length=0, frontier_pressure=1), 0)
        self.assertEqual(trainer._resolve_rollback_depth(history_length=1, frontier_pressure=1), 1)
        self.assertEqual(trainer._resolve_rollback_depth(history_length=4, frontier_pressure=2), 2)
        self.assertEqual(trainer._resolve_rollback_depth(history_length=4, frontier_pressure=5), 4)

    def test_default_rollback_still_bans_only_anchor_choice(self) -> None:
        model = ToyMLP(MLPConfig(input_dim=2, hidden_dim=2, output_dim=2))
        trainer = TWFCTrainer(
            model=model,
            config=TWFCConfig(
                rollback_depth=2,
                seed=5,
            ),
        )
        state = WeightState.uniform(model.parameter_count, trainer.domain)
        history: list[DecisionRecord] = []
        step_logs: list[CollapseStep] = []
        progress_snapshots = [self._dummy_snapshot(step=0, parameter_count=model.parameter_count)]

        first_snapshot = state.snapshot()
        state.collapse(0, 1)
        history.append(DecisionRecord(snapshot_before=first_snapshot, weight_index=0, chosen_value_index=1))
        step_logs.append(self._dummy_step(step=1, weight_index=0, chosen_value=float(trainer.domain[1])))
        progress_snapshots.append(self._dummy_snapshot(step=1, parameter_count=model.parameter_count))

        second_snapshot = state.snapshot()
        state.collapse(1, 3)
        history.append(DecisionRecord(snapshot_before=second_snapshot, weight_index=1, chosen_value_index=3))
        step_logs.append(self._dummy_step(step=2, weight_index=1, chosen_value=float(trainer.domain[3])))
        progress_snapshots.append(self._dummy_snapshot(step=2, parameter_count=model.parameter_count))

        forbidden_values: dict[int, set[int]] = {}
        trainer._rollback(
            state=state,
            history=history,
            step_logs=step_logs,
            progress_snapshots=progress_snapshots,
            forbidden_values=forbidden_values,
            frontier_pressure=3,
        )

        self.assertFalse(bool(state.collapsed.any()))
        self.assertEqual(len(history), 0)
        self.assertEqual(len(step_logs), 0)
        self.assertEqual(len(progress_snapshots), 1)
        self.assertEqual(forbidden_values, {0: {1}})

    def test_rollback_can_ban_multiple_reverted_choices(self) -> None:
        model = ToyMLP(MLPConfig(input_dim=2, hidden_dim=2, output_dim=2))
        trainer = TWFCTrainer(
            model=model,
            config=TWFCConfig(
                rollback_depth=1,
                rollback_depth_growth=1,
                rollback_ban_count=2,
                seed=7,
            ),
        )
        state = WeightState.uniform(model.parameter_count, trainer.domain)
        history: list[DecisionRecord] = []
        step_logs: list[CollapseStep] = []
        progress_snapshots = [self._dummy_snapshot(step=0, parameter_count=model.parameter_count)]

        first_snapshot = state.snapshot()
        state.collapse(0, 1)
        history.append(DecisionRecord(snapshot_before=first_snapshot, weight_index=0, chosen_value_index=1))
        step_logs.append(self._dummy_step(step=1, weight_index=0, chosen_value=float(trainer.domain[1])))
        progress_snapshots.append(self._dummy_snapshot(step=1, parameter_count=model.parameter_count))

        second_snapshot = state.snapshot()
        state.collapse(1, 3)
        history.append(DecisionRecord(snapshot_before=second_snapshot, weight_index=1, chosen_value_index=3))
        step_logs.append(self._dummy_step(step=2, weight_index=1, chosen_value=float(trainer.domain[3])))
        progress_snapshots.append(self._dummy_snapshot(step=2, parameter_count=model.parameter_count))

        forbidden_values: dict[int, set[int]] = {}
        trainer._rollback(
            state=state,
            history=history,
            step_logs=step_logs,
            progress_snapshots=progress_snapshots,
            forbidden_values=forbidden_values,
            frontier_pressure=2,
        )

        self.assertFalse(bool(state.collapsed.any()))
        self.assertEqual(len(history), 0)
        self.assertEqual(len(step_logs), 0)
        self.assertEqual(len(progress_snapshots), 1)
        self.assertEqual(forbidden_values, {0: {1}, 1: {3}})

    def test_trim_frontier_rollbacks_drops_only_skipped_frontiers(self) -> None:
        model = ToyMLP(MLPConfig(input_dim=2, hidden_dim=2, output_dim=2))
        trainer = TWFCTrainer(
            model=model,
            config=TWFCConfig(seed=11),
        )
        frontier_rollbacks = {0: 1, 1: 2, 2: 4, 3: 6}

        trainer._trim_frontier_rollbacks(
            frontier_rollbacks,
            current_frontier=1,
            previous_frontier=3,
        )

        self.assertEqual(frontier_rollbacks, {0: 1, 1: 2, 3: 6})


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

        self.assertEqual(result.context.dataset_name, "make_moons")
        self.assertEqual(result.context.dataset_seed, 11)
        self.assertEqual(result.context.feature_dim, 2)
        self.assertEqual(result.context.class_count, 2)
        self.assertEqual(result.context.architecture_label, model.architecture_label)
        self.assertEqual(result.context.parameter_count, model.parameter_count)
        self.assertEqual(result.context.domain_size, 5)
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
        self.assertEqual(result.snapshots[0].event_tags, ())
        self.assertEqual(result.snapshots[0].distribution_snapshot.shape, (model.parameter_count, 5))
        self.assertEqual(result.snapshots[0].collapsed_mask.shape, (model.parameter_count,))
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

        self.assertEqual(result.context.dataset_name, "iris")
        self.assertEqual(result.context.dataset_seed, 13)
        self.assertEqual(result.context.feature_dim, 4)
        self.assertEqual(result.context.class_count, 3)
        self.assertEqual(result.collapsed_count, 12)
        self.assertEqual(len(result.step_logs), 12)
        self.assertTrue(math.isfinite(result.final_shadow_metrics.test_loss))
        self.assertTrue(math.isfinite(result.final_hard_metrics.test_loss))
        self.assertGreaterEqual(result.final_hard_metrics.test_accuracy, 0.0)
        self.assertLessEqual(result.final_hard_metrics.test_accuracy, 1.0)

    def test_spiral_partial_run_supports_multilayer_model(self) -> None:
        dataset = make_spiral_dataset(n_samples=180, noise=0.14, seed=21)
        model = ToyMLP(MLPConfig(input_dim=2, output_dim=3, hidden_layers=(12, 12)))
        trainer = TWFCTrainer(
            model=model,
            config=TWFCConfig(
                initial_jitter=0.08,
                observation_budget=10,
                propagation_budget=8,
                max_steps=18,
                seed=21,
            ),
        )

        result = trainer.fit(dataset)

        self.assertEqual(result.context.dataset_name, "spiral")
        self.assertEqual(result.context.dataset_seed, 21)
        self.assertEqual(result.context.feature_dim, 2)
        self.assertEqual(result.context.class_count, 3)
        self.assertEqual(result.collapsed_count, 18)
        self.assertEqual(result.parameter_count, model.parameter_count)
        self.assertGreater(model.parameter_count, 150)
        self.assertTrue(math.isfinite(result.final_shadow_metrics.test_loss))
        self.assertTrue(math.isfinite(result.final_hard_metrics.test_loss))
        self.assertNotEqual(result.initial_shadow_metrics.test_loss, result.final_shadow_metrics.test_loss)
        self.assertGreaterEqual(result.final_hard_metrics.test_accuracy, 0.0)
        self.assertLessEqual(result.final_hard_metrics.test_accuracy, 1.0)
        self.assertGreaterEqual(result.final_hard_metrics.test_accuracy, result.initial_shadow_metrics.test_accuracy)

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
        self.assertTrue(any("rollback" in snapshot.event_tags for snapshot in result.snapshots[1:]))
        self.assertTrue(any("forced" in snapshot.event_tags for snapshot in result.snapshots[1:]))
        self.assertTrue(any("ban_growth" in snapshot.event_tags for snapshot in result.snapshots[1:]))
        self.assertTrue(any("frontier_pressure" in snapshot.event_tags for snapshot in result.snapshots[1:]))
        first_ban_snapshot = next(snapshot for snapshot in result.snapshots[1:] if snapshot.forbidden_entries)
        self.assertGreaterEqual(first_ban_snapshot.forbidden_entries[0].ban_count, 1)
        self.assertIn("[", first_ban_snapshot.forbidden_entries[0].parameter_name)
        self.assertEqual(first_ban_snapshot.distribution_snapshot.shape, (model.parameter_count, 5))

    def test_generalized_rollback_can_accumulate_multi_weight_bans(self) -> None:
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
                rollback_depth_growth=1,
                rollback_ban_count=2,
                max_frontier_rollbacks=2,
                max_attempt_multiplier=16,
                seed=17,
            ),
        )

        result = trainer.fit(dataset)

        self.assertGreaterEqual(result.rollback_count, 1)
        self.assertGreaterEqual(result.collapsed_count, 1)
        self.assertTrue(any(snapshot.frontier_pressure >= 1 for snapshot in result.snapshots[1:]))
        self.assertTrue(any(snapshot.forbidden_value_count >= 2 for snapshot in result.snapshots[1:]))
        self.assertTrue(
            any(
                len(snapshot.forbidden_entries) >= 2 or any(entry.ban_count >= 2 for entry in snapshot.forbidden_entries)
                for snapshot in result.snapshots[1:]
            )
        )

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

    def test_twfc_vs_sgd_comparison_artifacts_are_written(self) -> None:
        dataset = make_moons_dataset(n_samples=80, noise=0.06, seed=43)
        model = ToyMLP(MLPConfig(input_dim=2, hidden_dim=6, output_dim=2))
        trainer = TWFCTrainer(
            model=model,
            config=TWFCConfig(
                observation_budget=5,
                propagation_budget=4,
                max_steps=6,
                seed=43,
            ),
        )
        result = trainer.fit(dataset)
        baseline_result = train_sgd_classifier(
            model=model,
            dataset=dataset,
            config=SGDBaselineConfig(
                epochs=80,
                learning_rate=0.08,
                learning_rate_decay=0.01,
                batch_size=16,
                seed=43,
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            metrics_path = temp_root / "twfc_vs_sgd_metrics.png"
            comparison_plot_path = temp_root / "twfc_vs_sgd_boundaries.png"
            comparison_gif_path = temp_root / "twfc_vs_sgd.gif"

            saved_metrics = save_baseline_metrics_comparison_plot(result, baseline_result, metrics_path)
            saved_plot = save_baseline_comparison_plot(model, dataset, result, baseline_result, comparison_plot_path)
            saved_gif = save_baseline_comparison_gif(
                model,
                dataset,
                result,
                baseline_result,
                comparison_gif_path,
                max_frames=4,
                frame_duration_ms=220,
            )

            self.assertEqual(saved_metrics, metrics_path)
            self.assertEqual(saved_plot, comparison_plot_path)
            self.assertEqual(saved_gif, comparison_gif_path)
            self.assertTrue(metrics_path.exists())
            self.assertTrue(comparison_plot_path.exists())
            self.assertTrue(comparison_gif_path.exists())
            self.assertGreater(metrics_path.stat().st_size, 0)
            self.assertGreater(comparison_plot_path.stat().st_size, 0)
            self.assertGreater(comparison_gif_path.stat().st_size, 0)

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
            self.assertIn("## Shadow vs Hard Divergence", report_text)
            self.assertIn("## Highlights", report_text)
            self.assertIn("### Best Seed:", report_text)
            self.assertIn("### Worst Seed:", report_text)
            self.assertIn("| Seed | Tag | Collapsed |", report_text)
            self.assertIn("Acc Gap (S-H)", report_text)
            self.assertIn("Loss Gap (H-S)", report_text)
            self.assertIn("| Tag |", report_text)
            self.assertIn("Mean shadow-hard test accuracy gap", report_text)
            self.assertIn("Seeds where hard test accuracy matched or exceeded shadow", report_text)
            self.assertIn("Largest shadow accuracy lead", report_text)
            self.assertIn("Shadow vs hard:", report_text)
            self.assertIn("Peak Ban Focus", report_text)
            self.assertIn("Latest Ban Delta", report_text)
            self.assertIn("| BEST |", report_text)
            self.assertIn("| WORST |", report_text)
            self.assertIn("seed_gallery.png", report_text)
            self.assertIn("![Best Seed preview: storyboard](", report_text)
            self.assertIn("![Best Seed preview: metrics](", report_text)
            self.assertIn("![Worst Seed preview: storyboard](", report_text)
            self.assertIn("![Worst Seed preview: metrics](", report_text)
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
            self.assertIn("## Shadow vs Hard Divergence", report_text)
            self.assertIn("Largest hard accuracy lead", report_text)
            self.assertIn("Peak ban focus:", report_text)
            self.assertIn("Latest ban delta:", report_text)
            self.assertIn("Shadow vs hard:", report_text)
            self.assertNotIn("preview: storyboard", report_text)
            self.assertNotIn("preview: metrics", report_text)
            self.assertNotIn("Peak ban focus: `clean`", report_text)
            self.assertNotIn("Latest ban delta: `none`", report_text)
            self.assertIn("@ s", report_text)


class SGDBaselineSmokeTest(unittest.TestCase):
    def test_sgd_baseline_improves_on_iris_with_multilayer_model(self) -> None:
        dataset = load_iris_dataset(seed=9)
        model = ToyMLP(MLPConfig(input_dim=4, output_dim=3, hidden_layers=(16, 16)))

        result = train_sgd_classifier(
            model=model,
            dataset=dataset,
            config=SGDBaselineConfig(
                epochs=160,
                learning_rate=0.09,
                learning_rate_decay=0.01,
                batch_size=24,
                seed=9,
            ),
        )

        self.assertGreater(result.final_metrics.train_accuracy, result.initial_metrics.train_accuracy)
        self.assertGreater(result.final_metrics.test_accuracy, result.initial_metrics.test_accuracy)
        self.assertLess(result.final_metrics.train_loss, result.initial_metrics.train_loss)
        self.assertLess(result.final_metrics.test_loss, result.initial_metrics.test_loss)
        self.assertGreaterEqual(result.final_metrics.test_accuracy, 0.85)


if __name__ == "__main__":
    unittest.main()
