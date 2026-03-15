from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .data import DatasetSplit
from .model import ToyMLP
from .state import StateSnapshot, WeightState


@dataclass(frozen=True)
class TWFCConfig:
    domain: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    initial_jitter: float = 0.0
    observation_budget: int = 8
    propagation_budget: int = 6
    max_steps: int | None = None
    temperature: float = 0.18
    propagation_blend: float = 0.6
    backtrack_tolerance: float = 0.03
    hard_loss_weight: float = 0.35
    hard_gap_weight: float = 0.2
    rollback_depth: int = 2
    max_frontier_rollbacks: int = 3
    max_attempt_multiplier: int = 8
    seed: int = 7


@dataclass(frozen=True)
class WeightObservation:
    weight_index: int
    parameter_name: str
    entropy: float
    losses: np.ndarray
    posterior: np.ndarray


@dataclass(frozen=True)
class CollapseStep:
    step: int
    weight_index: int
    parameter_name: str
    chosen_value: float
    entropy: float
    loss_before: float
    loss_after: float
    hard_loss_after: float
    score_after: float
    propagated_indices: tuple[int, ...]
    backtracked: bool
    forced_commit: bool


@dataclass(frozen=True)
class CollapseOutcome:
    step_log: CollapseStep
    chosen_value_index: int
    accepted: bool


@dataclass(frozen=True)
class EvaluationMetrics:
    train_loss: float
    test_loss: float
    train_accuracy: float
    test_accuracy: float


@dataclass(frozen=True)
class ForbiddenWeightState:
    weight_index: int
    parameter_name: str
    ban_count: int
    banned_values: tuple[float, ...]


@dataclass(frozen=True)
class ExperimentSnapshot:
    step: int
    collapsed_count: int
    rollback_count: int
    backtrack_count: int
    forced_commit_count: int
    forbidden_weight_count: int
    forbidden_value_count: int
    frontier_pressure: int
    rollback_delta: int
    backtrack_delta: int
    forced_commit_delta: int
    forbidden_value_delta: int
    forbidden_entries: tuple[ForbiddenWeightState, ...]
    forbidden_delta_labels: tuple[str, ...]
    shadow_weights: np.ndarray
    hard_weights: np.ndarray
    shadow_metrics: EvaluationMetrics
    hard_metrics: EvaluationMetrics


@dataclass(frozen=True)
class DecisionRecord:
    snapshot_before: StateSnapshot
    weight_index: int
    chosen_value_index: int


@dataclass(frozen=True)
class ExperimentResult:
    initial_shadow_weights: np.ndarray
    initial_shadow_metrics: EvaluationMetrics
    final_shadow_metrics: EvaluationMetrics
    final_hard_metrics: EvaluationMetrics
    collapsed_count: int
    parameter_count: int
    backtrack_count: int
    rollback_count: int
    forced_commit_count: int
    step_logs: tuple[CollapseStep, ...]
    snapshots: tuple[ExperimentSnapshot, ...]
    shadow_weights: np.ndarray
    hard_weights: np.ndarray


class TWFCTrainer:
    def __init__(self, model: ToyMLP, config: TWFCConfig) -> None:
        self.model = model
        self.config = config
        self.domain = np.asarray(config.domain, dtype=np.float64)
        self.rng = np.random.default_rng(config.seed)
        self.neighbor_map = self._build_neighbor_map()

    def fit(self, dataset: DatasetSplit) -> ExperimentResult:
        state = WeightState.uniform(self.model.parameter_count, self.domain)
        self._apply_initial_jitter(state)

        initial_shadow = state.expected_vector()
        initial_shadow_metrics = self._evaluate(initial_shadow, dataset)

        step_logs: list[CollapseStep] = []
        history: list[DecisionRecord] = []
        forbidden_values: dict[int, set[int]] = {}
        frontier_rollbacks: dict[int, int] = {}
        progress_snapshots: list[ExperimentSnapshot] = [
            self._build_progress_snapshot(
                step=0,
                state=state,
                dataset=dataset,
                rollback_count=0,
                backtrack_count=0,
                forced_commit_count=0,
                forbidden_values=forbidden_values,
                frontier_pressure=0,
                previous_snapshot=None,
            )
        ]
        max_steps = self.config.max_steps or self.model.parameter_count
        attempt_limit = max_steps * max(self.config.max_attempt_multiplier, 1)
        rollback_count = 0
        forced_commit_count = 0
        attempts = 0

        while len(history) < max_steps and attempts < attempt_limit:
            attempts += 1

            unresolved = state.unresolved_indices()
            if unresolved.size == 0:
                break

            step_number = len(history) + 1
            frontier = len(history)
            snapshot_before = state.snapshot()
            force_commit_mode = history and (
                frontier_rollbacks.get(frontier, 0) >= max(self.config.max_frontier_rollbacks, 0)
            )
            if force_commit_mode:
                observation, outcome = self._select_forced_commit_action(
                    step=step_number,
                    state=state,
                    features=dataset.x_train,
                    labels=dataset.y_train,
                    forbidden_values=forbidden_values,
                )
            else:
                observation = self._select_next_weight(
                    state,
                    dataset.x_train,
                    dataset.y_train,
                    forbidden_values=forbidden_values,
                )
                outcome = self._collapse_with_backtracking(
                    step=step_number,
                    state=state,
                    observation=observation,
                    features=dataset.x_train,
                    labels=dataset.y_train,
                    forbidden_values=forbidden_values,
                )

            force_commit = force_commit_mode or (not history and not outcome.accepted)

            if outcome.accepted or force_commit:
                if force_commit and not outcome.step_log.forced_commit:
                    outcome = replace(
                        outcome,
                        step_log=replace(outcome.step_log, forced_commit=True),
                    )
                    forced_commit_count += 1

                history.append(
                    DecisionRecord(
                        snapshot_before=snapshot_before,
                        weight_index=observation.weight_index,
                        chosen_value_index=outcome.chosen_value_index,
                    )
                )
                step_logs.append(outcome.step_log)
                progress_snapshots.append(
                    self._build_progress_snapshot(
                        step=step_number,
                        state=state,
                        dataset=dataset,
                        rollback_count=rollback_count,
                        backtrack_count=sum(1 for step_log in step_logs if step_log.backtracked),
                        forced_commit_count=forced_commit_count,
                        forbidden_values=forbidden_values,
                        frontier_pressure=frontier_rollbacks.get(frontier, 0),
                        previous_snapshot=progress_snapshots[-1],
                    )
                )
                frontier_rollbacks[frontier] = 0
                continue

            rollback_count += 1
            frontier_rollbacks[frontier] = frontier_rollbacks.get(frontier, 0) + 1
            self._rollback(
                state=state,
                history=history,
                step_logs=step_logs,
                progress_snapshots=progress_snapshots,
                forbidden_values=forbidden_values,
            )

        final_shadow = state.expected_vector()
        final_hard = state.argmax_vector()

        return ExperimentResult(
            initial_shadow_weights=initial_shadow,
            initial_shadow_metrics=initial_shadow_metrics,
            final_shadow_metrics=self._evaluate(final_shadow, dataset),
            final_hard_metrics=self._evaluate(final_hard, dataset),
            collapsed_count=int(state.collapsed.sum()),
            parameter_count=self.model.parameter_count,
            backtrack_count=sum(1 for step_log in step_logs if step_log.backtracked),
            rollback_count=rollback_count,
            forced_commit_count=forced_commit_count,
            step_logs=tuple(step_logs),
            snapshots=tuple(progress_snapshots),
            shadow_weights=final_shadow,
            hard_weights=final_hard,
        )

    def _evaluate(self, flat_weights: np.ndarray, dataset: DatasetSplit) -> EvaluationMetrics:
        return EvaluationMetrics(
            train_loss=self.model.loss(flat_weights, dataset.x_train, dataset.y_train),
            test_loss=self.model.loss(flat_weights, dataset.x_test, dataset.y_test),
            train_accuracy=self.model.accuracy(flat_weights, dataset.x_train, dataset.y_train),
            test_accuracy=self.model.accuracy(flat_weights, dataset.x_test, dataset.y_test),
        )

    def _build_progress_snapshot(
        self,
        step: int,
        state: WeightState,
        dataset: DatasetSplit,
        rollback_count: int,
        backtrack_count: int,
        forced_commit_count: int,
        forbidden_values: dict[int, set[int]],
        frontier_pressure: int,
        previous_snapshot: ExperimentSnapshot | None = None,
    ) -> ExperimentSnapshot:
        shadow_weights = state.expected_vector()
        hard_weights = state.argmax_vector()
        previous = previous_snapshot
        forbidden_entries = self._forbidden_entries(forbidden_values)
        forbidden_weight_count, forbidden_value_count = self._forbidden_stats(forbidden_values)
        return ExperimentSnapshot(
            step=step,
            collapsed_count=int(state.collapsed.sum()),
            rollback_count=rollback_count,
            backtrack_count=backtrack_count,
            forced_commit_count=forced_commit_count,
            forbidden_weight_count=forbidden_weight_count,
            forbidden_value_count=forbidden_value_count,
            frontier_pressure=frontier_pressure,
            rollback_delta=rollback_count - (previous.rollback_count if previous is not None else 0),
            backtrack_delta=backtrack_count - (previous.backtrack_count if previous is not None else 0),
            forced_commit_delta=forced_commit_count - (previous.forced_commit_count if previous is not None else 0),
            forbidden_value_delta=forbidden_value_count - (previous.forbidden_value_count if previous is not None else 0),
            forbidden_entries=forbidden_entries,
            forbidden_delta_labels=self._forbidden_delta_labels(
                forbidden_entries,
                previous.forbidden_entries if previous is not None else (),
            ),
            shadow_weights=shadow_weights,
            hard_weights=hard_weights,
            shadow_metrics=self._evaluate(shadow_weights, dataset),
            hard_metrics=self._evaluate(hard_weights, dataset),
        )

    def _select_next_weight(
        self,
        state: WeightState,
        features: np.ndarray,
        labels: np.ndarray,
        forbidden_values: dict[int, set[int]] | None = None,
    ) -> WeightObservation:
        unresolved = state.unresolved_indices()
        candidate_indices = self._sample_observation_indices(unresolved)
        base_vector = state.expected_vector()

        observations = [
            self._observe_weight(
                state=state,
                weight_index=int(weight_index),
                features=features,
                labels=labels,
                base_vector=base_vector,
                forbidden_values=forbidden_values,
            )
            for weight_index in candidate_indices
        ]

        return min(
            observations,
            key=lambda observation: (
                observation.entropy,
                float(observation.losses.min()),
                observation.weight_index,
            ),
        )

    def _sample_observation_indices(self, unresolved: np.ndarray) -> np.ndarray:
        budget = self.config.observation_budget
        if budget <= 0 or unresolved.size <= budget:
            return unresolved
        return self.rng.choice(unresolved, size=budget, replace=False)

    def _observe_weight(
        self,
        state: WeightState,
        weight_index: int,
        features: np.ndarray,
        labels: np.ndarray,
        base_vector: np.ndarray | None = None,
        forbidden_values: dict[int, set[int]] | None = None,
    ) -> WeightObservation:
        current_vector = state.expected_vector() if base_vector is None else base_vector
        losses = np.full(self.domain.size, np.inf, dtype=np.float64)
        allowed_value_indices = self._allowed_value_indices(weight_index, forbidden_values)

        for domain_index in allowed_value_indices:
            trial_vector = current_vector.copy()
            trial_vector[weight_index] = self.domain[domain_index]
            losses[domain_index] = self.model.loss(trial_vector, features, labels)

        posterior = self._loss_to_distribution(losses)
        return WeightObservation(
            weight_index=weight_index,
            parameter_name=self.model.parameter_layout[weight_index].name,
            entropy=self._entropy(posterior),
            losses=losses,
            posterior=posterior,
        )

    def _collapse_with_backtracking(
        self,
        step: int,
        state: WeightState,
        observation: WeightObservation,
        features: np.ndarray,
        labels: np.ndarray,
        forbidden_values: dict[int, set[int]] | None = None,
    ) -> CollapseOutcome:
        loss_before = self.model.loss(state.expected_vector(), features, labels)
        ranked_value_indices = np.argsort(observation.losses)
        initial_snapshot = state.snapshot()
        best_trial_log: CollapseStep | None = None
        best_trial_snapshot: StateSnapshot | None = None
        best_value_index: int | None = None

        for rank, value_index in enumerate(ranked_value_indices):
            if not np.isfinite(observation.losses[value_index]):
                continue

            state.restore(initial_snapshot)
            state.collapse(observation.weight_index, int(value_index))
            propagated_indices = self._propagate(
                state,
                observation.weight_index,
                features,
                labels,
                forbidden_values=forbidden_values,
            )
            loss_after = self.model.loss(state.expected_vector(), features, labels)
            hard_loss_after = self.model.loss(state.argmax_vector(), features, labels)
            score_after = self._trial_score(loss_after, hard_loss_after)

            trial_log = CollapseStep(
                step=step,
                weight_index=observation.weight_index,
                parameter_name=observation.parameter_name,
                chosen_value=float(self.domain[value_index]),
                entropy=observation.entropy,
                loss_before=loss_before,
                loss_after=loss_after,
                hard_loss_after=hard_loss_after,
                score_after=score_after,
                propagated_indices=propagated_indices,
                backtracked=rank > 0,
                forced_commit=False,
            )

            if best_trial_log is None or score_after < best_trial_log.score_after:
                best_trial_log = trial_log
                best_trial_snapshot = state.snapshot()
                best_value_index = int(value_index)

            if loss_after <= loss_before + self.config.backtrack_tolerance:
                return CollapseOutcome(
                    step_log=trial_log,
                    chosen_value_index=int(value_index),
                    accepted=True,
                )

        if best_trial_log is None or best_trial_snapshot is None or best_value_index is None:
            raise RuntimeError("No collapse candidate was evaluated")

        state.restore(best_trial_snapshot)
        return CollapseOutcome(
            step_log=best_trial_log,
            chosen_value_index=best_value_index,
            accepted=False,
        )

    def _select_forced_commit_action(
        self,
        step: int,
        state: WeightState,
        features: np.ndarray,
        labels: np.ndarray,
        forbidden_values: dict[int, set[int]] | None = None,
    ) -> tuple[WeightObservation, CollapseOutcome]:
        initial_snapshot = state.snapshot()
        base_vector = state.expected_vector()
        best_weight_index: int | None = None
        best_outcome: CollapseOutcome | None = None

        for weight_index in state.unresolved_indices():
            state.restore(initial_snapshot)
            observation = self._observe_weight(
                state=state,
                weight_index=int(weight_index),
                features=features,
                labels=labels,
                base_vector=base_vector,
                forbidden_values=forbidden_values,
            )
            outcome = self._collapse_with_backtracking(
                step=step,
                state=state,
                observation=observation,
                features=features,
                labels=labels,
                forbidden_values=forbidden_values,
            )

            if best_outcome is None or outcome.step_log.score_after < best_outcome.step_log.score_after:
                best_weight_index = int(weight_index)
                best_outcome = outcome

        if best_weight_index is None or best_outcome is None:
            raise RuntimeError("Forced commit selection could not find a candidate")

        state.restore(initial_snapshot)
        best_observation = self._observe_weight(
            state=state,
            weight_index=best_weight_index,
            features=features,
            labels=labels,
            base_vector=base_vector,
            forbidden_values=forbidden_values,
        )
        best_outcome = self._collapse_with_backtracking(
            step=step,
            state=state,
            observation=best_observation,
            features=features,
            labels=labels,
            forbidden_values=forbidden_values,
        )
        return best_observation, best_outcome

    def _propagate(
        self,
        state: WeightState,
        collapsed_index: int,
        features: np.ndarray,
        labels: np.ndarray,
        forbidden_values: dict[int, set[int]] | None = None,
    ) -> tuple[int, ...]:
        neighbor_indices = [
            weight_index
            for weight_index in self.neighbor_map[collapsed_index]
            if not state.collapsed[weight_index]
        ]

        if not neighbor_indices:
            return ()

        neighbor_indices.sort(
            key=lambda weight_index: self._entropy(state.probabilities[weight_index]),
            reverse=True,
        )

        budget = self.config.propagation_budget
        if budget > 0:
            neighbor_indices = neighbor_indices[:budget]

        propagated: list[int] = []
        for weight_index in neighbor_indices:
            observation = self._observe_weight(
                state,
                weight_index,
                features,
                labels,
                forbidden_values=forbidden_values,
            )
            blended_distribution = (
                (1.0 - self.config.propagation_blend) * state.probabilities[weight_index]
                + self.config.propagation_blend * observation.posterior
            )
            state.set_distribution(weight_index, blended_distribution)
            propagated.append(weight_index)

        return tuple(propagated)

    def _rollback(
        self,
        state: WeightState,
        history: list[DecisionRecord],
        step_logs: list[CollapseStep],
        progress_snapshots: list[ExperimentSnapshot],
        forbidden_values: dict[int, set[int]],
    ) -> None:
        rollback_depth = min(max(self.config.rollback_depth, 1), len(history))
        rollback_target = history[-rollback_depth]
        state.restore(rollback_target.snapshot_before)
        forbidden_values.setdefault(rollback_target.weight_index, set()).add(rollback_target.chosen_value_index)

        del history[-rollback_depth:]
        del step_logs[-rollback_depth:]
        del progress_snapshots[-rollback_depth:]

    def _allowed_value_indices(
        self,
        weight_index: int,
        forbidden_values: dict[int, set[int]] | None,
    ) -> np.ndarray:
        if forbidden_values is None or weight_index not in forbidden_values:
            return np.arange(self.domain.size, dtype=np.int64)

        forbidden = forbidden_values[weight_index]
        allowed = np.array(
            [domain_index for domain_index in range(self.domain.size) if domain_index not in forbidden],
            dtype=np.int64,
        )
        if allowed.size > 0:
            return allowed

        forbidden.clear()
        return np.arange(self.domain.size, dtype=np.int64)

    def _build_neighbor_map(self) -> dict[int, tuple[int, ...]]:
        node_sets = [set(parameter.nodes) for parameter in self.model.parameter_layout]
        neighbor_map: dict[int, tuple[int, ...]] = {}

        for parameter_index, nodes in enumerate(node_sets):
            neighbors = tuple(
                other_index
                for other_index, other_nodes in enumerate(node_sets)
                if parameter_index != other_index and nodes.intersection(other_nodes)
            )
            neighbor_map[parameter_index] = neighbors

        return neighbor_map

    def _forbidden_stats(self, forbidden_values: dict[int, set[int]]) -> tuple[int, int]:
        active_value_count = sum(len(values) for values in forbidden_values.values() if values)
        active_weight_count = sum(1 for values in forbidden_values.values() if values)
        return active_weight_count, active_value_count

    def _forbidden_entries(
        self,
        forbidden_values: dict[int, set[int]],
    ) -> tuple[ForbiddenWeightState, ...]:
        entries: list[ForbiddenWeightState] = []
        for weight_index, banned_indices in forbidden_values.items():
            if not banned_indices:
                continue

            sorted_indices = tuple(sorted(banned_indices))
            entries.append(
                ForbiddenWeightState(
                    weight_index=weight_index,
                    parameter_name=self.model.parameter_layout[weight_index].name,
                    ban_count=len(sorted_indices),
                    banned_values=tuple(float(self.domain[index]) for index in sorted_indices),
                )
            )

        entries.sort(key=lambda entry: (-entry.ban_count, entry.weight_index))
        return tuple(entries)

    def _forbidden_delta_labels(
        self,
        current_entries: tuple[ForbiddenWeightState, ...],
        previous_entries: tuple[ForbiddenWeightState, ...],
    ) -> tuple[str, ...]:
        current_map = {entry.weight_index: entry for entry in current_entries}
        previous_map = {entry.weight_index: entry for entry in previous_entries}
        changes: list[tuple[int, int, str]] = []

        for weight_index in set(current_map) | set(previous_map):
            current_entry = current_map.get(weight_index)
            previous_entry = previous_map.get(weight_index)
            current_values = current_entry.banned_values if current_entry is not None else ()
            previous_values = previous_entry.banned_values if previous_entry is not None else ()

            if current_values == previous_values:
                continue

            current_count = current_entry.ban_count if current_entry is not None else 0
            previous_count = previous_entry.ban_count if previous_entry is not None else 0
            parameter_name = current_entry.parameter_name if current_entry is not None else previous_entry.parameter_name

            if current_count == 0:
                label = f"{parameter_name} cleared"
            elif previous_count == 0:
                label = f"{parameter_name} +{current_count}"
            elif current_count > previous_count:
                label = f"{parameter_name} +{current_count - previous_count}"
            elif current_count < previous_count:
                label = f"{parameter_name} {current_count - previous_count}"
            else:
                label = f"{parameter_name} changed"

            magnitude = max(abs(current_count - previous_count), 1)
            changes.append((magnitude, weight_index, label))

        changes.sort(key=lambda item: (-item[0], item[1]))
        return tuple(label for _, _, label in changes[:3])

    def _loss_to_distribution(self, losses: np.ndarray) -> np.ndarray:
        finite_mask = np.isfinite(losses)
        if not finite_mask.any():
            return np.full(losses.shape, 1.0 / losses.size, dtype=np.float64)

        temperature = max(self.config.temperature, 1e-6)
        scaled = -(losses[finite_mask] - losses[finite_mask].min()) / temperature
        scaled -= scaled.max()

        finite_weights = np.exp(scaled)
        probabilities = np.zeros(losses.shape, dtype=np.float64)
        probabilities[finite_mask] = finite_weights / finite_weights.sum()
        return probabilities

    def _trial_score(self, shadow_loss: float, hard_loss: float) -> float:
        hard_gap = max(0.0, hard_loss - shadow_loss)
        return (
            shadow_loss
            + self.config.hard_loss_weight * hard_loss
            + self.config.hard_gap_weight * hard_gap
        )

    def _apply_initial_jitter(self, state: WeightState) -> None:
        if self.config.initial_jitter <= 0.0:
            return

        noise = self.rng.normal(
            0.0,
            self.config.initial_jitter,
            size=state.probabilities.shape,
        )
        noise -= noise.mean(axis=1, keepdims=True)

        logits = np.log(np.clip(state.probabilities, 1e-12, 1.0)) + noise
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        state.probabilities[:] = probabilities

    @staticmethod
    def _entropy(probabilities: np.ndarray) -> float:
        safe_probabilities = np.clip(probabilities, 1e-12, 1.0)
        return float(-(safe_probabilities * np.log(safe_probabilities)).sum())
