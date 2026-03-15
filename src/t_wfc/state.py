from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StateSnapshot:
    probabilities: np.ndarray
    collapsed: np.ndarray


@dataclass
class WeightState:
    domain: np.ndarray
    probabilities: np.ndarray
    collapsed: np.ndarray

    @classmethod
    def uniform(cls, parameter_count: int, domain: np.ndarray) -> "WeightState":
        normalized_domain = np.asarray(domain, dtype=np.float64)
        probabilities = np.full(
            (parameter_count, normalized_domain.size),
            1.0 / normalized_domain.size,
            dtype=np.float64,
        )
        collapsed = np.zeros(parameter_count, dtype=bool)
        return cls(domain=normalized_domain, probabilities=probabilities, collapsed=collapsed)

    def expected_vector(self) -> np.ndarray:
        return self.probabilities @ self.domain

    def argmax_vector(self) -> np.ndarray:
        value_indices = self.probabilities.argmax(axis=1)
        return self.domain[value_indices]

    def unresolved_indices(self) -> np.ndarray:
        return np.flatnonzero(~self.collapsed)

    def snapshot(self) -> StateSnapshot:
        return StateSnapshot(self.probabilities.copy(), self.collapsed.copy())

    def restore(self, snapshot: StateSnapshot) -> None:
        self.probabilities[:] = snapshot.probabilities
        self.collapsed[:] = snapshot.collapsed

    def collapse(self, weight_index: int, value_index: int) -> None:
        self.probabilities[weight_index] = 0.0
        self.probabilities[weight_index, value_index] = 1.0
        self.collapsed[weight_index] = True

    def set_distribution(self, weight_index: int, distribution: np.ndarray) -> None:
        if self.collapsed[weight_index]:
            return

        normalized = np.asarray(distribution, dtype=np.float64)
        total = float(normalized.sum())
        if not np.isfinite(total) or total <= 0.0:
            normalized = np.full(self.domain.size, 1.0 / self.domain.size, dtype=np.float64)
        else:
            normalized = normalized / total

        self.probabilities[weight_index] = normalized
