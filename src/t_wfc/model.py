from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MLPConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int


@dataclass(frozen=True)
class ParameterMeta:
    index: int
    name: str
    nodes: tuple[tuple[str, int], ...]


class ToyMLP:
    def __init__(self, config: MLPConfig) -> None:
        self.config = config
        self.parameter_layout = self._build_parameter_layout()

    @property
    def parameter_count(self) -> int:
        return len(self.parameter_layout)

    def unpack(self, flat_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        expected_size = self.parameter_count
        if flat_weights.shape != (expected_size,):
            raise ValueError(f"flat_weights must have shape ({expected_size},)")

        cursor = 0
        w1_size = self.config.input_dim * self.config.hidden_dim
        w1 = flat_weights[cursor : cursor + w1_size].reshape(self.config.input_dim, self.config.hidden_dim)
        cursor += w1_size

        b1 = flat_weights[cursor : cursor + self.config.hidden_dim]
        cursor += self.config.hidden_dim

        w2_size = self.config.hidden_dim * self.config.output_dim
        w2 = flat_weights[cursor : cursor + w2_size].reshape(self.config.hidden_dim, self.config.output_dim)
        cursor += w2_size

        b2 = flat_weights[cursor : cursor + self.config.output_dim]
        return w1, b1, w2, b2

    def forward(self, flat_weights: np.ndarray, features: np.ndarray) -> np.ndarray:
        w1, b1, w2, b2 = self.unpack(flat_weights)
        hidden = np.tanh(features @ w1 + b1)
        return hidden @ w2 + b2

    def predict(self, flat_weights: np.ndarray, features: np.ndarray) -> np.ndarray:
        logits = self.forward(flat_weights, features)
        return logits.argmax(axis=1)

    def accuracy(self, flat_weights: np.ndarray, features: np.ndarray, labels: np.ndarray) -> float:
        predictions = self.predict(flat_weights, features)
        return float((predictions == labels).mean())

    def loss(self, flat_weights: np.ndarray, features: np.ndarray, labels: np.ndarray) -> float:
        logits = self.forward(flat_weights, features)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        log_probs = shifted - np.log(exp_logits.sum(axis=1, keepdims=True))
        return float(-log_probs[np.arange(labels.shape[0]), labels].mean())

    def _build_parameter_layout(self) -> list[ParameterMeta]:
        layout: list[ParameterMeta] = []
        cursor = 0

        for input_index in range(self.config.input_dim):
            for hidden_index in range(self.config.hidden_dim):
                layout.append(
                    ParameterMeta(
                        index=cursor,
                        name=f"w1[{input_index},{hidden_index}]",
                        nodes=(("input", input_index), ("hidden", hidden_index)),
                    )
                )
                cursor += 1

        for hidden_index in range(self.config.hidden_dim):
            layout.append(
                ParameterMeta(
                    index=cursor,
                    name=f"b1[{hidden_index}]",
                    nodes=(("hidden", hidden_index),),
                )
            )
            cursor += 1

        for hidden_index in range(self.config.hidden_dim):
            for output_index in range(self.config.output_dim):
                layout.append(
                    ParameterMeta(
                        index=cursor,
                        name=f"w2[{hidden_index},{output_index}]",
                        nodes=(("hidden", hidden_index), ("output", output_index)),
                    )
                )
                cursor += 1

        for output_index in range(self.config.output_dim):
            layout.append(
                ParameterMeta(
                    index=cursor,
                    name=f"b2[{output_index}]",
                    nodes=(("output", output_index),),
                )
            )
            cursor += 1

        return layout
