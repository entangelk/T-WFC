from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MLPConfig:
    input_dim: int
    output_dim: int
    hidden_dim: int = 0
    hidden_layers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")

        resolved_hidden_layers = self.resolved_hidden_layers
        if not resolved_hidden_layers:
            raise ValueError("At least one hidden layer is required")
        if any(width <= 0 for width in resolved_hidden_layers):
            raise ValueError("All hidden layer widths must be positive")

    @property
    def resolved_hidden_layers(self) -> tuple[int, ...]:
        if self.hidden_layers:
            return tuple(int(width) for width in self.hidden_layers)
        if self.hidden_dim > 0:
            return (int(self.hidden_dim),)
        return ()

    @property
    def layer_dims(self) -> tuple[int, ...]:
        return (self.input_dim, *self.resolved_hidden_layers, self.output_dim)

    @property
    def architecture_label(self) -> str:
        return "-".join(str(width) for width in self.layer_dims)


@dataclass(frozen=True)
class ParameterMeta:
    index: int
    name: str
    nodes: tuple[tuple[str, int], ...]


class ToyMLP:
    def __init__(self, config: MLPConfig) -> None:
        self.config = config
        self.layer_dims = config.layer_dims
        self.layer_shapes = tuple(zip(self.layer_dims[:-1], self.layer_dims[1:]))
        self.parameter_layout = self._build_parameter_layout()

    @property
    def parameter_count(self) -> int:
        return len(self.parameter_layout)

    @property
    def architecture_label(self) -> str:
        return self.config.architecture_label

    def unpack(self, flat_weights: np.ndarray) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        expected_size = self.parameter_count
        if flat_weights.shape != (expected_size,):
            raise ValueError(f"flat_weights must have shape ({expected_size},)")

        cursor = 0
        weights: list[np.ndarray] = []
        biases: list[np.ndarray] = []
        for fan_in, fan_out in self.layer_shapes:
            weight_size = fan_in * fan_out
            weights.append(flat_weights[cursor : cursor + weight_size].reshape(fan_in, fan_out))
            cursor += weight_size

            biases.append(flat_weights[cursor : cursor + fan_out])
            cursor += fan_out

        return tuple(weights), tuple(biases)

    def pack(self, weights: Iterable[np.ndarray], biases: Iterable[np.ndarray]) -> np.ndarray:
        flat_parts: list[np.ndarray] = []
        for weight, bias in zip(weights, biases):
            flat_parts.append(np.asarray(weight, dtype=np.float64).reshape(-1))
            flat_parts.append(np.asarray(bias, dtype=np.float64).reshape(-1))
        if not flat_parts:
            raise ValueError("At least one layer is required to pack parameters")
        return np.concatenate(flat_parts).astype(np.float64, copy=False)

    def forward(self, flat_weights: np.ndarray, features: np.ndarray) -> np.ndarray:
        logits, _, _, _, _ = self.forward_with_cache(flat_weights, features)
        return logits

    def forward_with_cache(
        self,
        flat_weights: np.ndarray,
        features: np.ndarray,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        weights, biases = self.unpack(flat_weights)
        activations: list[np.ndarray] = [features]
        pre_activations: list[np.ndarray] = []

        current = features
        for layer_index, (weight, bias) in enumerate(zip(weights, biases)):
            pre_activation = current @ weight + bias
            pre_activations.append(pre_activation)
            if layer_index == len(weights) - 1:
                current = pre_activation
            else:
                current = np.tanh(pre_activation)
            activations.append(current)

        return current, tuple(activations), tuple(pre_activations), weights, biases

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

    def loss_and_gradient(self, flat_weights: np.ndarray, features: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
        logits, activations, pre_activations, weights, biases = self.forward_with_cache(flat_weights, features)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        log_probs = shifted - np.log(exp_logits.sum(axis=1, keepdims=True))
        loss = float(-log_probs[np.arange(labels.shape[0]), labels].mean())

        gradient_logits = probabilities
        gradient_logits[np.arange(labels.shape[0]), labels] -= 1.0
        gradient_logits /= labels.shape[0]

        weight_gradients: list[np.ndarray] = [np.empty_like(weight) for weight in weights]
        bias_gradients: list[np.ndarray] = [np.empty(bias.shape, dtype=np.float64) for bias in biases]
        backprop_signal = gradient_logits

        for layer_index in range(len(weights) - 1, -1, -1):
            weight_gradients[layer_index] = activations[layer_index].T @ backprop_signal
            bias_gradients[layer_index] = backprop_signal.sum(axis=0)

            if layer_index == 0:
                continue

            hidden_signal = backprop_signal @ weights[layer_index].T
            hidden_activation = activations[layer_index]
            backprop_signal = hidden_signal * (1.0 - hidden_activation**2)

        return loss, self.pack(weight_gradients, bias_gradients)

    def random_vector(
        self,
        rng: np.random.Generator,
        *,
        scale: float = 1.0,
    ) -> np.ndarray:
        weights: list[np.ndarray] = []
        biases: list[np.ndarray] = []

        for fan_in, fan_out in self.layer_shapes:
            layer_scale = scale / np.sqrt(max(fan_in, 1))
            weights.append(rng.normal(0.0, layer_scale, size=(fan_in, fan_out)))
            biases.append(np.zeros(fan_out, dtype=np.float64))

        return self.pack(weights, biases)

    def _build_parameter_layout(self) -> list[ParameterMeta]:
        layout: list[ParameterMeta] = []
        cursor = 0
        node_names = self._layer_node_names()

        for layer_index, (fan_in, fan_out) in enumerate(self.layer_shapes, start=1):
            source_name = node_names[layer_index - 1]
            target_name = node_names[layer_index]

            for source_index in range(fan_in):
                for target_index in range(fan_out):
                    layout.append(
                        ParameterMeta(
                            index=cursor,
                            name=f"w{layer_index}[{source_index},{target_index}]",
                            nodes=((source_name, source_index), (target_name, target_index)),
                        )
                    )
                    cursor += 1

            for target_index in range(fan_out):
                layout.append(
                    ParameterMeta(
                        index=cursor,
                        name=f"b{layer_index}[{target_index}]",
                        nodes=((target_name, target_index),),
                    )
                )
                cursor += 1

        return layout

    def _layer_node_names(self) -> tuple[str, ...]:
        hidden_names = tuple(f"hidden{layer_index}" for layer_index in range(1, len(self.config.resolved_hidden_layers) + 1))
        return ("input", *hidden_names, "output")
