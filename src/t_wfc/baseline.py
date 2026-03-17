from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import DatasetSplit
from .model import ToyMLP
from .trainer import EvaluationMetrics


@dataclass(frozen=True)
class SGDBaselineConfig:
    epochs: int = 240
    learning_rate: float = 0.08
    learning_rate_decay: float = 0.01
    momentum: float = 0.9
    batch_size: int = 32
    weight_scale: float = 1.0
    seed: int = 7


@dataclass(frozen=True)
class SGDBaselineResult:
    initial_metrics: EvaluationMetrics
    final_metrics: EvaluationMetrics
    history: tuple[EvaluationMetrics, ...]
    weights: np.ndarray


def train_sgd_classifier(
    model: ToyMLP,
    dataset: DatasetSplit,
    config: SGDBaselineConfig,
) -> SGDBaselineResult:
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.learning_rate_decay < 0.0:
        raise ValueError("learning_rate_decay must be non-negative")
    if not 0.0 <= config.momentum < 1.0:
        raise ValueError("momentum must be in [0.0, 1.0)")
    if config.batch_size == 0:
        raise ValueError("batch_size must be positive or -1 for full-batch mode")

    rng = np.random.default_rng(config.seed)
    weights = model.random_vector(rng, scale=config.weight_scale)
    velocity = np.zeros_like(weights)
    history: list[EvaluationMetrics] = [_evaluate(model, weights, dataset)]

    train_size = dataset.x_train.shape[0]
    batch_size = train_size if config.batch_size < 0 else min(config.batch_size, train_size)

    for epoch in range(config.epochs):
        learning_rate = config.learning_rate / (1.0 + config.learning_rate_decay * epoch)
        permutation = rng.permutation(train_size)

        for start in range(0, train_size, batch_size):
            batch_indices = permutation[start : start + batch_size]
            _, gradient = model.loss_and_gradient(
                weights,
                dataset.x_train[batch_indices],
                dataset.y_train[batch_indices],
            )
            velocity = config.momentum * velocity - learning_rate * gradient
            weights = weights + velocity

        history.append(_evaluate(model, weights, dataset))

    return SGDBaselineResult(
        initial_metrics=history[0],
        final_metrics=history[-1],
        history=tuple(history),
        weights=weights,
    )


def _evaluate(model: ToyMLP, weights: np.ndarray, dataset: DatasetSplit) -> EvaluationMetrics:
    return EvaluationMetrics(
        train_loss=model.loss(weights, dataset.x_train, dataset.y_train),
        test_loss=model.loss(weights, dataset.x_test, dataset.y_test),
        train_accuracy=model.accuracy(weights, dataset.x_train, dataset.y_train),
        test_accuracy=model.accuracy(weights, dataset.x_test, dataset.y_test),
    )
