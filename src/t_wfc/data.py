from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

IRIS_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "iris.csv"


@dataclass(frozen=True)
class DatasetSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def load_dataset(name: str, **kwargs: float | int | str | Path) -> DatasetSplit:
    if name == "make_moons":
        return make_moons_dataset(
            n_samples=int(kwargs.get("n_samples", 120)),
            noise=float(kwargs.get("noise", 0.08)),
            test_ratio=float(kwargs.get("test_ratio", 0.25)),
            seed=int(kwargs.get("seed", 7)),
        )
    if name == "iris":
        csv_path = kwargs.get("csv_path")
        return load_iris_dataset(
            test_ratio=float(kwargs.get("test_ratio", 0.25)),
            seed=int(kwargs.get("seed", 7)),
            csv_path=None if csv_path is None else Path(str(csv_path)),
        )
    if name == "spiral":
        return make_spiral_dataset(
            n_samples=int(kwargs.get("n_samples", 600)),
            noise=float(kwargs.get("noise", 0.16)),
            test_ratio=float(kwargs.get("test_ratio", 0.25)),
            seed=int(kwargs.get("seed", 7)),
            classes=int(kwargs.get("classes", 3)),
            turns=float(kwargs.get("turns", 1.75)),
        )
    raise ValueError(f"Unsupported dataset: {name}")


def make_moons_dataset(
    n_samples: int = 120,
    noise: float = 0.08,
    test_ratio: float = 0.25,
    seed: int = 7,
) -> DatasetSplit:
    if n_samples < 8:
        raise ValueError("n_samples must be at least 8")
    if not 0.0 < test_ratio < 0.5:
        raise ValueError("test_ratio must be between 0.0 and 0.5")

    rng = np.random.default_rng(seed)
    outer_count = n_samples // 2
    inner_count = n_samples - outer_count

    outer_theta = rng.uniform(0.0, np.pi, size=outer_count)
    inner_theta = rng.uniform(0.0, np.pi, size=inner_count)

    outer_arc = np.column_stack((np.cos(outer_theta), np.sin(outer_theta)))
    inner_arc = np.column_stack((1.0 - np.cos(inner_theta), 0.5 - np.sin(inner_theta)))

    features = np.vstack((outer_arc, inner_arc))
    labels = np.concatenate(
        (
            np.zeros(outer_count, dtype=np.int64),
            np.ones(inner_count, dtype=np.int64),
        )
    )

    features += rng.normal(0.0, noise, size=features.shape)

    permutation = rng.permutation(n_samples)
    features = features[permutation]
    labels = labels[permutation]

    test_count = max(1, int(round(n_samples * test_ratio)))
    test_count = min(test_count, n_samples - 1)
    train_x = features[:-test_count]
    train_y = labels[:-test_count]
    test_x = features[-test_count:]
    test_y = labels[-test_count:]

    return _standardize_split(train_x, train_y, test_x, test_y)


def load_iris_dataset(
    test_ratio: float = 0.25,
    seed: int = 7,
    csv_path: Path | None = None,
) -> DatasetSplit:
    if not 0.0 < test_ratio < 0.5:
        raise ValueError("test_ratio must be between 0.0 and 0.5")

    path = IRIS_CSV_PATH if csv_path is None else csv_path
    if not path.exists():
        raise FileNotFoundError(f"Iris dataset not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")

    if len(header) < 5:
        raise ValueError(f"Unexpected iris.csv header: {header!r}")

    row_count = int(header[0])
    feature_count = int(header[1])
    raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if raw.shape != (row_count, feature_count + 1):
        raise ValueError(f"Unexpected iris.csv shape: {raw.shape}")

    features = raw[:, :feature_count]
    labels = raw[:, feature_count].astype(np.int64)

    rng = np.random.default_rng(seed)
    train_indices, test_indices = _stratified_split_indices(labels, test_ratio, rng)
    train_x = features[train_indices]
    train_y = labels[train_indices]
    test_x = features[test_indices]
    test_y = labels[test_indices]

    return _standardize_split(train_x, train_y, test_x, test_y)


def make_spiral_dataset(
    n_samples: int = 600,
    noise: float = 0.16,
    test_ratio: float = 0.25,
    seed: int = 7,
    classes: int = 3,
    turns: float = 1.75,
) -> DatasetSplit:
    if n_samples < classes * 6:
        raise ValueError("n_samples must be at least classes * 6")
    if classes < 2:
        raise ValueError("classes must be at least 2")
    if turns <= 0.0:
        raise ValueError("turns must be positive")
    if not 0.0 < test_ratio < 0.5:
        raise ValueError("test_ratio must be between 0.0 and 0.5")

    rng = np.random.default_rng(seed)
    class_counts = np.full(classes, n_samples // classes, dtype=np.int64)
    class_counts[: n_samples % classes] += 1

    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    full_rotation = turns * 2.0 * np.pi

    for class_index, class_count in enumerate(class_counts.tolist()):
        base_radius = np.linspace(0.08, 1.0, class_count, dtype=np.float64)
        base_angle = np.linspace(0.0, full_rotation, class_count, dtype=np.float64)
        angle_offset = (2.0 * np.pi * class_index) / classes

        radius = np.clip(base_radius + rng.normal(0.0, noise * 0.08, size=class_count), 0.02, None)
        theta = base_angle + angle_offset + rng.normal(0.0, noise, size=class_count)

        x_values = radius * np.cos(theta)
        y_values = radius * np.sin(theta)
        feature_blocks.append(np.column_stack((x_values, y_values)))
        label_blocks.append(np.full(class_count, class_index, dtype=np.int64))

    features = np.vstack(feature_blocks)
    labels = np.concatenate(label_blocks)

    train_indices, test_indices = _stratified_split_indices(labels, test_ratio, rng)
    train_x = features[train_indices]
    train_y = labels[train_indices]
    test_x = features[test_indices]
    test_y = labels[test_indices]

    return _standardize_split(train_x, train_y, test_x, test_y)


def _stratified_split_indices(
    labels: np.ndarray,
    test_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    train_blocks: list[np.ndarray] = []
    test_blocks: list[np.ndarray] = []

    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        shuffled = rng.permutation(class_indices)
        test_count = max(1, int(round(class_indices.size * test_ratio)))
        test_count = min(test_count, class_indices.size - 1)
        test_blocks.append(shuffled[:test_count])
        train_blocks.append(shuffled[test_count:])

    train_indices = rng.permutation(np.concatenate(train_blocks))
    test_indices = rng.permutation(np.concatenate(test_blocks))
    return train_indices, test_indices


def _standardize_split(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> DatasetSplit:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0

    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    return DatasetSplit(
        x_train=train_x.astype(np.float64),
        y_train=train_y,
        x_test=test_x.astype(np.float64),
        y_test=test_y,
    )
