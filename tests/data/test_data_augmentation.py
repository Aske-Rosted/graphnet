"""Unit tests for the data augmentations in `graphnet.data.augmentation`."""

import os
from typing import List

import numpy as np
import torch

import graphnet.data.augmentation as augmentation_module
from graphnet.constants import EXAMPLE_DATA_DIR
from graphnet.data.augmentation import (
    RandomRowDropout,
    RandomSensorDropout,
)
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.datamodule import GraphNeTDataModule
from graphnet.data.dataset import SQLiteDataset
from graphnet.models.detector.prometheus import ORCA150SuperDense
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodesAsPulses

NAMES = ["x", "y", "z", "time", "charge"]
PROMETHEUS_DB = os.path.join(
    EXAMPLE_DATA_DIR, "sqlite/prometheus/prometheus-events.db"
)


def _reset_generators(seed: int = 0) -> None:
    """Start the augmentations' random numbers afresh from torch `seed`."""
    torch.manual_seed(seed)
    augmentation_module._GENERATORS.clear()


def _event(n_sensors: int = 500, rows_per_sensor: int = 4) -> np.ndarray:
    """Return rows of `n_sensors` sensors with `rows_per_sensor` rows each."""
    rng = np.random.default_rng(1)
    position = rng.normal(size=(n_sensors, 3))
    rows = np.repeat(position, rows_per_sensor, axis=0)
    time = rng.uniform(0, 1000, size=(len(rows), 1))
    charge = rng.exponential(1.5, size=(len(rows), 1))
    return np.hstack([rows, time, charge])


def _is_subset(rows: np.ndarray, of: np.ndarray, columns: List[int]) -> bool:
    """Check that every row of `rows` is a row of `of` in `columns`."""
    reference = {tuple(row) for row in of[:, columns]}
    return all(tuple(row) in reference for row in rows[:, columns])


def test_row_dropout_rate_and_charge_sum() -> None:
    """Test the drop rate and the preserved charge sum."""
    _reset_generators()
    event = _event()
    out = RandomRowDropout(0.2)(event, NAMES)
    assert out is not event
    assert abs(len(out) / len(event) - 0.8) < 0.03
    assert np.isclose(out[:, 4].sum(), event[:, 4].sum())
    assert _is_subset(out, event, [0, 1, 2, 3])

    out = RandomRowDropout(0.2, preserve_charge_sum=False)(event, NAMES)
    assert out[:, 4].sum() < event[:, 4].sum()
    assert _is_subset(out, event, [0, 1, 2, 3, 4])


def test_sensor_dropout() -> None:
    """Test that whole sensors are removed."""
    _reset_generators()
    event = _event(n_sensors=500, rows_per_sensor=4)
    out = RandomSensorDropout(0.3, sensor_features=["x", "y", "z"])(
        event, NAMES
    )
    _, counts = np.unique(out[:, :3], axis=0, return_counts=True)
    assert np.all(counts == 4)
    assert abs(len(counts) / 500 - 0.7) < 0.06
    assert np.isclose(out[:, 4].sum(), event[:, 4].sum())
    assert _is_subset(out, event, [0, 1, 2, 3])
    assert (
        RandomSensorDropout(1.0, sensor_features=["x", "y", "z"])(event, NAMES)
        is event
    )


def test_datamodule_augments_training_dataset_only() -> None:
    """Test that `GraphNeTDataModule` augments the training dataset only."""
    dataset_args = {
        "path": PROMETHEUS_DB,
        "pulsemaps": "total",
        "truth_table": "mc_truth",
        "features": FEATURES.PROMETHEUS,
        "truth": TRUTH.PROMETHEUS,
        "data_representation": KNNGraph(
            detector=ORCA150SuperDense(),
            node_definition=NodesAsPulses(),
            input_feature_names=FEATURES.PROMETHEUS,
        ),
        "augmentations": [
            RandomRowDropout(0.1, preserve_charge_sum=False),
        ],
    }
    dm = GraphNeTDataModule(
        dataset_reference=SQLiteDataset,
        dataset_args=dataset_args,
        train_dataloader_kwargs={"batch_size": 2, "num_workers": 1},
        test_selection=[0, 1],
    )
    assert len(dm.train_dataloader.dataset._augmentations) == 1
    assert dm.val_dataloader.dataset._augmentations == []
    assert dm.test_dataloader.dataset._augmentations == []
