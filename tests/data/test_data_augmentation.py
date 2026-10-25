"""Unit tests for the data augmentations in `graphnet.data.augmentation`."""

import os
from typing import Any, Dict, List, cast

import numpy as np
import pytest
import torch

import graphnet.data.augmentation as augmentation_module
from graphnet.constants import EXAMPLE_DATA_DIR, TEST_DATA_DIR
from graphnet.data.augmentation import (
    RandomRowDropout,
    RandomSensorDropout,
)
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.datamodule import GraphNeTDataModule
from graphnet.data.dataset import Dataset, SQLiteDataset
from graphnet.models.detector import IceCube86
from graphnet.models.detector.prometheus import ORCA150SuperDense
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.utilities.config import DatasetConfig

NAMES = ["x", "y", "z", "time", "charge"]
ICECUBE_DB = os.path.join(
    TEST_DATA_DIR,
    "sqlite/oscNext_genie_level7_v02/oscNext_genie_level7_v02_first_5_frames.db",  # noqa: E501
)
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


def test_row_dropout_rate_range() -> None:
    """Test that a drop-rate range draws a rate per event."""
    _reset_generators()
    event = _event()
    augmentation = RandomRowDropout((0.0, 0.5))
    kept = np.array(
        [len(augmentation(event, NAMES)) / len(event) for _ in range(200)]
    )
    assert abs(kept.mean() - 0.75) < 0.03
    assert kept.min() < 0.6 and kept.max() > 0.9


def test_row_dropout_thinning() -> None:
    """Test that thinning rarely removes rows of high charge."""
    _reset_generators()
    event = _event()
    event[: len(event) // 2, 4] = 1.0
    event[len(event) // 2 :, 4] = 10.0
    out = RandomRowDropout(0.5, mode="thinning", preserve_charge_sum=False)(
        event, NAMES
    )
    n_low = np.sum(out[:, 4] == 1.0)
    n_high = np.sum(out[:, 4] == 10.0)
    assert abs(n_low / (len(event) // 2) - 0.5) < 0.05
    assert n_high == len(event) // 2


def test_apply_probability_and_min_rows() -> None:
    """Test events that are returned unchanged."""
    _reset_generators()
    event = _event()
    assert RandomRowDropout(0.5, apply_probability=0.0)(event, NAMES) is event
    assert RandomRowDropout(1.0, min_rows=1)(event, NAMES) is event
    assert RandomRowDropout(0.5)(event[:0], NAMES).shape == (0, 5)

    augmentation = RandomRowDropout(0.5, apply_probability=0.3)
    unchanged = [augmentation(event, NAMES) is event for _ in range(1000)]
    assert abs(np.mean(unchanged) - 0.7) < 0.05


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


def test_random_streams() -> None:
    """Test reproducibility and independence of the random draws."""
    event = _event()

    def draws(augmentations: List[RandomRowDropout]) -> List[int]:
        return [len(a(event, NAMES)) for a in augmentations for _ in range(5)]

    _reset_generators(3)
    first = draws([RandomRowDropout(0.5)])
    _reset_generators(3)
    assert draws([RandomRowDropout(0.5)]) == first
    _reset_generators(4)
    assert draws([RandomRowDropout(0.5)]) != first

    # Copies of an augmentation share the stream of their process
    _reset_generators(3)
    a, b = RandomRowDropout(0.5), RandomRowDropout(0.5)
    assert len(a(event, NAMES)) != len(b(event, NAMES))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"drop_rate": 1.5},
        {"drop_rate": (0.5, 0.2)},
        {"drop_rate": (0.1, 0.2, 0.3)},
        {"drop_rate": 0.1, "mode": "other"},
        {"drop_rate": 0.1, "apply_probability": 2.0},
        {"drop_rate": 0.1, "charge_feature": None},
        {"drop_rate": 0.1, "min_rows": -1},
    ],
)
def test_invalid_arguments(kwargs: Dict[str, Any]) -> None:
    """Test that invalid arguments raise."""
    with pytest.raises(ValueError):
        RandomRowDropout(**kwargs)


def test_missing_feature() -> None:
    """Test that a missing feature raises."""
    event = _event()
    with pytest.raises(ValueError, match="charge"):
        RandomRowDropout(0.1)(event, ["x", "y", "z", "time", "q"])
    with pytest.raises(ValueError, match="string"):
        RandomSensorDropout(0.1, sensor_features=["string"])(event, NAMES)


def _icecube_dataset_kwargs() -> Dict[str, Any]:
    return {
        "path": ICECUBE_DB,
        "pulsemaps": "SRTInIcePulses",
        "features": FEATURES.ICECUBE86,
        "truth": TRUTH.ICECUBE86,
        "data_representation": KNNGraph(
            detector=IceCube86(),
            node_definition=NodesAsPulses(),
            input_feature_names=FEATURES.ICECUBE86,
        ),
    }


def test_dataset_augments_input_rows() -> None:
    """Test that a `Dataset` augments the rows before the representation."""
    _reset_generators()
    plain = SQLiteDataset(**_icecube_dataset_kwargs())
    augmented = SQLiteDataset(
        **_icecube_dataset_kwargs(),
        augmentations=[RandomRowDropout(0.5, preserve_charge_sum=False)],
    )
    n_plain = sum(plain[i].x.shape[0] for i in range(len(plain)))
    n_augmented = sum(augmented[i].x.shape[0] for i in range(len(augmented)))
    assert 0.35 < n_augmented / n_plain < 0.65

    # Each read is augmented anew
    assert any(
        augmented[0].x.shape[0] != augmented[0].x.shape[0] for _ in range(10)
    )


def test_config_round_trip_and_training_selections(tmp_path: str) -> None:
    """Test the config and that only training selections are augmented."""
    dataset = SQLiteDataset(
        **_icecube_dataset_kwargs(),
        augmentations=[
            RandomRowDropout((0.0, 0.1), apply_probability=0.8),
            RandomSensorDropout(
                0.05, sensor_features=["dom_x", "dom_y", "dom_z"]
            ),
        ],
    )
    path = os.path.join(tmp_path, "dataset.yml")
    dataset.config.dump(path)
    reloaded = Dataset.from_config(path)
    assert isinstance(reloaded, SQLiteDataset)
    assert [type(a) for a in reloaded._augmentations] == [
        RandomRowDropout,
        RandomSensorDropout,
    ]
    assert reloaded._augmentations[0]._drop_rate == (0.0, 0.1)
    assert reloaded._augmentations[0]._apply_probability == 0.8

    config = DatasetConfig.load(path)
    config.selection = {
        "train": "event_no % 2 == 0",
        "validation": "event_no % 2 == 1",
    }
    loaded = Dataset.from_config(config)
    assert isinstance(loaded, dict)
    datasets = cast(Dict[str, Dataset], loaded)
    train, validation = datasets["train"], datasets["validation"]
    assert len(train._augmentations) == 2
    assert validation._augmentations == []


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
