"""Tests for joined truth tables and SQL-side selections in SQLiteDataset."""

import os
import sqlite3
from typing import Any

import numpy as np
import pandas as pd
import pytest

from graphnet.data.dataset import SQLiteDataset
from graphnet.models.data_representation import KNNGraph
from graphnet.models.detector.prometheus import Prometheus

_FEATURES = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z", "t"]


@pytest.fixture
def database(tmp_path: Any) -> str:
    """Create a database with pulses and truth split over two tables."""
    rng = np.random.default_rng(0)
    n_events = 20
    pulses = pd.DataFrame(
        {
            "event_no": np.repeat(np.arange(n_events), 5),
            **{f: rng.normal(size=5 * n_events) for f in _FEATURES},
        }
    )
    truth = pd.DataFrame(
        {"event_no": np.arange(n_events), "energy": np.arange(n_events) + 1.0}
    )
    extra = pd.DataFrame(
        {
            "event_no": np.arange(n_events),
            "shift": np.arange(n_events) - 10.0,
        }
    )
    path = os.path.join(str(tmp_path), "multi_truth.db")
    with sqlite3.connect(path) as con:
        pulses.to_sql("pulses", con, index=False)
        truth.to_sql("truth", con, index=False)
        extra.to_sql("extra", con, index=False)
    return path


def _dataset(database: str, **kwargs: Any) -> SQLiteDataset:
    return SQLiteDataset(
        path=database,
        pulsemaps="pulses",
        features=_FEATURES,
        truth=["energy", "shift"],
        truth_table=["truth", "extra"],
        data_representation=KNNGraph(
            detector=Prometheus(), input_feature_names=_FEATURES
        ),
        **kwargs,
    )


def test_joined_truth_tables(database: str) -> None:
    """Truth columns are read from both joined tables."""
    dataset = _dataset(database)
    assert len(dataset) == 20
    graph = dataset[3]
    assert float(graph["energy"]) == 4.0
    assert float(graph["shift"]) == -7.0


def test_super_selection_uses_sql(database: str) -> None:
    """Selections are evaluated in SQL, across the joined tables."""
    dataset = _dataset(
        database,
        selection="abs(shift) < 3 and energy > 9",
        use_super_selection=True,
    )
    selected = sorted(int(np.ravel(i)[0]) for i in dataset._indices)
    assert selected == [9, 10, 11, 12]


def test_missing_column_in_joined_tables(database: str) -> None:
    """A column missing from all joined tables is reported once."""
    dataset = _dataset(database)
    missing = dataset._check_missing_columns(
        ["energy", "nope"], ["truth", "extra"]
    )
    assert missing == ["nope"]
