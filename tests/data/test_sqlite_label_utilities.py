"""Tests for the SQLite label-preparation utilities."""

import os
import sqlite3
from typing import Any

import pandas as pd

from graphnet.data.utilities.sqlite_utilities import (
    add_first_pulse_time_to_truth,
    add_starting,
    drop_column,
    query_database,
)


def _database(tmp_path: Any) -> str:
    path = os.path.join(str(tmp_path), "labels.db")
    with sqlite3.connect(path) as con:
        con.execute(
            "CREATE TABLE truth (event_no INTEGER PRIMARY KEY NOT NULL, "
            "containment_type INTEGER, energy FLOAT)"
        )
        con.executemany(
            "INSERT INTO truth VALUES (?, ?, ?)",
            [(0, 2, 1.0), (1, 3, 2.0), (2, 5, 3.0), (3, None, 4.0)],
        )
        pd.DataFrame(
            {
                "event_no": [0, 0, 1, 1, 1, 2, 3],
                "dom_time": [5.0, 3.0, 9.0, 7.0, 8.0, 1.0, 4.0],
            }
        ).to_sql("SRTInIcePulses", con, index=False)
    return path


def test_add_first_pulse_time_and_starting(tmp_path: Any) -> None:
    """Labels are added to the truth table and can be recomputed."""
    path = _database(tmp_path)
    add_first_pulse_time_to_truth(path)
    add_starting(path)
    add_starting(path, force=True)

    truth = query_database(
        path, "SELECT * FROM truth ORDER BY event_no"
    ).set_index("event_no")
    assert truth["first_pulse_time"].tolist() == [3.0, 7.0, 1.0, 4.0]
    assert truth["starting"].tolist()[:3] == [0, 1, 1]
    assert pd.isna(truth["starting"].iloc[3])


def test_drop_column(tmp_path: Any) -> None:
    """A column is removed while the other data is kept."""
    path = _database(tmp_path)
    drop_column(path, "truth", "energy")
    truth = query_database(path, "SELECT * FROM truth ORDER BY event_no")
    assert list(truth.columns) == ["event_no", "containment_type"]
    assert truth["event_no"].tolist() == [0, 1, 2, 3]
