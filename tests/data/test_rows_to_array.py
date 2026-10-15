"""Unit tests for the conversion of query rows in `Dataset`."""

import numpy as np

from graphnet.data.dataset.dataset import _rows_to_array


def test_rows_to_array_matches_asarray() -> None:
    """Float and integer rows give the values of `np.asarray`."""
    rng = np.random.default_rng(0)
    rows = [tuple(r) for r in rng.normal(size=(1000, 5)).tolist()]
    assert np.array_equal(_rows_to_array(rows), np.asarray(rows))
    ints = [(1, 2, 3), (4, 5, 6)]
    assert np.array_equal(_rows_to_array(ints), np.asarray(ints))
    assert _rows_to_array(ints).shape == (2, 3)


def test_rows_to_array_falls_back_for_null_values() -> None:
    """Rows with NULL (None) values are converted as before."""
    rows = [(1.0, None), (2.0, 3.0)]
    converted = _rows_to_array(rows)
    assert converted.dtype == object
    assert converted.tolist() == np.asarray(rows).tolist()


def test_rows_to_array_passes_other_inputs_through() -> None:
    """No rows, or an array, are returned unchanged."""
    assert _rows_to_array([]) == []
    array = np.ones((3, 2))
    assert _rows_to_array(array) is array
