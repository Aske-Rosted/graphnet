"""Unit tests for the conversion of query rows in `Dataset`."""

import numpy as np

from graphnet.data.dataset.dataset import _rows_to_array


def test_rows_to_array_falls_back_for_null_values() -> None:
    """Rows with NULL (None) values are converted as before."""
    rows = [(1.0, None), (2.0, 3.0)]
    converted = _rows_to_array(rows)
    assert converted.dtype == object
    assert converted.tolist() == np.asarray(rows).tolist()
