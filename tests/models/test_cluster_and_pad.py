"""`cluster_and_pad` gives the same clusters as its former implementation."""

from typing import List, Tuple

import numpy as np
import pytest

from graphnet.models.data_representation.graphs.utils import (
    cluster_and_pad,
    lex_sort,
)


def _former(
    x: np.ndarray, cluster_columns: List[int], sort_by: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (clustered_x, counts, padded_x) of the former implementation."""
    x = lex_sort(x=x, cluster_columns=sort_by + cluster_columns)
    unique_sensors, counts = np.unique(
        x[:, cluster_columns], axis=0, return_counts=True
    )
    table = np.concatenate([unique_sensors, counts.reshape(-1, 1)], axis=1)
    table = lex_sort(x=table, cluster_columns=cluster_columns)
    clustered = table[:, 0 : unique_sensors.shape[1]]
    counts = table[:, clustered.shape[1] :].flatten().astype(int)
    padded = np.empty((len(counts), max(counts), x.shape[1]))
    padded.fill(np.nan)
    for i in range(len(counts)):
        padded[i, : counts[i]] = x[: counts[i]]
        x = x[counts[i] :]
    return clustered, counts, padded


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sort_by", [[], [3]])
@pytest.mark.parametrize("seed", range(5))
def test_cluster_and_pad_matches_former_implementation(
    dtype: type, sort_by: List[int], seed: int
) -> None:
    """Same clusters, order, counts and padded pulses, bit for bit."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 400))
    # few sensor positions (many pulses per sensor), shared coordinates,
    # tied times
    positions = rng.integers(-3, 4, size=(int(rng.integers(1, 30)), 3)) * 125.0
    x = np.column_stack(
        [
            positions[rng.integers(0, len(positions), n)],
            rng.integers(0, 50, n) * 2.0,  # time
            rng.exponential(1.0, n),  # charge
        ]
    ).astype(dtype)
    cluster = cluster_and_pad(
        x=x.copy(), cluster_columns=[0, 1, 2], sort_by=sort_by
    )
    clustered, counts, padded = _former(x.copy(), [0, 1, 2], sort_by)
    assert cluster.clustered_x.dtype == clustered.dtype
    assert np.array_equal(cluster.clustered_x, clustered)
    assert np.array_equal(cluster._counts, counts)
    assert cluster._padded_x.dtype == padded.dtype
    assert np.array_equal(cluster._padded_x, padded, equal_nan=True)
