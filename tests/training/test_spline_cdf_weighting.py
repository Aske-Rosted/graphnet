"""Tests for `PooledSplineCDFWeighter`."""

import os
import sqlite3
from typing import Any, List

import numpy as np
import pandas as pd
import pytest

from graphnet.training.weight_fitting import PooledSplineCDFWeighter


@pytest.fixture
def databases(tmp_path: Any) -> List[str]:
    """Two databases with a steeply falling (power-law) energy spectrum."""
    rng = np.random.default_rng(0)
    paths = []
    for i, n in enumerate([4000, 2000]):
        energy = 10 ** (2 + rng.exponential(scale=0.8, size=n))
        truth = pd.DataFrame(
            {"event_no": np.arange(n) + 10000 * i, "energy": energy}
        )
        path = os.path.join(str(tmp_path), f"db_{i}.db")
        with sqlite3.connect(path) as con:
            truth.to_sql("truth", con, index=False)
        paths.append(path)
    return paths


def test_weights_flatten_the_spectrum(databases: List[str]) -> None:
    """Weighted events are ~uniform in log10(energy) and sum to target."""
    weighter = PooledSplineCDFWeighter(databases)
    weighter.fit("energy", transform="log10")
    out = weighter.apply(target_total_weight=1000.0, deploy=True)

    df = pd.concat(out.values())
    assert np.isclose(df["energy_uniform_weight"].sum(), 1000.0, rtol=1e-6)

    log_e = np.log10(df["energy"])
    low, high = np.percentile(log_e, [5, 80])
    counts, _ = np.histogram(
        log_e,
        bins=np.linspace(low, high, 6),
        weights=df["energy_uniform_weight"],
    )
    assert counts.max() / counts.min() < 1.5

    with sqlite3.connect(databases[0]) as con:
        table = pd.read_sql("select * from energy_uniform_weight", con)
    assert len(table) == 4000
