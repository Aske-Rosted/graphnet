"""Unit tests for node definitions."""

from typing import Any, Dict

import numpy as np
import pandas as pd
import sqlite3
import torch
from graphnet.models.graphs.nodes import PercentileClusters
from graphnet.models.data_representation import ClusterSummaryFeatures
from graphnet.constants import EXAMPLE_DATA_DIR


def test_percentile_cluster() -> None:
    """Test that percentiles outputted by PercentileCluster.

    Here we check that it matches percentiles obtained from
    "traditional" ways.
    """
    # definitions
    percentiles = [0, 10, 50, 90, 100]
    database = f"{EXAMPLE_DATA_DIR}/sqlite/prometheus/prometheus-events.db"
    #  Grab first event in database
    with sqlite3.connect(database) as con:
        query = "select event_no from mc_truth limit 1"
        event_no = pd.read_sql(query, con)
        query = f'select sensor_pos_x, sensor_pos_y, sensor_pos_z, t from total where event_no = {str(event_no["event_no"][0])}'  # noqa: E501
        df = pd.read_sql(query, con)

    # Save original feature names, create variables.
    original_features = list(df.columns)
    x = np.array(df)
    tensor = torch.tensor(x)

    # Construct node definition
    # This defines each DOM as a cluster, and will summarize pulses seen by
    # DOMs using percentiles.
    node_definition = PercentileClusters(
        cluster_on=["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"],
        percentiles=percentiles,
        input_feature_names=original_features,
    )

    # Apply node definition to torch tensor with raw pulses
    graph = node_definition(tensor)
    new_features = node_definition._output_feature_names
    x_tilde = graph.numpy()

    # Calculate percentiles "the normal way" and compare that output of
    # node definition match.

    unique_doms = (
        df.groupby(["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"])
        .size()
        .reset_index()
    )
    for i in range(len(unique_doms)):
        idx_original = (
            (df["sensor_pos_x"] == unique_doms["sensor_pos_x"][i])
            & ((df["sensor_pos_y"] == unique_doms["sensor_pos_y"][i]))
            & (df["sensor_pos_z"] == unique_doms["sensor_pos_z"][i])
        )
        idx_tilde = (
            (
                x_tilde[:, new_features.index("sensor_pos_x")]
                == unique_doms["sensor_pos_x"][i]
            )
            & (
                x_tilde[:, new_features.index("sensor_pos_y")]
                == unique_doms["sensor_pos_y"][i]
            )
            & (
                x_tilde[:, new_features.index("sensor_pos_z")]
                == unique_doms["sensor_pos_z"][i]
            )
        )
        for percentile in percentiles:
            pct_idx = new_features.index(f"t_pct{percentile}")
            try:
                assert np.isclose(
                    x_tilde[idx_tilde, pct_idx],
                    np.percentile(df.loc[idx_original, "t"], percentile),
                )
            except AssertionError as e:
                print(f"Percentile {percentile} does not match.")
                raise e


_NAMES = ["dom_x", "dom_y", "dom_z", "dom_time", "charge"]
# Two DOMs: A with pulses at 1000/1002/1004 ns (1, 2, 3 PE),
# B with pulses at 1000/1050/1200 ns (1 PE each).
_PULSES = torch.tensor(
    [
        [0.0, 0.0, 0.0, 1000.0, 1.0],
        [0.0, 0.0, 0.0, 1002.0, 2.0],
        [0.0, 0.0, 0.0, 1004.0, 3.0],
        [1.0, 0.0, 0.0, 1000.0, 1.0],
        [1.0, 0.0, 0.0, 1050.0, 1.0],
        [1.0, 0.0, 0.0, 1200.0, 1.0],
    ],
    dtype=torch.float64,
)


def _summary(**kwargs: Any) -> ClusterSummaryFeatures:
    return ClusterSummaryFeatures(
        cluster_on=_NAMES[:3],
        input_feature_names=_NAMES,
        charge_after_t=[],
        time_after_charge_pct=[],
        **kwargs,
    )


def test_cluster_summary_total_charge_fraction() -> None:
    """The fraction column is log10(cluster charge / event charge)."""
    node_definition = _summary(total_charge_fraction=True)
    names = node_definition._output_feature_names
    nodes = node_definition(_PULSES).numpy()

    fraction = nodes[:, names.index("total_charge_fraction")]
    assert np.allclose(10**fraction, [6 / 9, 3 / 9])
    assert np.allclose(nodes[:, names.index("total_charge")], np.log10([6, 3]))


def test_cluster_summary_charge_weighted_changes_time_std() -> None:
    """Charge weighting changes the time std of unevenly charged DOMs."""
    plain = _summary()
    weighted = _summary(charge_weighted=True)
    idx = plain._output_feature_names.index("time_std")
    assert not np.allclose(
        plain(_PULSES).numpy()[:, idx], weighted(_PULSES).numpy()[:, idx]
    )


def test_cluster_summary_is_invariant_to_event_time_offset() -> None:
    """Features do not depend on when the event happened."""
    node_definition = _summary(
        total_charge_fraction=True, charge_weighted=True
    )
    shifted = _PULSES.clone()
    shifted[:, 3] += 5e4
    assert np.allclose(
        node_definition(_PULSES).numpy(), node_definition(shifted).numpy()
    )


def test_cluster_summary_empty_event() -> None:
    """An event without pulses gives an empty node set."""
    node_definition = _summary(total_charge_fraction=True)
    nodes = node_definition(torch.zeros((0, 5), dtype=torch.float64))
    assert nodes.shape == (0, len(node_definition._output_feature_names))


def test_cluster_summary_charge_after_t() -> None:
    """Charge within t of the first pulse, including full-window DOMs."""
    names = ["dom_x", "dom_y", "dom_z", "dom_time", "charge"]
    # DOM A: pulses at 0/2/4 ns (1, 2, 3 PE); DOM B: 0/50/200 ns (1 PE each)
    pulses = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 4.0, 3.0],
            [1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 50.0, 1.0],
            [1.0, 0.0, 0.0, 200.0, 1.0],
        ],
        dtype=torch.float64,
    )
    node_definition = ClusterSummaryFeatures(
        cluster_on=names[:3],
        input_feature_names=names,
        charge_after_t=[10, 100, 500],
        time_after_charge_pct=[],
        charge_standardization=1.0,
    )
    feature_names = node_definition._output_feature_names
    nodes = node_definition(pulses).numpy()
    columns = [
        feature_names.index(f"charge_after_{t}ns") for t in (10, 100, 500)
    ]
    assert np.allclose(nodes[:, columns], [[6, 6, 6], [1, 2, 3]])


def test_cluster_summary_charge_percentile_times() -> None:
    """Charge-percentile times do not depend on `total_charge_fraction`."""
    kwargs: Dict[str, Any] = dict(
        cluster_on=_NAMES[:3],
        input_feature_names=_NAMES,
        charge_after_t=[],
        time_after_charge_pct=[1, 50, 100],
        time_standardization=1.0,
    )
    plain = ClusterSummaryFeatures(**kwargs)
    with_fraction = ClusterSummaryFeatures(
        total_charge_fraction=True, **kwargs
    )
    names = plain._output_feature_names
    columns = [names.index(f"time_after_charge_pct{p}") for p in (1, 50, 100)]
    first = plain(_PULSES).numpy()[:, names.index("time_of_first_hit")]
    times = plain(_PULSES).numpy()[:, columns] - first[:, None]

    # A: 1/2/3 PE at 0/2/4 ns -> 50% reached at 2 ns (3 of 6 PE);
    # B: 1 PE each at 0/50/200 ns -> 50% reached at 50 ns.
    assert np.allclose(times, [[0, 2, 4], [0, 50, 200]])
    fraction_names = with_fraction._output_feature_names
    fraction_columns = [
        fraction_names.index(f"time_after_charge_pct{p}") for p in (1, 50, 100)
    ]
    assert np.allclose(
        with_fraction(_PULSES).numpy()[:, fraction_columns],
        plain(_PULSES).numpy()[:, columns],
    )


def test_cluster_summary_charge_weighted_time_std() -> None:
    """Charge-weighted time std is the weighted spread around the mean."""
    node_definition = ClusterSummaryFeatures(
        cluster_on=_NAMES[:3],
        input_feature_names=_NAMES,
        charge_after_t=[],
        time_after_charge_pct=[],
        time_standardization=1.0,
        charge_weighted=True,
    )
    std = node_definition(_PULSES).numpy()[
        :, node_definition._output_feature_names.index("time_std")
    ]
    t, w = np.array([0.0, 2.0, 4.0]), np.array([1.0, 2.0, 3.0])
    mean = np.sum(w * t) / w.sum()
    expected_a = np.sqrt(np.sum(w * (t - mean) ** 2) / w.sum())
    assert np.allclose(std, [expected_a, np.std([0.0, 50.0, 200.0])])


def test_cluster_summary_reference_time_is_charge_weighted() -> None:
    """Times are relative to the charge-weighted median of the event."""
    # One bright DOM (100 PE at 0 ns) and two faint ones (1 PE at 100/200 ns)
    pulses = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 100.0],
            [1.0, 0.0, 0.0, 100.0, 1.0],
            [2.0, 0.0, 0.0, 200.0, 1.0],
        ],
        dtype=torch.float64,
    )
    node_definition = ClusterSummaryFeatures(
        cluster_on=_NAMES[:3],
        input_feature_names=_NAMES,
        charge_after_t=[],
        time_after_charge_pct=[],
        time_standardization=1.0,
    )
    first_hit = node_definition(pulses).numpy()[
        :, node_definition._output_feature_names.index("time_of_first_hit")
    ]
    assert np.allclose(first_hit, [0.0, 100.0, 200.0])
