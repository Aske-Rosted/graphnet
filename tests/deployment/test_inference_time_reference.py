"""Unit tests for the time references of `I3InferenceModule`."""

import numpy as np
from torch_geometric.data import Batch

from graphnet.deployment.deployment_module import add_time_references
from graphnet.models.data_representation import EdgelessGraph
from graphnet.models.data_representation.graphs.nodes import (
    ClusterSummaryFeatures,
)
from graphnet.models.detector.icecube import IceCube86

_NAMES = ["dom_x", "dom_y", "dom_z", "dom_time", "charge"]


def test_relative_time_is_written_as_absolute_time() -> None:
    """The graph's reference is added to the relative time prediction."""
    graph_definition = EdgelessGraph(
        detector=IceCube86(replace_with_identity=_NAMES),
        node_definition=ClusterSummaryFeatures(
            cluster_on=_NAMES[:3],
            input_feature_names=_NAMES,
            charge_after_t=[],
            time_after_charge_pct=[],
            reference_charge_quantile=0.1,  # as the tref10 models
            reference_time_attribute="reference_time",
        ),
        input_feature_names=_NAMES,
    )
    # 10% of the charge has arrived at 15000 ns (DOM A: 30 of 101 PE)
    pulses = np.array(
        [
            [2.0, 0.0, 0.0, 10000.0, 1.0],
            [0.0, 0.0, 0.0, 15000.0, 30.0],
            [1.0, 0.0, 0.0, 15100.0, 70.0],
        ]
    )
    data = Batch.from_data_list(
        [graph_definition(input_features=pulses, input_feature_names=_NAMES)]
    )
    columns = ["HEPPosX", "HEPTime"]
    predictions = np.array([[12.5, -40.0]], dtype=np.float32)

    absolute, references = add_time_references(
        predictions, data, columns, {"HEPTime": "reference_time"}
    )
    assert references == {"reference_time": 15000.0}
    assert absolute.tolist() == [[12.5, 14960.0]]

    # without pulses the event has no reference
    absolute, references = add_time_references(
        np.full((1, 2), np.nan), None, columns, {"HEPTime": "reference_time"}
    )
    assert np.isnan(references["reference_time"])
    assert np.isnan(absolute).all()
