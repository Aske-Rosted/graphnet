"""Tests that predictions and additional attributes stay aligned."""

from typing import List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler
from torch_geometric.data import Batch, Data

from graphnet.data.constants import FEATURES
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.gnn.gnn import GNN
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.models.task.reconstruction import EnergyReconstruction
from graphnet.training.loss_functions import MSELoss


class _MeanBackbone(GNN):
    """Event-level backbone: per-event mean of the node features."""

    def __init__(self) -> None:
        super().__init__(nb_inputs=2, nb_outputs=2)

    def forward(self, data: Data) -> Tensor:
        counts = torch.bincount(data.batch, minlength=data.num_graphs)
        sums = torch.zeros(data.num_graphs, 2).index_add_(
            0, data.batch, data.x
        )
        return sums / counts[:, None]


def _model() -> StandardModel:
    torch.manual_seed(0)
    return StandardModel(
        data_representation=KNNGraph(
            detector=IceCube86(),
            node_definition=NodesAsPulses(),
            input_feature_names=FEATURES.DEEPCORE,
        ),
        backbone=_MeanBackbone(),
        tasks=EnergyReconstruction(
            hidden_size=2, target_labels="energy", loss_function=MSELoss()
        ),
    )


def _events(n: int = 23) -> List[Data]:
    rng = np.random.default_rng(0)
    return [
        Data(
            x=torch.tensor(rng.normal(size=(int(k), 2)), dtype=torch.float),
            event_no=torch.tensor([i]),
            n_pulses=torch.tensor([int(k)]),
        )
        for i, k in enumerate(rng.integers(1, 6, size=n))
    ]


def _split_and_drop(graphs: List[Data]) -> List[Batch]:
    """Drop single-pulse events and return two sub-batches."""
    graphs = [g for g in graphs if int(g.n_pulses) > 1]
    half = len(graphs) // 2
    parts = [graphs[:half], graphs[half:]]
    return [Batch.from_data_list(p) for p in parts if len(p) > 0]


def test_predict_as_dataframe_keeps_attributes_aligned() -> None:
    """Attributes match predictions despite shuffling and dropped events."""
    model = _model()
    events = _events()
    loader = DataLoader(
        events,  # type: ignore[arg-type]
        batch_size=5,
        sampler=RandomSampler(events),  # type: ignore[arg-type]
        collate_fn=_split_and_drop,
    )
    results = model.predict_as_dataframe(
        loader, additional_attributes=["event_no"]
    )

    expected_events = sorted(
        int(e.event_no) for e in events if int(e.n_pulses) > 1
    )
    assert sorted(results["event_no"].tolist()) == expected_events

    model.inference()
    with torch.no_grad():
        for _, row in results.iterrows():
            event = events[int(row["event_no"])]
            single = model(Batch.from_data_list([event]))[0]
            assert np.isclose(float(single), row[model.prediction_labels[0]])


def test_predict_returns_one_tensor_per_task() -> None:
    """`predict` keeps returning one tensor per task."""
    model = _model()
    loader = DataLoader(
        _events(),  # type: ignore[arg-type]
        batch_size=4,
        collate_fn=Batch.from_data_list,
    )
    predictions = model.predict(loader)
    assert len(predictions) == 1
    assert predictions[0].shape == (23, 1)
