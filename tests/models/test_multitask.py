"""Unit tests for multitask support in `StandardModel` and its tasks."""

import pickle
from typing import Any, Dict, List

import pytest
import torch
from torch import Tensor
from torch_geometric.data import Data

from graphnet.data.constants import FEATURES
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.gnn.gnn import GNN
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.models.task.multitask_utils import LossWeightBalancing
from graphnet.models.task.reconstruction import (
    DirectionReconstruction,
    DirectionReconstructionWithKappa,
    EnergyReconstruction,
)
from graphnet.training.loss_functions import MSELoss


class _ConstantBackbone(GNN):
    """Backbone returning a learnable vector per event."""

    def __init__(self, nb_outputs: int):
        super().__init__(nb_inputs=1, nb_outputs=nb_outputs)
        self.weight = torch.nn.Parameter(torch.randn(nb_outputs))

    def forward(self, data: Data) -> Tensor:
        return self.weight.expand(data.num_graphs, -1)


def _graph_definition() -> KNNGraph:
    return KNNGraph(
        detector=IceCube86(),
        node_definition=NodesAsPulses(),
        nb_nearest_neighbours=8,
        input_feature_names=FEATURES.DEEPCORE,
    )


def _energy_tasks(hidden_sizes: List[int]) -> List[EnergyReconstruction]:
    return [
        EnergyReconstruction(
            hidden_size=h, target_labels="energy", loss_function=MSELoss()
        )
        for h in hidden_sizes
    ]


def _batch(n: int = 3) -> Data:
    data = Data(x=torch.zeros(n, 1), energy=torch.ones(n))
    data.num_graphs = n
    return data


def test_split_routing_and_detached_chunks() -> None:
    """Tasks receive their chunks; shared chunks carry no gradient."""
    backbone = _ConstantBackbone(nb_outputs=6)
    model = StandardModel(
        data_representation=_graph_definition(),
        backbone=backbone,
        tasks=_energy_tasks([2, 4, 6, 4]),
        split=[[2, 2, 2], [0, [0, 1], [None, [0, 1, 2]], [2, [1]]]],
    )
    captured: Dict[int, Tensor] = {}
    for i, task in enumerate(model._tasks):
        task.register_forward_pre_hook(
            lambda _m, args, i=i: captured.__setitem__(i, args[0])
        )
    model(_batch())

    w = backbone.weight.detach()
    assert torch.equal(captured[0][0], w[0:2])
    assert torch.equal(captured[1][0], w[0:4])
    assert torch.equal(captured[2][0], w[0:6])
    assert torch.equal(captured[3][0], torch.cat([w[4:6], w[2:4]]))
    assert captured[0].requires_grad
    assert not captured[2].requires_grad  # fully detached
    assert captured[3].requires_grad  # own chunk keeps its gradient


def test_split_validates_sizes() -> None:
    """Split sizes must add up to the backbone output."""
    with pytest.raises(AssertionError):
        StandardModel(
            data_representation=_graph_definition(),
            backbone=_ConstantBackbone(nb_outputs=6),
            tasks=_energy_tasks([2, 2]),
            split=[[2, 2], [0, 1]],
        )


def test_loss_weight_balancing() -> None:
    """Losses pass through before activation and are reweighted after."""
    balancing = LossWeightBalancing(n_tasks=2, late_activation=1)
    losses = [torch.tensor(0.5), torch.tensor(2.0)]
    assert balancing(losses, epoch=0) == losses

    with torch.no_grad():
        balancing.noise_params[1].fill_(1.0)
    weighted = balancing(losses, epoch=1)
    softplus = torch.nn.functional.softplus
    assert torch.isclose(weighted[0], softplus(losses[0]))
    assert torch.isclose(
        weighted[1], torch.exp(torch.tensor(-1.0)) * softplus(losses[1]) + 0.5
    )


def test_learned_multitask_weights_param_groups() -> None:
    """Balancing parameters get their own group with a reduced lr."""
    model = StandardModel(
        data_representation=_graph_definition(),
        backbone=_ConstantBackbone(nb_outputs=4),
        tasks=_energy_tasks([4, 4]),
        optimizer_kwargs={"lr": 1e-3},
        learned_multitask_weights=0,
    )
    groups = model.configure_optimizers()["optimizer"].param_groups
    assert len(groups) == 2
    assert groups[0]["lr"] == pytest.approx(1e-3)
    assert groups[1]["lr"] == pytest.approx(1e-5)
    assert len(groups[1]["params"]) == 2


def test_detach_backbone() -> None:
    """A task with `detach_backbone` sends no gradient to its input."""
    task = EnergyReconstruction(
        hidden_size=4,
        target_labels="energy",
        loss_function=MSELoss(),
        detach_backbone=True,
    )
    x = torch.randn(3, 4, requires_grad=True)
    task(x).sum().backward()
    assert x.grad is None


def test_direction_tasks() -> None:
    """Direction outputs are unit vectors; kappa scaling maps k to k + k^2."""
    x = torch.randn(5, 3)
    direction = DirectionReconstruction(
        hidden_size=3, target_labels="direction", loss_function=MSELoss()
    )._forward(x)
    assert torch.allclose(direction.norm(dim=1), torch.ones(5))

    kw: Dict[str, Any] = dict(
        hidden_size=3, target_labels="direction", loss_function=MSELoss()
    )
    plain = DirectionReconstructionWithKappa(**kw)._forward(x)
    scaled = DirectionReconstructionWithKappa(scaling=True, **kw)._forward(x)
    assert torch.allclose(scaled[:, :3], plain[:, :3])
    assert torch.allclose(scaled[:, 3], plain[:, 3] + plain[:, 3] ** 2)


def test_tasks_are_picklable() -> None:
    """Default transforms do not prevent pickling."""
    task = _energy_tasks([4])[0]
    pickle.loads(pickle.dumps(task))
