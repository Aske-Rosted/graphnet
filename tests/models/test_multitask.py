"""Unit tests for multitask support in `StandardModel` and its tasks."""

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
from graphnet.models.task.loss_balancing import (
    FAMO,
    LossBalancing,
    UncertaintyWeighting,
)
from graphnet.models.task.reconstruction import (
    DirectionReconstruction,
    DirectionReconstructionWithKappa,
    EnergyReconstruction,
)
from graphnet.training.loss_functions import CauchyLoss, MSELoss


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


class _TokenBackbone(_ConstantBackbone):
    """Constant backbone with a token, a norm layer and no_weight_decay."""

    def __init__(self, nb_outputs: int):
        super().__init__(nb_outputs)
        self.tokens = torch.nn.Parameter(torch.zeros(nb_outputs))
        self.norm = torch.nn.LayerNorm(nb_outputs)

    def no_weight_decay(self) -> set:
        return {"tokens"}


def _mixed_tasks() -> List[EnergyReconstruction]:
    """Cauchy (bound log 0.1), MSE (bound 0), heteroscedastic (none)."""
    losses = [CauchyLoss(alpha=0.1, frac=0.0), MSELoss(), CauchyLoss(frac=1)]
    return [
        EnergyReconstruction(
            hidden_size=h, target_labels="energy", loss_function=loss
        )
        for h, loss in zip([2, 2, 6], losses)
    ]


def _balanced_model(
    balancing: LossBalancing, split_last_detached: bool = True
) -> StandardModel:
    last = [None, [0, 1, 2]] if split_last_detached else [0, 1, 2]
    return StandardModel(
        data_representation=_graph_definition(),
        backbone=_TokenBackbone(nb_outputs=6),
        tasks=_mixed_tasks(),
        split=[[2, 2, 2], [0, 1, last]],
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs={"lr": 1e-3},
        loss_balancing=balancing,
    )


def test_uncertainty_weighting_uses_lower_bounds() -> None:
    """Balanced terms use L - L_min; tasks without a bound pass through."""
    balancing = UncertaintyWeighting()
    _balanced_model(balancing)
    assert balancing.balanced_tasks == [0, 1]

    losses = [torch.tensor(0.5), torch.tensor(2.0), torch.tensor(-7.0)]
    with torch.no_grad():
        balancing.log_variances.copy_(torch.tensor([0.0, 1.0]))
    out = balancing(losses, epoch=0)
    expected_0 = 0.5 - torch.log(torch.tensor(0.1))
    assert torch.isclose(out[0], expected_0)
    assert torch.isclose(out[1], torch.exp(torch.tensor(-1.0)) * 2.0 + 0.5)
    assert out[2] is losses[2]


def test_balancing_and_weight_decay_param_groups() -> None:
    """Balancing params and excluded params get groups without decay."""
    balancing = UncertaintyWeighting(lr_scale=20.0)
    model = StandardModel(
        data_representation=_graph_definition(),
        backbone=_TokenBackbone(nb_outputs=6),
        tasks=_mixed_tasks(),
        split=[[2, 2, 2], [0, 1, [None, [0, 1, 2]]]],
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs={"lr": 1e-3, "weight_decay": 0.01},
        loss_balancing=balancing,
        exclude_from_weight_decay=True,
    )
    groups = model.configure_optimizers()["optimizer"].param_groups
    assert [g["weight_decay"] for g in groups] == [0.01, 0.0, 0.0]
    assert groups[2]["lr"] == pytest.approx(2e-2)
    assert groups[2]["params"] == [balancing.log_variances]

    backbone = model.backbone
    no_decay = {id(p) for p in groups[1]["params"]}
    assert id(backbone.tokens) in no_decay
    assert id(backbone.norm.weight) in no_decay
    assert id(model._tasks[0]._affine.bias) in no_decay
    assert id(backbone.weight) not in no_decay
    assert id(model._tasks[0]._affine.weight) not in no_decay


@pytest.mark.parametrize(
    "scheduler_class, scheduler_kwargs",
    [
        (torch.optim.lr_scheduler.LambdaLR, {"lr_lambda": lambda e: 0.9**e}),
        (torch.optim.lr_scheduler.StepLR, {"step_size": 2, "gamma": 0.5}),
    ],
)
def test_lr_scale_survives_scheduler(
    scheduler_class: Any, scheduler_kwargs: Dict[str, Any]
) -> None:
    """A group's lr stays lr_scale times the scheduled lr."""
    model = _balanced_model(UncertaintyWeighting(lr_scale=20.0))
    model._scheduler_class = scheduler_class
    model._scheduler_kwargs = scheduler_kwargs
    config = model.configure_optimizers()
    optimizer = config["optimizer"]
    scheduler = config["lr_scheduler"]["scheduler"]
    for _ in range(6):
        main, balancing = optimizer.param_groups[0], optimizer.param_groups[-1]
        assert balancing["lr"] == pytest.approx(20.0 * main["lr"])
        assert main["lr"] == pytest.approx(scheduler.get_last_lr()[0])
        optimizer.step()
        model.lr_scheduler_step(scheduler, None)
    assert optimizer.param_groups[0]["lr"] < 1e-3


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


def test_task_head_and_loss_run_in_float32_under_autocast() -> None:
    """Predictions and losses stay float32 under bfloat16 autocast."""
    task = EnergyReconstruction(
        hidden_size=4, target_labels="energy", loss_function=MSELoss()
    )
    x = torch.randn(8, 4)
    data = Data(energy=torch.rand(8))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        pred = task(x.bfloat16())
        loss = task.compute_loss(pred, data)
    assert pred.dtype == torch.float32
    assert loss.dtype == torch.float32
    # Same as a full-precision evaluation of the (bf16-rounded) input
    assert torch.allclose(pred, task(x.bfloat16().float()))


def test_famo_logit_gradient_matches_autograd() -> None:
    """The logit update direction is J_softmax^T delta."""
    famo = FAMO(lower_bounds=[0.0, 0.0, 0.0], lr=0.0, weight_decay=0.0)
    _balanced_model(famo)
    logits = torch.tensor([0.3, -0.2, 0.1], requires_grad=True)
    with torch.no_grad():
        famo.logits.copy_(logits)
    prev, new = torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0.5, 1.9, 2.0])
    delta = torch.log(prev) - torch.log(new)
    (expected,) = torch.autograd.grad(
        torch.softmax(logits, 0), logits, grad_outputs=delta
    )
    famo._update(prev, new)
    # Adam's first step moves each logit by -lr * sign(grad); check m instead
    assert torch.allclose(famo._adam_m / 0.1, expected, atol=1e-6)
