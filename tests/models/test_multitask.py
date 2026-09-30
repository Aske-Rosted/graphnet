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
from graphnet.models.task.loss_balancing import (
    LossBalancing,
    UncertaintyWeighting,
)
from graphnet.models.task.reconstruction import (
    DirectionReconstruction,
    DirectionReconstructionWithKappa,
    EnergyReconstruction,
)
from graphnet.models.task.task import IdentityTaskWithUncertainty
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


def test_split_validates_sizes() -> None:
    """Split sizes must add up to the backbone output."""
    with pytest.raises(AssertionError):
        StandardModel(
            data_representation=_graph_definition(),
            backbone=_ConstantBackbone(nb_outputs=6),
            tasks=_energy_tasks([2, 2]),
            split=[[2, 2], [0, 1]],
        )


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


def test_uncertainty_weighting_start_epoch_and_overrides() -> None:
    """Before `start_epoch` losses pass through; overrides replace bounds."""
    balancing = UncertaintyWeighting(
        start_epoch=2, lower_bounds=[None, -1.0, 0.0]
    )
    _balanced_model(balancing)
    assert balancing.balanced_tasks == [0, 1, 2]
    losses = [torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0)]
    assert balancing(losses, epoch=1) == losses
    out = balancing(losses, epoch=2)
    assert torch.isclose(out[1], torch.tensor(2.0))  # 1 - (-1)


def test_uncertainty_weights_converge_to_inverse_excess() -> None:
    """For fixed losses the weights settle at 1 / (2 (L - L_min))."""
    balancing = UncertaintyWeighting(lower_bounds=[0.0, 0.0, 0.0])
    _balanced_model(balancing)
    losses = [torch.tensor(0.5), torch.tensor(2.0), torch.tensor(8.0)]
    optimizer = torch.optim.Adam(balancing.parameters(), lr=0.05)
    for _ in range(2000):
        optimizer.zero_grad()
        torch.stack(balancing(losses, epoch=0)).sum().backward()
        optimizer.step()
    expected = torch.tensor([1.0, 0.25, 0.0625])
    assert torch.allclose(balancing.weights(), expected, rtol=1e-2)


def test_unbounded_shared_task_warns(monkeypatch: Any) -> None:
    """A shared task without a lower bound triggers a warning."""
    messages: List[str] = []
    monkeypatch.setattr(
        LossBalancing, "warning", lambda self, msg: messages.append(msg)
    )
    _balanced_model(UncertaintyWeighting(), split_last_detached=True)
    assert messages == []
    _balanced_model(UncertaintyWeighting(), split_last_detached=False)
    assert len(messages) == 1 and "Task 2" in messages[0]


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


def test_loss_parameters_are_not_weight_decayed() -> None:
    """A learned loss scale joins the group without weight decay."""
    loss = CauchyLoss(alpha=0.1, frac=0.0, learn_alpha=True, nb_outputs=1)
    task = EnergyReconstruction(
        hidden_size=2, target_labels="energy", loss_function=loss
    )
    model = StandardModel(
        data_representation=_graph_definition(),
        backbone=_TokenBackbone(nb_outputs=2),
        tasks=[task],
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs={"lr": 1e-3, "weight_decay": 0.01},
        exclude_from_weight_decay=True,
    )
    groups = model.configure_optimizers()["optimizer"].param_groups
    assert [g["weight_decay"] for g in groups] == [0.01, 0.0]
    assert id(loss.log_alpha) in {id(p) for p in groups[1]["params"]}


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


def test_identity_task_with_uncertainty() -> None:
    """Values pass through; scales are positive and start at one."""
    task = IdentityTaskWithUncertainty(
        nb_outputs=2,
        target_labels=["a", "b"],
        hidden_size=4,
        loss_function=CauchyLoss(frac=1.0),
    )
    assert task.nb_inputs == 4
    assert task.default_prediction_labels == [
        "target_0_pred",
        "target_1_pred",
        "target_0_scale",
        "target_1_scale",
    ]
    x = torch.tensor([[1.0, -2.0, 0.0, -1.0]])
    out = task._forward(x)
    assert torch.allclose(out[:, :2], x[:, :2])
    assert torch.allclose(out[:, 2:], torch.exp(x[:, 2:]))

    pred = task(torch.randn(8, 4))
    data = Data(a=torch.randn(8), b=torch.randn(8))
    loss = task.compute_loss(pred, data)
    assert torch.isfinite(loss)
    assert task._loss_function.lower_bound is None


def test_tasks_are_picklable() -> None:
    """Default transforms do not prevent pickling."""
    task = _energy_tasks([4])[0]
    pickle.loads(pickle.dumps(task))


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
