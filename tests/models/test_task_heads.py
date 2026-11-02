"""Unit tests for configurable task heads."""

import torch

from graphnet.models.task.heads import MLPHead
from graphnet.models.task.reconstruction import (
    EnergyReconstruction,
    PositionReconstruction,
)
from graphnet.training.loss_functions import MSELoss
from graphnet.utilities.config import ModelConfig


def _task(**kwargs: object) -> PositionReconstruction:
    return PositionReconstruction(
        hidden_size=12,
        target_labels=["x", "y", "z"],
        loss_function=MSELoss(),
        **kwargs,
    )


def test_head_from_model_config() -> None:
    """A head given as `ModelConfig` (as in YAML configs) is built."""
    config = ModelConfig(
        class_name="MLPHead",
        arguments={"hidden_layers": [7], "activation": "ReLU"},
    )
    head = MLPHead.from_config(config, trust=True)
    task = EnergyReconstruction(
        hidden_size=4,
        target_labels="energy",
        loss_function=MSELoss(),
        head=head,
    )
    assert task(torch.randn(3, 4)).shape == (3, 1)
    assert isinstance(task._affine.layers[2], torch.nn.ReLU)


def test_head_is_trained_and_saved() -> None:
    """Head parameters receive gradients and are in the state dict."""
    task = _task(head=[8])
    task(torch.randn(4, 12)).sum().backward()
    head_parameters = list(task._affine.parameters())
    assert all(p.grad is not None for p in head_parameters)
    assert any(key.startswith("_affine.layers.") for key in task.state_dict())
