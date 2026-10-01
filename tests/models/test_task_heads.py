"""Unit tests for configurable task heads."""

import pytest
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


def test_default_head_is_linear() -> None:
    """Without `head` the task keeps its single linear layer."""
    task = _task()
    assert isinstance(task._affine, torch.nn.Linear)
    assert set(task.state_dict()) == {"_affine.weight", "_affine.bias"}


def test_head_shorthand_builds_mlp() -> None:
    """A list of widths builds an MLP with default options."""
    task = _task(head=[16, 8])
    head = task._affine
    assert isinstance(head, MLPHead)
    kinds = [type(layer).__name__ for layer in head.layers]
    assert kinds == [
        "LayerNorm",
        "Linear",
        "GELU",
        "Linear",
        "GELU",
        "Linear",
    ]
    assert head.layers[1].in_features == 12
    assert head.layers[-1].out_features == task.nb_inputs == 3
    assert task(torch.randn(5, 12)).shape == (5, 3)


def test_mlp_head_options() -> None:
    """Activation, input norm and dropout follow the arguments."""
    head = MLPHead([10], activation="SiLU", input_norm=False, dropout=0.1)
    task = _task(head=head)
    kinds = [type(layer).__name__ for layer in task._affine.layers]
    assert kinds == ["Linear", "SiLU", "Dropout", "Linear"]


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


def test_head_misuse() -> None:
    """A head is built once and not combined with `disable_affine`."""
    head = MLPHead([4])
    _task(head=head)
    with pytest.raises(AssertionError, match="already built"):
        _task(head=head)
    with pytest.raises(AssertionError, match="disable_affine"):
        _task(head=[4], disable_affine=True)
    with pytest.raises(AssertionError, match="no activation"):
        MLPHead([4], activation="NotAnActivation")
