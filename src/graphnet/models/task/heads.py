"""Prediction heads mapping a task's input to its raw outputs.

A `LearnedTask` maps the latent representation it receives (`hidden_size`
columns) to the `nb_inputs` columns its `_forward` expects. By default this
is a single linear layer. A `TaskHead` replaces it with a configurable
module; the task builds the head with its own input and output sizes, so a
head configuration never states them.
"""

from typing import List, Sequence

import torch
from torch import Tensor

from graphnet.models import Model


class TaskHead(Model):
    """Base class for prediction heads of `LearnedTask`s.

    Subclasses store their options in `__init__` and create their layers in
    `_build`, which the task calls once with its input and output sizes.
    """

    def __init__(self) -> None:
        """Construct `TaskHead`."""
        super().__init__()
        self._built = False

    def build(self, in_features: int, out_features: int) -> "TaskHead":
        """Create the layers for the given input and output sizes.

        Args:
            in_features: Number of input columns (the task's `hidden_size`).
            out_features: Number of output columns (the task's `nb_inputs`).

        Returns:
            The head itself.
        """
        assert not self._built, (
            f"{self.__class__.__name__} is already built; use a separate "
            "head for each task."
        )
        self._build(in_features, out_features)
        self._built = True
        return self

    def _build(self, in_features: int, out_features: int) -> None:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """Map `x` to the task's raw outputs."""
        raise NotImplementedError


class MLPHead(TaskHead):
    """Multi-layer perceptron head.

    `[LayerNorm] -> (Linear -> activation [-> Dropout]) x n -> Linear`, with
    one hidden block per entry in `hidden_layers`. The input normalization
    equalizes the scales of concatenated inputs, e.g. several backbone
    outputs routed to one task.
    """

    def __init__(
        self,
        hidden_layers: Sequence[int],
        activation: str = "GELU",
        input_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        """Construct `MLPHead`.

        Args:
            hidden_layers: Widths of the hidden layers.
            activation: Name of the activation class in `torch.nn`, e.g.
                `GELU`, `SiLU` or `ReLU`.
            input_norm: If True, apply `LayerNorm` to the head input.
            dropout: Dropout probability after each hidden activation.
        """
        super().__init__()
        assert len(hidden_layers) > 0, "Give at least one hidden layer."
        assert hasattr(
            torch.nn, activation
        ), f"`torch.nn` has no activation `{activation}`."
        self._hidden_layers = list(hidden_layers)
        self._activation = activation
        self._input_norm = input_norm
        self._dropout = dropout
        self.layers = torch.nn.Sequential()  # filled by `build`

    def _build(self, in_features: int, out_features: int) -> None:
        layers: List[torch.nn.Module] = []
        if self._input_norm:
            layers.append(torch.nn.LayerNorm(in_features))
        width = in_features
        for hidden in self._hidden_layers:
            layers.append(torch.nn.Linear(width, hidden))
            layers.append(getattr(torch.nn, self._activation)())
            if self._dropout > 0:
                layers.append(torch.nn.Dropout(self._dropout))
            width = hidden
        layers.append(torch.nn.Linear(width, out_features))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Map `x` to the task's raw outputs."""
        assert self._built, "Call `build` first."
        return self.layers(x)
