"""Utilities for training models with several tasks."""

from typing import List

import torch
from torch import Tensor
from torch.nn.functional import softplus

from graphnet.models.model import Model


class LossWeightBalancing(Model):
    """Weigh task losses with learned homoscedastic uncertainties.

    Implements the uncertainty weighting of Kendall, Gal & Cipolla, "Multi-
    Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and
    Semantics" (CVPR 2018, https://arxiv.org/abs/1705.07115), following
    https://github.com/murnanedaniel/Dynamic-Loss-Weighting. Each task loss
    `l_i` is made positive with a softplus and replaced by
    `exp(-eta_i) * softplus(l_i) + eta_i / 2`, with one learned `eta_i` per
    task.
    """

    def __init__(self, n_tasks: int, late_activation: int = 1):
        """Construct `LossWeightBalancing`.

        Args:
            n_tasks: Number of task losses to weigh.
            late_activation: First epoch at which the weighting is applied;
                before that the losses are passed through unchanged.
        """
        super().__init__()
        self.noise_params = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(())) for _ in range(n_tasks)]
        )
        self._late_activation = late_activation

    def forward(self, losses: List[Tensor], epoch: int) -> List[Tensor]:
        """Weigh `losses` if `epoch` has reached `late_activation`."""
        if epoch < self._late_activation:
            return losses
        return [
            (
                torch.exp(-eta) * softplus(loss, threshold=1e6) + 0.5 * eta
            ).mean()
            for loss, eta in zip(losses, self.noise_params)
        ]
