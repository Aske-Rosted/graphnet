from typing import List
from graphnet.models.task import StandardLearnedTask

from graphnet.models import Model

import torch

# import all the classification tasks
import sys
import inspect
import importlib


class LossWeightBalancing(Model):
    """Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry
    and Semantics (Kendall et al; CVPR 2018).

    Implementation from
    https://github.com/murnanedaniel/Dynamic-Loss-Weighting/blob/master/loss_models.py
    """

    def __init__(
        self, tasks: List[StandardLearnedTask], late_activation: int = 1
    ):
        super(LossWeightBalancing, self).__init__()
        # Initialize the noise parameters as one
        self.noise_params = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(())) for _ in range(len(tasks))]
        )
        self.late_activation = late_activation

    def forward(self, losses: list) -> torch.tensor:
        """
        Computes the total loss as a function of a list of classification losses.
        TODO: Handle regressions losses, which require a factor of 2 (see arxiv.org/abs/1705.07115 page 4)


        Each loss coeff is of the form: :math:`\frac{1}{\sqrt{\eta_i}} \cdot \ell_i + \log(\eta_i)`
        Total loss: :math:`\ell = \sum_{i=1}^{k} \left\[ \frac{1}{\sqrt{\eta_i}} \cdot \ell_i + \log(\eta_i) \right\]`
        """
        if self.current_epoch >= self.late_activation:
            for i, loss in enumerate(losses):

                loss = torch.log1p(
                    torch.exp(loss)
                )  # softplus ensures the loss is always positive.
                losses[i] = (
                    torch.exp(-self.noise_params[i]) * loss
                    + 0.5 * self.noise_params[i]
                ).mean()

        return losses
