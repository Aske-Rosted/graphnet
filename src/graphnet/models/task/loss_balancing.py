"""Balancing of the task losses of multitask models.

A `LossBalancing` module turns the per-task losses of a `StandardModel` into
the terms that are summed and minimised. Balancing uses each task's loss
relative to its lower bound (`LossFunction.lower_bound`), i.e. how far the
task is from its best possible loss; this makes it applicable to negative
log-likelihood losses, which can be negative. Tasks whose loss has no known
bound are passed through unchanged.
"""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import torch
from torch import Tensor

from graphnet.models.model import Model

if TYPE_CHECKING:
    from graphnet.models.task import Task


class LossBalancing(Model):
    """Base class for balancing the task losses of a multitask model.

    Subclasses implement `_balance`, which maps the losses of the balanced
    tasks (those with a lower bound) to the terms to minimise. Balancing
    parameters get their own optimizer parameter group
    (see `param_group_options`).
    """

    def __init__(
        self,
        start_epoch: int = 0,
        lower_bounds: Optional[Sequence[Optional[float]]] = None,
        lr_scale: float = 1.0,
    ) -> None:
        """Construct `LossBalancing`.

        Args:
            start_epoch: First epoch in which the losses are balanced; before
                that they are passed through unchanged.
            lower_bounds: Optional per-task lower bounds overriding those of
                the tasks' loss functions. Entries set to `None` fall back
                to the loss function's bound.
            lr_scale: Factor applied to the model's learning rate for the
                balancing parameters (kept under learning-rate schedulers).
        """
        super().__init__()
        self._start_epoch = start_epoch
        self._bound_overrides = (
            list(lower_bounds) if lower_bounds is not None else None
        )
        self._lr_scale = lr_scale
        # Plain list: the loss functions are owned by the tasks.
        self._loss_functions: List[Any] = []
        self._balanced: List[int] = []

    def setup(self, tasks: Sequence["Task"], detached: Sequence[bool]) -> None:
        """Register the tasks whose losses are balanced.

        Called by the model once its tasks are known.

        Args:
            tasks: The model's tasks, in order.
            detached: Whether each task's input is detached from the shared
                model, such that its loss only trains its own parameters.
        """
        if self._bound_overrides is not None:
            assert len(self._bound_overrides) == len(
                tasks
            ), "`lower_bounds` must have one entry per task."
        self._loss_functions = [task._loss_function for task in tasks]
        self._balanced = [
            i for i in range(len(tasks)) if self._lower_bound(i) is not None
        ]
        for i in range(len(tasks)):
            if i not in self._balanced and not detached[i]:
                self.warning(
                    f"Task {i} ({tasks[i].__class__.__name__}) has no known"
                    " loss lower bound and is not balanced, although it"
                    " trains shared parameters. Provide `lower_bounds` or"
                    " detach it."
                )
        self._build(len(self._balanced))

    @property
    def balanced_tasks(self) -> List[int]:
        """Return the indices of the balanced tasks."""
        return list(self._balanced)

    @property
    def param_group_options(self) -> Dict[str, Any]:
        """Return optimizer options for the balancing parameters."""
        return {"weight_decay": 0.0, "lr_scale": self._lr_scale}

    def forward(self, losses: List[Tensor], epoch: int) -> List[Tensor]:
        """Return the balanced losses, one per task."""
        if epoch < self._start_epoch or len(self._balanced) == 0:
            return list(losses)
        excess = torch.stack(
            [
                (losses[i] - self._lower_bound(i)).clamp(min=0.0)
                for i in self._balanced
            ]
        )
        balanced = self._balance(excess)
        out = list(losses)
        for k, i in enumerate(self._balanced):
            out[i] = balanced[k]
        return out

    def on_train_batch_end(self, model: Any, batch: Any) -> None:
        """Update after an optimizer step (called by the model)."""

    def weights(self) -> Optional[Tensor]:
        """Return the current weights of the balanced tasks, if defined."""
        return None

    def _lower_bound(self, i: int) -> Optional[float]:
        if self._bound_overrides is not None:
            override = self._bound_overrides[i]
            if override is not None:
                return float(override)
        return self._loss_functions[i].lower_bound

    @abstractmethod
    def _build(self, n_tasks: int) -> None:
        """Create the balancing state for `n_tasks` balanced tasks."""

    @abstractmethod
    def _balance(self, excess: Tensor) -> Tensor:
        """Map the losses above their bounds, shape [n], to terms [n]."""


class UncertaintyWeighting(LossBalancing):
    """Weigh task losses with learned homoscedastic uncertainties.

    Uncertainty weighting of Kendall, Gal & Cipolla, "Multi-Task Learning
    Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    (CVPR 2018, https://arxiv.org/abs/1705.07115), applied to each task's
    loss above its lower bound, L' = L - L_min >= 0:

        term_i = exp(-eta_i) * L'_i + eta_i / 2,

    with one learned `eta_i` per balanced task. For a fixed loss the weight
    exp(-eta_i) settles at 1 / (2 L'_i): tasks far from their best possible
    loss are down-weighted relative to tasks close to it.
    """

    def _build(self, n_tasks: int) -> None:
        self.log_variances = torch.nn.Parameter(torch.zeros(n_tasks))

    def _balance(self, excess: Tensor) -> Tensor:
        eta = self.log_variances
        return torch.exp(-eta) * excess + 0.5 * eta

    def weights(self) -> Tensor:
        """Return the current task weights exp(-eta)."""
        return torch.exp(-self.log_variances.detach())
