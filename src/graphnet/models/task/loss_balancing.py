"""Balancing of the task losses of multitask models.

A `LossBalancing` module turns the per-task losses of a `StandardModel` into
the terms that are summed and minimised. Balancing uses each task's loss
relative to its lower bound (`LossFunction.lower_bound`), i.e. how far the
task is from its best possible loss; this makes it applicable to negative
log-likelihood losses, which can be negative. Tasks whose loss has no known
bound are passed through unchanged.
"""

from abc import abstractmethod
from contextlib import nullcontext
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


class FAMO(LossBalancing):
    """Fast Adaptive Multitask Optimization.

    Liu, Liu, Jin, Stone & Liu, "FAMO: Fast Adaptive Multitask
    Optimization" (NeurIPS 2023, https://arxiv.org/abs/2306.03792). Task
    weights z = softmax(xi) are adapted after every optimizer step so that
    all tasks decrease their log-loss at a similar rate. The model minimises

        sum_i (z_i / c) * log(D_i),   c = sum_j z_j / D_j,

    with D_i = L_i - L_min_i + eps the loss above its lower bound, i.e. the
    gradient of task i is weighted by z_i / (c D_i). The logits xi are
    updated from the change in log(D) over the step, delta_i =
    log(D_i before) - log(D_i after), by an Adam step along
    J_softmax^T delta (plus weight decay): tasks that improved more than
    the weighted average lose weight, slower tasks gain weight. Rescaling a
    task loss by a constant has no effect.

    The logits are buffers, not model parameters: they are not updated by
    the model's optimizer, move with the model and are saved in checkpoints.
    Defaults follow the authors' reference implementation
    (https://github.com/Cranial-XIX/FAMO, `methods/weight_methods.py`),
    whose training scripts also clip the model's gradient norm at 1.
    """

    def __init__(
        self,
        start_epoch: int = 0,
        lower_bounds: Optional[Sequence[Optional[float]]] = None,
        lr: float = 0.025,
        weight_decay: float = 1e-5,
        update_on: str = "same_batch",
        eps: float = 1e-8,
    ) -> None:
        """Construct `FAMO`.

        Args:
            start_epoch: First epoch in which the losses are balanced.
            lower_bounds: Optional per-task lower bounds overriding those of
                the tasks' loss functions.
            lr: Learning rate of the Adam update of the task logits.
            weight_decay: Weight decay of the task logits (pulls the weights
                towards uniform).
            update_on: "same_batch" re-evaluates the losses on the batch of
                the step after the optimizer step (one extra forward pass
                without gradients, as in the reference implementation);
                "next_batch" uses the losses of the next training batch
                instead (no extra cost, noisier).
            eps: Added to the losses above their bounds before the log.
        """
        super().__init__(start_epoch=start_epoch, lower_bounds=lower_bounds)
        assert update_on in ("same_batch", "next_batch"), update_on
        self._lr = lr
        self._weight_decay = weight_decay
        self._update_on = update_on
        self._eps = eps
        self._betas = (0.9, 0.999)
        self._prev: Optional[Tensor] = None

    def _build(self, n_tasks: int) -> None:
        self.register_buffer("logits", torch.zeros(n_tasks))
        self.register_buffer("_adam_m", torch.zeros(n_tasks))
        self.register_buffer("_adam_v", torch.zeros(n_tasks))
        self.register_buffer("_adam_step", torch.zeros(()))

    def weights(self) -> Tensor:
        """Return the current task weights softmax(xi)."""
        return torch.softmax(self.logits, dim=0)

    def _balance(self, excess: Tensor) -> Tensor:
        d = excess + self._eps
        z = self.weights().to(d.dtype)
        c = (z / d).sum().detach()
        if self.training and self._update_on == "next_batch":
            if self._prev is not None:
                self._update(self._prev, d.detach())
        if self.training:
            self._prev = d.detach()
        return z / c * torch.log(d)

    def on_train_batch_end(self, model: Any, batch: Any) -> None:
        """Update the task weights from the loss change over the step."""
        if self._update_on != "same_batch" or self._prev is None:
            return
        if model.current_epoch < self._start_epoch:
            return
        with torch.no_grad(), _forward_context(model):
            batch = [batch] if not isinstance(batch, list) else batch
            losses = model._task_losses(model(batch), batch)
        new = torch.stack(
            [
                (losses[i].detach() - self._lower_bound(i)).clamp(min=0.0)
                for i in self._balanced
            ]
        )
        self._update(self._prev, new + self._eps)
        self._prev = None

    def _update(self, prev: Tensor, new: Tensor) -> None:
        """Adam step on the logits from the log-loss decrease."""
        prev, new = _mean_across_ranks(prev), _mean_across_ranks(new)
        delta = torch.log(prev) - torch.log(new)
        z = self.weights()
        # J_softmax^T delta = z * (delta - z.delta)
        grad = z * (delta - (z * delta).sum())
        grad = grad + self._weight_decay * self.logits
        b1, b2 = self._betas
        self._adam_step += 1
        self._adam_m.mul_(b1).add_((1 - b1) * grad)
        self._adam_v.mul_(b2).add_((1 - b2) * grad * grad)
        m_hat = self._adam_m / (1 - b1**self._adam_step)
        v_hat = self._adam_v / (1 - b2**self._adam_step)
        self.logits -= self._lr * m_hat / (v_hat.sqrt() + 1e-8)


def _mean_across_ranks(x: Tensor) -> Tensor:
    """Average `x` over distributed ranks (identity if not distributed)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        x = x.clone()
        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
        x /= torch.distributed.get_world_size()
    return x


def _forward_context(model: Any) -> Any:
    """Return the trainer's precision context (e.g. autocast), if any."""
    trainer = getattr(model, "_trainer", None)
    if trainer is not None:
        return trainer.precision_plugin.forward_context()
    return nullcontext()
