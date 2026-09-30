"""Standard model class(es)."""

from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union, Type
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch.optim import Adam

from graphnet.models.gnn.gnn import GNN
from graphnet.models import Model
from .easy_model import EasySyntax
from graphnet.models.task import StandardLearnedTask
from graphnet.models.task.loss_balancing import LossBalancing
from graphnet.models.data_representation import (
    GraphDefinition,
    DataRepresentation,
)


class StandardModel(EasySyntax):
    """A Standard way of combining model components in GraphNeT.

    This model is compatible with the vast majority of supervised
    learning tasks such as regression, binary and multi-label
    classification.

    Capable of producing both event-level and pulse-level predictions.
    """

    def __init__(
        self,
        tasks: Union[StandardLearnedTask, List[StandardLearnedTask]],
        data_representation: Optional[DataRepresentation] = None,
        graph_definition: Optional[GraphDefinition] = None,
        backbone: Optional[Model] = None,
        gnn: Optional[GNN] = None,
        split: Optional[List[List]] = None,
        optimizer_class: Type[torch.optim.Optimizer] = Adam,
        optimizer_kwargs: Optional[Dict] = None,
        scheduler_class: Optional[type] = None,
        scheduler_kwargs: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None,
        loss_balancing: Optional[LossBalancing] = None,
        exclude_from_weight_decay: bool = False,
    ) -> None:
        """Construct `StandardModel`.

        Args:
            tasks: Task(s) predicted from the backbone output.
            data_representation: Representation turning input data into
                model input.
            graph_definition: Deprecated alias of `data_representation`.
            backbone: Model producing the latent output fed to the tasks.
            gnn: Deprecated alias of `backbone`.
            split: Optional routing of the backbone output to the tasks, as
                `[sizes, indices]`. The output is split along the last
                dimension into chunks of `sizes`; `indices[i]` selects the
                input of task `i` as either a chunk index, a list of chunk
                indices (concatenated), or a pair `[own, shared]` where the
                `shared` chunks are concatenated after the `own` chunk(s)
                with gradients detached (`own` may be `None`).
            optimizer_class: Optimizer class used during training.
            optimizer_kwargs: Keyword arguments passed to `optimizer_class`.
            scheduler_class: Learning-rate scheduler class.
            scheduler_kwargs: Keyword arguments passed to `scheduler_class`.
            scheduler_config: Lightning scheduler configuration.
            loss_balancing: Optional `LossBalancing` combining the task
                losses (e.g. `UncertaintyWeighting`). Its parameters get
                their own optimizer parameter group without weight decay.
            exclude_from_weight_decay: If True, biases, normalization
                layers and the parameters listed by the backbone's
                `no_weight_decay()` are not weight-decayed (relevant for
                optimizers with decoupled weight decay such as AdamW).
        """
        # Base class constructor
        super().__init__(
            tasks=tasks,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
            scheduler_config=scheduler_config,
        )
        # DEPRECATION ARG GRAPH_DEFINITION: REMOVE AT 2.0 LAUNCH
        # See https://github.com/graphnet-team/graphnet/issues/647

        if (data_representation is None) & (graph_definition is not None):
            data_representation = graph_definition
            # Code continues after warning
            self.warning(
                "DeprecationWarning: Argument `graph_definition` will be"
                " deprecated in GraphNeT 2.0. Please use `data_representation`"
                " instead."
                ""
            )
        elif (data_representation is None) & (graph_definition is None):
            # Code stops
            raise TypeError(
                "__init__() missing 1 required keyword argument:"
                "'data_representation'"
            )

        # deprecation warnings
        if (backbone is None) & (gnn is not None):
            backbone = gnn
            # Code continues after warning
            self.warning(
                "DeprecationWarning: Argument `gnn` will be deprecated in"
                " GraphNeT 2.0. Please use `backbone` instead."
                ""
            )
        elif (backbone is None) & (gnn is None):
            # Code stops
            raise TypeError(
                "__init__() missing 1 required keyword argument:'backbone'"
            )

        # Checks
        assert isinstance(backbone, Model)
        assert isinstance(data_representation, DataRepresentation)

        # Member variable(s)
        self._data_representation = data_representation
        self.backbone = backbone

        self._split_sizes: Optional[List[int]] = None
        self._split_indices: List = []
        if split is not None:
            self._split_sizes, self._split_indices = split[0], split[1]
            assert len(self._split_indices) == len(
                self._tasks
            ), "`split` must provide one index entry per task."
            assert len(self._split_sizes) == (
                max(self._flatten_split_indices(self._split_indices)) + 1
            ), "`split` sizes and indices are inconsistent."
            assert sum(self._split_sizes) == self.backbone.nb_outputs, (
                "`split` sizes do not add up to the backbone output"
                " dimension."
            )

        self._exclude_from_weight_decay = exclude_from_weight_decay
        self.loss_balancing = loss_balancing
        if self.loss_balancing is not None:
            self.loss_balancing.setup(self._tasks, self._detached_tasks())

    def _detached_tasks(self) -> List[bool]:
        """Return whether each task's input is detached from the backbone."""
        detached = [task._detach_backbone for task in self._tasks]
        for i, indices in enumerate(self._split_indices):
            if (
                isinstance(indices, list)
                and len(indices) == 2
                and indices[0] is None
                and isinstance(indices[1], list)
            ):
                detached[i] = True
        return detached

    def _task_losses(self, preds: Tensor, data: List[Data]) -> List[Tensor]:
        """Return the (unbalanced) loss of each task."""
        data_merged = {}
        target_labels_merged = list(set(self.target_labels))
        for label in target_labels_merged:
            data_merged[label] = torch.cat([d[label] for d in data], dim=0)
        for task in self._tasks:
            if task._loss_weight is not None:
                data_merged[task._loss_weight] = torch.cat(
                    [d[task._loss_weight] for d in data], dim=0
                )

        return [
            task.compute_loss(pred, data_merged)
            for task, pred in zip(self._tasks, preds)
        ]

    def compute_loss(
        self, preds: Tensor, data: List[Data], verbose: bool = False
    ) -> Tensor:
        """Compute and sum losses across tasks."""
        losses = self._task_losses(preds, data)
        # Unbalanced task losses, e.g. for balancers updating after a step
        self._last_task_losses = torch.stack(
            [loss.detach() for loss in losses]
        )

        if not self.training:
            # Log the individual (unbalanced) task losses during validation.
            for i, loss in enumerate(losses):
                self.log(
                    f"i_loss_{i}",
                    loss,
                    prog_bar=False,
                    logger=True,
                    on_step=False,
                    on_epoch=True,
                    batch_size=len(preds[0]),
                    sync_dist=True,
                )
            if self.loss_balancing is not None:
                weights = self.loss_balancing.weights()
                if weights is not None:
                    tasks = self.loss_balancing.balanced_tasks
                    for i, weight in zip(tasks, weights):
                        self.log(
                            f"loss_weight_{i}",
                            weight,
                            on_step=False,
                            on_epoch=True,
                            batch_size=len(preds[0]),
                            sync_dist=True,
                        )

        if self.loss_balancing is not None:
            losses = self.loss_balancing(losses, self.current_epoch)

        if verbose:
            self.info(f"{losses}")
        assert all(
            loss.dim() == 0 for loss in losses
        ), "Please reduce loss for each task separately"
        return torch.sum(torch.stack(losses))

    def forward(
        self, data: Union[Data, List[Data]]
    ) -> List[Union[Tensor, Data]]:
        """Forward pass, chaining model components."""
        if isinstance(data, Data):
            data = [data]
        x_list = []
        for d in data:
            x = self.backbone(d)
            x_list.append(x)
        x = torch.cat(x_list, dim=0)

        if self._split_sizes is None:
            return [task(x) for task in self._tasks]

        chunks = x.split(list(self._split_sizes), dim=-1)
        return [
            task(self._route(chunks, indices))
            for indices, task in zip(self._split_indices, self._tasks)
        ]

    @staticmethod
    def _route(
        chunks: Tuple[Tensor, ...], indices: Union[int, List]
    ) -> Tensor:
        """Assemble one task's input from the split backbone output."""
        if isinstance(indices, int):
            return chunks[indices]
        if not isinstance(indices, list):
            raise TypeError(
                "Expected `split` indices of type int or list, got"
                f" {type(indices)}."
            )
        if all(isinstance(i, int) for i in indices):
            return torch.cat([chunks[i] for i in indices], dim=-1)

        assert len(indices) == 2, (
            "A nested `split` entry must be `[own, shared]`, with the"
            " `shared` chunks detached from the backbone."
        )
        own, shared = indices
        detached = torch.cat([chunks[i] for i in shared], dim=-1).detach()
        if own is None:
            return detached
        if isinstance(own, int):
            own = [own]
        return torch.cat(
            [torch.cat([chunks[i] for i in own], dim=-1), detached], dim=-1
        )

    @classmethod
    def _flatten_split_indices(cls, nested: List) -> Iterator[int]:
        for item in nested:
            if isinstance(item, list):
                yield from cls._flatten_split_indices(item)
            elif item is not None:
                yield item

    def shared_step(self, batch: List[Data], batch_idx: int) -> Tensor:
        """Perform shared step.

        Applies the forward pass and the following loss calculation,
        shared between the training and validation step.
        """
        preds = self(batch)
        loss = self.compute_loss(preds, batch)
        return loss

    def on_train_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        """Let the loss balancing update after the optimizer step."""
        if self.loss_balancing is not None:
            self.loss_balancing.on_train_batch_end(self, batch)

    def _no_decay_parameter_ids(self) -> Set[int]:
        """Return ids of biases, norm layers and backbone-listed params."""
        ids: Set[int] = set()
        norm_types = (
            torch.nn.LayerNorm,
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.GroupNorm,
        )
        for module in self.modules():
            if isinstance(module, norm_types):
                ids.update(id(p) for p in module.parameters())
        for name, param in self.named_parameters():
            if name.endswith(".bias"):
                ids.add(id(param))
        if hasattr(self.backbone, "no_weight_decay"):
            listed = set(self.backbone.no_weight_decay())
            for name, param in self.backbone.named_parameters():
                if name.split(".")[0] in listed or name in listed:
                    ids.add(id(param))
        return ids

    def _optimizer_parameters(self) -> Union[Iterator, List[Dict]]:
        """Return the parameter groups passed to the optimizer.

        The loss-balancing parameters form their own group with the options
        of `LossBalancing.param_group_options` (no weight decay, lr scale).
        With `exclude_from_weight_decay`, biases, normalization layers and
        the backbone's `no_weight_decay()` parameters form a group without
        weight decay.
        """
        balancing: List[torch.nn.Parameter] = []
        if self.loss_balancing is not None:
            balancing = list(self.loss_balancing.parameters())
        if not balancing and not self._exclude_from_weight_decay:
            return super()._optimizer_parameters()

        balancing_ids = {id(p) for p in balancing}
        no_decay_ids = (
            self._no_decay_parameter_ids()
            if self._exclude_from_weight_decay
            else set()
        )
        decay: List[torch.nn.Parameter] = []
        no_decay: List[torch.nn.Parameter] = []
        for param in self.parameters():
            if id(param) in balancing_ids:
                continue
            (no_decay if id(param) in no_decay_ids else decay).append(param)

        groups: List[Dict] = [{"params": decay}]
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})
        if balancing:
            assert self.loss_balancing is not None
            groups.append(
                {
                    "params": balancing,
                    **self.loss_balancing.param_group_options,
                }
            )
        return groups

    def validate_tasks(self) -> None:
        """Verify that self._tasks contain compatible elements."""
        accepted_tasks = StandardLearnedTask
        for task in self._tasks:
            assert isinstance(task, accepted_tasks)

    # DEPRECATION ARG GRAPH_DEFINITION: REMOVE AT 2.0 LAUNCH
    # See https://github.com/graphnet-team/graphnet/issues/647
    @property
    def _graph_definition(self) -> DataRepresentation:
        """Return the graph definition."""
        self.warning(
            "DeprecationWarning: `_graph_definition` will be deprecated in"
            " GraphNeT 2.0. Please use `_data_representation` instead."
        )
        return self._data_representation
