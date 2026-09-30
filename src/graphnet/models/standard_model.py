"""Standard model class(es)."""

from typing import Dict, Iterator, List, Optional, Tuple, Union, Type
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch.optim import Adam

from graphnet.models.gnn.gnn import GNN
from graphnet.models import Model
from .easy_model import EasySyntax
from graphnet.models.task import StandardLearnedTask
from graphnet.models.task.multitask_utils import LossWeightBalancing
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
        learned_multitask_weights: int = -1,
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
            learned_multitask_weights: If not -1, weigh the task losses with
                learned uncertainties (`LossWeightBalancing`), active from
                this epoch on.
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

        self.loss_weight_balancing: Optional[LossWeightBalancing] = None
        if learned_multitask_weights != -1:
            self.loss_weight_balancing = LossWeightBalancing(
                n_tasks=len(self._tasks),
                late_activation=learned_multitask_weights,
            )

    def compute_loss(
        self, preds: Tensor, data: List[Data], verbose: bool = False
    ) -> Tensor:
        """Compute and sum losses across tasks."""
        data_merged = {}
        target_labels_merged = list(set(self.target_labels))
        for label in target_labels_merged:
            data_merged[label] = torch.cat([d[label] for d in data], dim=0)
        for task in self._tasks:
            if task._loss_weight is not None:
                data_merged[task._loss_weight] = torch.cat(
                    [d[task._loss_weight] for d in data], dim=0
                )

        losses = [
            task.compute_loss(pred, data_merged)
            for task, pred in zip(self._tasks, preds)
        ]
        if self.loss_weight_balancing is not None:
            losses = self.loss_weight_balancing(losses, self.current_epoch)

        if not self.training:
            # Log the individual task losses during validation.
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

    def _optimizer_parameters(self) -> Union[Iterator, List[Dict]]:
        """Return the parameters passed to the optimizer.

        The loss-balancing uncertainties get their own parameter group
        with the learning rate scaled down by two orders of magnitude.
        """
        if self.loss_weight_balancing is None:
            return super()._optimizer_parameters()
        balancing = list(self.loss_weight_balancing.parameters())
        balancing_ids = {id(p) for p in balancing}
        others = [p for p in self.parameters() if id(p) not in balancing_ids]
        lr = self._optimizer_kwargs.get("lr", 1e-3)
        return [
            {"params": others},
            {"params": balancing, "lr": lr * 1e-2},
        ]

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
