"""Class(es) for building/connecting graphs."""

from typing import List, Optional
from abc import abstractmethod

import torch
from torch_geometric.data import Data

from graphnet.utilities.decorators import final
from graphnet.utilities.config import save_model_config
from graphnet.models import Model
from graphnet.models.components.pool import _group_identical


class NodeDefinition(Model):  # pylint: disable=too-few-public-methods
    """Base class for graph building."""

    @save_model_config
    def __init__(self) -> None:
        """Construct `Detector`."""
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

    @final
    def forward(self, x: torch.tensor) -> Data:
        """Construct nodes from raw node features.

        Args:
            x: standardized node features with shape ´[num_pulses, d]´,
            where ´d´ is the number of node features.

        Returns:
            graph: a graph without edges
        """
        graph = self._construct_nodes(x)
        return graph

    @property
    def nb_outputs(self) -> int:
        """Return number of output features.

        This the default, but may be overridden by specific inheriting classes.
        """
        return self.nb_inputs

    @final
    def set_number_of_inputs(self, node_feature_names: List[str]) -> None:
        """Return number of inputs expected by node definition.

        Args:
            node_feature_names: name of each node feature column.
        """
        assert isinstance(node_feature_names, list)
        self.nb_inputs = len(node_feature_names)

    @abstractmethod
    def _construct_nodes(self, x: torch.tensor) -> Data:
        """Construct nodes from raw node features ´x´.

        Args:
            x: standardized node features with shape ´[num_pulses, d]´,
            where ´d´ is the number of node features.

        Returns:
            graph: graph without edges.
        """


class NodesAsPulses(NodeDefinition):
    """Represent each measured pulse of Cherenkov Radiation as a node."""

    def _construct_nodes(self, x: torch.Tensor) -> Data:
        return Data(x=x)


class NodesAsDOMTimeSeries(NodeDefinition):
    """Represent each node as DOM a with a time series of pulses."""

    def __init__(
        self,
        keys: List[str] = [
            "dom_x",
            "dom_y",
            "dom_z",
            "dom_time",
            "charge",
        ],
        id_columns: List[str] = ["dom_x", "dom_y", "dom_z"],
        time_index: str = "dom_time",
        charge_index: str = "charge",
    ) -> None:
        """Construct nodes as DOMs with time series of pulses.

        Args:
            keys: List of node feature names.
            id_columns: List of columns that uniquely identify a DOM.
            time_index: Name of the column that contains the time index.
            charge_index: Name of the column that contains the charge.
        """
        assert isinstance(keys, type(id_columns))

        self._keys = keys
        self._id_columns = [self._keys.index(key) for key in id_columns]
        self._time_index = self._keys.index(time_index)
        self._charge_index = self._keys.index(charge_index)

        super().__init__()

    def _sort_by_n_pulses(
        self, time_series: List[torch.Tensor]
    ) -> torch.Tensor:
        """Sort time series by number of pulses."""
        sort_index = (
            torch.tensor([len(ts) for ts in time_series])
            .sort(descending=True)
            .indices
        )
        sorted_time_series = [time_series[i] for i in sort_index]
        return sorted_time_series, sort_index

    def _construct_nodes(self, x: torch.Tensor) -> Data:
        """Construct nodes from raw node features ´x´."""
        # sort by time
        x = x[x[:, self._time_index].sort().indices]
        # Group pulses on the same DOM
        dom_index = _group_identical(x[:, self._id_columns])
        time_series = [
            x[dom_index == index_key] for index_key in dom_index.unique()
        ]
        x = torch.stack(
            [image[:, self._id_columns].mean(axis=0) for image in time_series]
        )
        time = torch.stack(
            [
                sum(image[:, self._time_index] * image[:, self._charge_index])
                for image in time_series
            ]
        )
        charge = torch.stack(
            [image[:, self._charge_index].sum() for image in time_series]
        )
        time = time / charge
        x = torch.column_stack([x, time, charge])
        time_series, sort_ind = self._sort_by_n_pulses(time_series)
        cutter = torch.tensor([len(ts) for ts in time_series])
        x = x[sort_ind]
        time_series = torch.concat(time_series)
        return Data(x=x, time_series=time_series, cutter=cutter, n_doms=len(x))
