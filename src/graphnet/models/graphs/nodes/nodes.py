"""Class(es) for building/connecting graphs."""

from typing import List, Tuple, Optional
from abc import abstractmethod

import torch
from torch_geometric.data import Data

from graphnet.utilities.decorators import final
from graphnet.models import Model
from graphnet.models.components.pool import _group_identical
from time import time
from graphnet.models.graphs.utils import (
    cluster_summarize_with_percentiles,
    identify_indices,
    lex_sort,
    pulse_template,
    spe_atwd_old,
    pulseseries_to_wf,
)
from copy import deepcopy
import numpy as np


class NodeDefinition(Model):  # pylint: disable=too-few-public-methods
    """Base class for graph building."""

    def __init__(
        self, input_feature_names: Optional[List[str]] = None
    ) -> None:
        """Construct `Detector`."""
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)
        if input_feature_names is not None:
            self.set_output_feature_names(
                input_feature_names=input_feature_names
            )

    @final
    def forward(self, x: torch.tensor) -> Tuple[Data, List[str]]:
        """Construct nodes from raw node features.

        Args:
            x: standardized node features with shape ´[num_pulses, d]´,
            where ´d´ is the number of node features.
            node_feature_names: list of names for each column in ´x´.

        Returns:
            graph: a graph without edges
            new_features_name: List of new feature names.
        """
        graph = self._construct_nodes(x=x)
        try:
            self._output_feature_names
        except AttributeError as e:
            self.error(
                f"""{self.__class__.__name__} was instantiated without
                       `input_feature_names` and it was not set prior to this
                       forward call. If you are using this class outside a
                       `GraphDefinition`, please instatiate
                       with `input_feature_names`."""
            )  # noqa
            raise e
        return graph, self._output_feature_names

    @property
    def nb_outputs(self) -> int:
        """Return number of output features.

        This the default, but may be overridden by specific inheriting classes.
        """
        return len(self._output_feature_names)

    @final
    def set_number_of_inputs(self, input_feature_names: List[str]) -> None:
        """Return number of inputs expected by node definition.

        Args:
            input_feature_names: name of each input feature column.
        """
        assert isinstance(input_feature_names, list)
        self.nb_inputs = len(input_feature_names)

    @final
    def set_output_feature_names(self, input_feature_names: List[str]) -> None:
        """Set output features names as a member variable.

        Args:
            input_feature_names: List of column names of the input to the
            node definition.
        """
        self._output_feature_names = self._define_output_feature_names(
            input_feature_names
        )

    @abstractmethod
    def _define_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        """Construct names of output columns.

        Args:
            input_feature_names: List of column names for the input data.

        Returns:
            A list of column names for each column in
            the node definition output.
        """

    @abstractmethod
    def _construct_nodes(self, x: torch.tensor) -> Tuple[Data, List[str]]:
        """Construct nodes from raw node features ´x´.

        Args:
            x: standardized node features with shape ´[num_pulses, d]´,
            where ´d´ is the number of node features.
            feature_names: List of names for reach column in `x`. Identical
            order of appearance. Length `d`.

        Returns:
            graph: graph without edges.
            new_node_features: A list of node features names.
        """


class NodesAsPulses(NodeDefinition):
    """Represent each measured pulse of Cherenkov Radiation as a node."""

    def _define_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        return input_feature_names

    def _construct_nodes(self, x: torch.Tensor) -> Tuple[Data, List[str]]:
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
        max_activations: Optional[int] = None,
    ) -> None:
        """Construct nodes as DOMs with time series of pulses.

        Args:
            keys: List of node feature names.
            id_columns: List of columns that uniquely identify a DOM.
            time_index: Name of the column that contains the time index.
            charge_index: Name of the column that contains the charge.
            max_activations: Maximum number of activations to keep per DOM.
        """
        assert isinstance(keys, type(id_columns))

        self._keys = keys
        self._id_columns = [self._keys.index(key) for key in id_columns]
        self._time_index = self._keys.index(time_index)
        self._charge_index = self._keys.index(charge_index)
        self._max_activations = max_activations
        super().__init__()

    def _define_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        return input_feature_names

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
        # cast to numpy
        x = x.numpy()
        # sort by time
        x = x[np.argsort(x[:, self._time_index])]
        # undo log10 scaling since we want to sum up charge
        x[:, self._charge_index] = np.power(10, x[:, self._charge_index])
        # shift time to positive values with a small offset
        x[:, self._time_index] += -np.min(x[:, self._time_index])
        # Group pulses on the same DOM
        x = lex_sort(x, self._id_columns)

        unique_sensors, counts = np.unique(
            x[:, self._id_columns], return_counts=True, axis=0
        )
        # sort DOMs and pulse-counts
        sort_this = np.concatenate(
            [unique_sensors, counts.reshape(-1, 1)], axis=1
        )
        sort_this = lex_sort(x=sort_this, cluster_columns=self._id_columns)
        unique_sensors = sort_this[:, 0 : unique_sensors.shape[1]]
        counts = sort_this[:, unique_sensors.shape[1] :].flatten().astype(int)

        time_series = np.split(
            x[:, [self._charge_index, self._time_index]], counts.cumsum()[:-1]
        )

        # add total charge to unique dom features and apply log10 scaling
        time_charge = np.stack(
            [
                (image[0, 1], np.arcsinh(5 * image[:, 0].sum()) / 5)
                for image in time_series
            ]
        )
        x = np.column_stack([unique_sensors, time_charge])

        # time_series, sort_ind = self._sort_by_n_pulses(time_series)
        # cutter = torch.tensor([len(ts) for ts in time_series])
        # x = x[sort_ind]
        if self._max_activations is not None:
            counts[counts > self._max_activations] = self._max_activations
            time_series = [
                image[: self._max_activations] for image in time_series
            ]
        time_series = np.concatenate(time_series)
        # apply inverse hyperbolic sine to charge values (handles zeros unlike log scaling)
        time_series[:, 0] = np.arcsinh(5 * time_series[:, 0]) / 5

        return Data(
            x=torch.tensor(x),
            time_series=torch.tensor(time_series),
            cutter=torch.tensor(counts),
            n_doms=len(x),
        )


@torch.jit.script
def log10powsum(tensor: torch.Tensor) -> torch.Tensor:
    """Convert to power of 10 and sum and convert back to log10."""
    return torch.log10(torch.sum(torch.pow(10, tensor)))


@torch.jit.script
def pad_charge(tensor: torch.Tensor, time_range: torch.Tensor) -> torch.Tensor:
    """Pad charge tensor to have same length as time range.

    Args:
        tensor: tensor of shape (num_pulses,2) with time and charge.

    Returns:
        tensor: padded charge tensor of shape (len(time_range),1).
    """
    padded = torch.ones(len(time_range)) * torch.tensor(-16.0)
    for val in time_range:
        if (tensor[:, 0] == val).any():
            padded[time_range == val] = tensor[tensor[:, 0] == val, 1]

    return padded


@torch.jit.script
def return_closest(tt: torch.Tensor, tr: torch.Tensor) -> torch.Tensor:
    """Return closest index of time range to time tensor.

    Args:
        tt: pulse time of shape (num_pulses,1).
        tr: time range of shape (len(time_range),1).

    Returns:
        tensor: binned time index of shape (num_pulses,1).
    """
    return torch.argmin(abs(tt - tr.unsqueeze(1)), dim=0)


@torch.jit.script
def sum_charge(
    x: torch.Tensor,
    dom_index: torch.Tensor,
    id_columns: List[int],
    time_index: int,
    charge_index: int,
) -> torch.Tensor:
    """Sum charge of pulses in the same time bin.

    Args:
        tensor: tensor of shape (num_pulses,2) with time and charge.

    Returns:
        tensor: padded charge tensor of shape (len(time_range),1).
    """
    x = torch.stack(
        [
            torch.hstack(
                [
                    x[dom_index == index][:, id_columns + [time_index]][0],
                    log10powsum(x[dom_index == index][:, charge_index]),
                ]
            )
            for index in torch.unique(dom_index)
        ]
    )
    return x


@torch.jit.script
def create_time_series(
    x: torch.Tensor,
    dom_index: torch.Tensor,
    id_columns: List[int],
    time_index: int,
    charge_index: int,
    time_range: torch.Tensor,
) -> List[torch.Tensor]:
    """Create time series.

    Args:
        tensor: padded charge tensor of shape (len(time_range),1).
        dom_index: indexing of doms for grouping.
        id_columns: list of columns that uniquely identify a DOM.
        time_index: index of time column.
        charge_index: index of charge column.
        time_range: time range to be used for padding.

    Returns:
        tensor: time series of shape (num_pulses,1).
    """
    time_series, unique_doms = [], []
    for index_key in torch.unique(dom_index):
        unique_doms.append(
            torch.hstack(
                [
                    x[dom_index == index_key][0][id_columns],
                    x[dom_index == index_key][0][time_index],
                    x[dom_index == index_key][:, charge_index].max(),
                ]
            )
        )
        time_series.append(
            pad_charge(
                x[dom_index == index_key][:, [time_index] + [charge_index]],
                time_range,
            )
        )
    unique_doms = torch.vstack(unique_doms)
    time_series = torch.vstack(time_series)
    return [unique_doms, time_series]


# @torch.jit.script
# def get_unique_dom_features(
#     x: torch.Tensor,
#     dom_index: torch.Tensor,
#     id_columns: List[int],
#     time_index: int,
#     charge_index: int,
#     time_range: torch.Tensor,
# ) -> torch.Tensor:
#     """Get unique dom features.

#     Args:
#         tensor: padded charge tensor of shape )"""
#     unique_doms = []
#     for index_key in torch.unique(dom_index):
#         unique_doms.append(
#             torch.hstack(
#                 [
#                     x[dom_index == index_key][0][id_columns],
#                     x[dom_index == index_key][0][time_index],
#                     x[dom_index == index_key][:, charge_index].max(),
#                 ]
#             )
#         )
#     unique_doms = torch.vstack(unique_doms)
#     return unique_doms


def create_sparse_charge_series(
    dom_index: torch.Tensor, time_index: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """Create sparse charge series.

    Args:
        dom_index: indexing of doms for grouping.
        time_index: indexing of time bins for grouping.
        values: values to be summed up.
    """
    i = torch.vstack([dom_index, time_index])
    v = values
    s = torch.sparse_coo_tensor(
        i, v, (dom_index.max() + 1, time_index.max() + 1)
    )
    s = s.coalesce()
    return s


# torch.jit.script
# def sort_by_n_pulses(time_series: List[torch.Tensor]
# ) -> list[torch.Tensor]:
#     """Sort time series by number of pulses."""
#     sort_index = (
#         torch.tensor([len(ts) for ts in time_series])
#         .sort(descending=True)
#         .indices
#     )
#     sorted_time_series = [time_series[i] for i in sort_index]
#     return sorted_time_series, sort_index


class NodesAsDOMTimeWindow(NodeDefinition):
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
        granularity: int = 100,
    ) -> None:
        """Construct nodes as DOMs with time series of pulses.

        Args:
            keys: List of node feature names.
            id_columns: List of columns that uniquely identify a DOM.
            time_index: Name of the column that contains the time index.
            charge_index: Name of the column that contains the charge.
            granularity: Number of time bins to use.
        """
        assert isinstance(keys, type(id_columns))

        self._keys = keys
        self._id_columns = [self._keys.index(key) for key in id_columns]
        self._time_index = self._keys.index(time_index)
        self._charge_index = self._keys.index(charge_index)
        self._granularity = granularity

        super().__init__()

    def _construct_nodes(self, x: torch.Tensor) -> Data:
        """Construct nodes from raw node features ´x´."""
        # sort by time
        x = x[x[:, self._time_index].sort().indices]
        # undo log10 scaling since we want to sum up charge
        x[:, self._charge_index] = torch.pow(10, x[:, self._charge_index])
        # shift time to positive values with a small offset
        x[:, self._time_index] += 0.1 - min(x[:, self._time_index])

        # create time range
        self._time_range = torch.logspace(
            torch.log2(min(x[:, self._time_index])),
            (torch.log2(max(x[:, self._time_index]))),
            self._granularity,
            base=2.0,
        )

        # group pulses on the same DOM
        dom_index = _group_identical(x[:, self._id_columns])

        # get unique dom features
        # unique_doms = get_unique_dom_features(x,dom_index,self._id_columns,self._time_index,self._charge_index,self._time_range)
        val, ind = dom_index.sort(stable=True)
        counts = torch.concat(
            [torch.tensor([0]), val.bincount().cumsum(-1)[:-1]]
        )
        unique_doms = x[:, self._id_columns + [self._time_index]][ind][counts]

        # get coarse time index
        coarse_time_index = return_closest(
            x[:, self._time_index], self._time_range
        )

        # Create torch sparse tensor summing up charge in the same time bin
        time_series = create_sparse_charge_series(
            dom_index, coarse_time_index, x[:, self._charge_index]
        )

        # add total charge to unique dom features
        unique_doms = torch.hstack(
            [
                unique_doms,
                torch._sparse_sum(time_series, dim=1).to_dense().unsqueeze(1),
            ]
        )
        # apply inverse hyperbolic sine to charge values (handles zeros unlike log scaling)

        unique_doms[:, -1] = torch.asinh(5 * unique_doms[:, -1]) / 5
        time_series = torch.asinh(5 * time_series) / 5
        # convert to dense tensor

        time_series = time_series.to_dense()

        # x[:,self._time_index] = self._time_range[return_closest(x[:,self._time_index],self._time_range)]

        # dom_index = _group_identical(x[:, self._id_columns+[self._time_index]])

        # x = sum_charge(x,dom_index,self._id_columns,self._time_index,self._charge_index)

        # dom_index = _group_identical(x[:, self._id_columns])

        # unique_doms, time_series = create_time_series(x,dom_index,self._id_columns,self._time_index,self._charge_index,self._time_range)

        return Data(x=unique_doms, time_series=time_series)


class PercentileClusters(NodeDefinition):
    """Represent nodes as clusters with percentile summary node features.

    If `cluster_on` is set to the xyz coordinates of DOMs
    e.g. `cluster_on = ['dom_x', 'dom_y', 'dom_z']`, each node will be a
    unique DOM and the pulse information (charge, time) is summarized using
    percentiles.
    """

    def __init__(
        self,
        cluster_on: List[str],
        percentiles: List[int],
        add_counts: bool = True,
        input_feature_names: Optional[List[str]] = None,
    ) -> None:
        """Construct `PercentileClusters`.

        Args:
            cluster_on: Names of features to create clusters from.
            percentiles: List of percentiles. E.g. `[10, 50, 90]`.
            add_counts: If True, number of duplicates is added to output array.
            input_feature_names: (Optional) column names for input features.
        """
        self._cluster_on = cluster_on
        self._percentiles = percentiles
        self._add_counts = add_counts
        # Base class constructor
        super().__init__(input_feature_names=input_feature_names)

    def _define_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        (
            cluster_idx,
            summ_idx,
            new_feature_names,
        ) = self._get_indices_and_feature_names(
            input_feature_names, self._add_counts
        )
        self._cluster_indices = cluster_idx
        self._summarization_indices = summ_idx
        return new_feature_names

    def _get_indices_and_feature_names(
        self,
        feature_names: List[str],
        add_counts: bool,
    ) -> Tuple[List[int], List[int], List[str]]:
        cluster_idx, summ_idx, summ_names = identify_indices(
            feature_names, self._cluster_on
        )
        new_feature_names = deepcopy(self._cluster_on)
        for feature in summ_names:
            for pct in self._percentiles:
                new_feature_names.append(f"{feature}_pct{pct}")
        if add_counts:
            # add "counts" as the last feature
            new_feature_names.append("counts")
        return cluster_idx, summ_idx, new_feature_names

    def _construct_nodes(self, x: torch.Tensor) -> Data:
        # Cast to Numpy
        x = x.numpy()
        # Construct clusters with percentile-summarized features
        if hasattr(self, "_summarization_indices"):
            array = cluster_summarize_with_percentiles(
                x=x,
                summarization_indices=self._summarization_indices,
                cluster_indices=self._cluster_indices,
                percentiles=self._percentiles,
                add_counts=self._add_counts,
            )
        else:
            self.error(
                f"""{self.__class__.__name__} was not instatiated with
                `input_feature_names` and has not been set later.
                Please instantiate this class with `input_feature_names`
                if you're using it outside `GraphDefinition`."""
            )  # noqa
            raise AttributeError

        return Data(x=torch.tensor(array))


class NodesAsRefoldedDOM(NodeDefinition):
    """Represent each node as DOM a with a time series of pulses."""

    def __init__(
        self,
        keys: List[str] = [
            "dom_x",
            "dom_y",
            "dom_z",
            "dom_time",
            "charge",
            "atwd",
            "width",
        ],
        id_columns: List[str] = ["dom_x", "dom_y", "dom_z"],
        time_index: str = "dom_time",
        charge_index: str = "charge",
        atwd_index: str = "atwd",
        width_index: str = "width",
        max_activations: Optional[int] = None,
    ) -> None:
        """Construct nodes as DOMs with time series of pulses.

        Args:
            keys: List of node feature names.
            id_columns: List of columns that uniquely identify a DOM.
            time_index: Name of the column that contains the time index.
            charge_index: Name of the column that contains the charge.
            atwd_index: Name of the column that contains the atwd.
            width_index: Name of the column that contains the width.
            max_activations: Maximum number of activations to keep per DOM.
        """
        assert isinstance(keys, type(id_columns))

        self._keys = keys
        self._id_columns = [self._keys.index(key) for key in id_columns]
        self._time_index = self._keys.index(time_index)
        self._charge_index = self._keys.index(charge_index)
        self._atwd_index = self._keys.index(atwd_index)
        self._width_index = self._keys.index(width_index)
        self._max_activations = max_activations
        super().__init__()

    def _define_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        return input_feature_names + ["n_pulses"]

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
        # cast to numpy
        x = x.numpy()
        # sort by time
        x = x[np.argsort(x[:, self._time_index])]
        # undo log10 scaling since we want to sum up charge
        # x[:, self._charge_index] = np.power(10, x[:, self._charge_index])
        # shift time to positive values with a small offset
        x[:, self._time_index] += -np.min(x[:, self._time_index])
        # find largest charge bin in time
        _counts, edges_ = np.histogram(
            x[:, self._time_index], bins=1000, weights=x[:, self._charge_index]
        )
        largest_charge_time = (
            0.5 * (edges_[:-1] + edges_[1:])[np.argmax(_counts)]
        )

        # weigthed_mean_time = np.average(x[:, self._time_index],weights=x[:, self._charge_index])
        # Group pulses on the same DOM
        x = lex_sort(x, self._id_columns)

        unique_sensors, counts = np.unique(
            x[:, self._id_columns], return_counts=True, axis=0
        )
        # sort DOMs and pulse-counts
        sort_this = np.concatenate(
            [unique_sensors, counts.reshape(-1, 1)], axis=1
        )
        sort_this = lex_sort(x=sort_this, cluster_columns=self._id_columns)
        unique_sensors = sort_this[:, 0 : unique_sensors.shape[1]]
        counts = sort_this[:, unique_sensors.shape[1] :].flatten().astype(int)

        time_series = np.split(
            x[
                :,
                [
                    self._charge_index,
                    self._time_index,
                    self._atwd_index,
                    self._width_index,
                ],
            ],
            counts.cumsum()[:-1],
        )

        # add total charge to unique dom features and apply log10 scaling
        time_charge_n = np.stack(
            [
                (image[0, 1], np.log10(image[:, 0].sum()), len(image))
                for image in time_series
            ]
        )
        x = np.column_stack([unique_sensors, time_charge_n])

        # perform refolding
        ts = []
        time_range = np.linspace(
            largest_charge_time - 1600, largest_charge_time + 3200, 1000
        )
        for ind, time_series in enumerate(time_series):
            charge_series = pulseseries_to_wf(
                time_series, spe_atwd_old, time_range, 1
            ).astype(np.float16)
            ts.append(charge_series)
        ts = np.array(ts)
        # time_series = np.concatenate(time_series)
        x = np.column_stack([x, ts])

        return Data(
            x=torch.tensor(x),
            time_series=torch.tensor(ts),
            time_range=time_range,
            cutter=torch.tensor(counts),
            n_doms=len(x),
        )
