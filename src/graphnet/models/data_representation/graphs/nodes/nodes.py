"""Class(es) for building/connecting graphs."""

from typing import List, Tuple, Optional, Dict, Union
from abc import abstractmethod

import torch
from torch_geometric.data import Data

from graphnet.utilities.decorators import final
from graphnet.models import Model
from graphnet.models.data_representation.graphs.utils import (
    cluster_and_pad,
    identify_indices,
    lex_sort,
    ice_transparency,
)
from copy import deepcopy

from time import time

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
    def forward(self, x: torch.tensor) -> torch.tensor:
        """Construct nodes from raw node features.

        Args:
            x: standardized node features with shape ´[num_pulses, d]´,
            where ´d´ is the number of node features.
            node_feature_names: list of names for each column in ´x´.

        Returns:
            graph: a graph without edges
        """
        data = self._construct_nodes(x=x)

        return data

    @property
    def _output_feature_names(self) -> List[str]:
        """Return output feature names."""
        try:
            self._hidden_output_feature_names
        except AttributeError as e:
            self.error(
                f"""{self.__class__.__name__} was instantiated without
                       `input_feature_names` and it was not set prior to this
                       forward call. If you are using this class outside a
                       `GraphDefinition`, please instatiate
                       with `input_feature_names`."""
            )  # noqa
            raise e
        return self._hidden_output_feature_names

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
        self._hidden_output_feature_names = self._define_output_feature_names(
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
    def _construct_nodes(self, x: torch.tensor) -> torch.tensor:
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

    def _construct_nodes(self, x: torch.Tensor) -> torch.Tensor:
        return x


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
        ) = self._get_indices_and_feature_names(input_feature_names)
        self._cluster_indices = cluster_idx
        self._summarization_indices = summ_idx
        return new_feature_names

    def _get_indices_and_feature_names(
        self,
        feature_names: List[str],
    ) -> Tuple[List[int], List[int], List[str]]:
        cluster_idx, summ_idx, summ_names = identify_indices(
            feature_names, self._cluster_on
        )
        new_feature_names = deepcopy(self._cluster_on)
        for feature in summ_names:
            for pct in self._percentiles:
                new_feature_names.append(f"{feature}_pct{pct}")
        if self._add_counts:
            # add "counts" as the last feature
            new_feature_names.append("counts")
        return cluster_idx, summ_idx, new_feature_names

    def _construct_nodes(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to Numpy
        x = x.numpy()
        # Construct clusters with percentile-summarized features
        if hasattr(self, "_summarization_indices"):
            cluster_class = cluster_and_pad(
                x=x, cluster_columns=self._cluster_indices
            )
            cluster_class.add_percentile_summary(
                summarization_indices=self._summarization_indices,
                percentiles=self._percentiles,
            )
            if self._add_counts:
                cluster_class.add_counts()
            array = cluster_class.clustered_x
        else:
            self.error(
                f"""{self.__class__.__name__} was not instatiated with
                `input_feature_names` and has not been set later.
                Please instantiate this class with `input_feature_names`
                if you're using it outside `GraphDefinition`."""
            )  # noqa
            raise AttributeError

        return torch.tensor(array)


class NodeAsDOMTimeSeries(NodeDefinition):
    """Represent each node as a DOM with time and charge time series data."""

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
        time_column: str = "dom_time",
        charge_column: str = "charge",
        max_activations: Optional[int] = None,
    ) -> None:
        """Construct `NodeAsDOMTimeSeries`.

        Args:
            keys: Names of features in the data (in order).
            id_columns: List of columns that uniquely identify a DOM.
            time_column: Name of time column.
            charge_column: Name of charge column.
            max_activations: Maximum number of activations to include in
                the time series.
        """
        self._keys = keys
        super().__init__(input_feature_names=self._keys)
        self._id_columns = [self._keys.index(key) for key in id_columns]
        self._time_index = self._keys.index(time_column)
        try:
            self._charge_index: Optional[int] = self._keys.index(charge_column)
        except ValueError:
            self.warning(
                "Charge column with name {charge_column} not found. "
                "Running without."
            )

            self._charge_index = None

        self._max_activations = max_activations

    def _define_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        return input_feature_names + ["new_node_col"]

    def _construct_nodes(self, x: torch.Tensor) -> torch.Tensor:
        """Construct nodes from raw node features ´x´."""
        # Cast to Numpy
        x = x.numpy()
        if x.shape[0] == 0:
            return Data(x=torch.tensor(np.column_stack([x, []])))
        # if there is no charge column add a dummy column
        # of zeros with the same shape as the time column
        if self._charge_index is None:
            charge_index: int = len(self._keys)
            x = np.insert(x, charge_index, np.zeros(x.shape[0]), axis=1)
        else:
            charge_index = self._charge_index

        # Sort by time
        x = x[x[:, self._time_index].argsort()]
        # Undo log10 scaling so we can sum charges
        x[:, charge_index] = np.power(10, x[:, charge_index])
        # Shift time to start at 0
        x[:, self._time_index] -= np.min(x[:, self._time_index])
        # Group pulses on the same DOM
        x = lex_sort(x, self._id_columns)

        unique_sensors, counts = np.unique(
            x[:, self._id_columns], axis=0, return_counts=True
        )

        sort_this = np.concatenate(
            [unique_sensors, counts.reshape(-1, 1)], axis=1
        )
        sort_this = lex_sort(x=sort_this, cluster_columns=self._id_columns)
        unique_sensors = sort_this[:, 0 : unique_sensors.shape[1]]
        counts = sort_this[:, unique_sensors.shape[1] :].flatten().astype(int)

        new_node_col = np.zeros(x.shape[0])
        new_node_col[counts.cumsum()[:-1]] = 1
        new_node_col[0] = 1
        x = np.column_stack([x, new_node_col])

        return Data(x=torch.tensor(x))


class NodeAsDOMSummary(NodeDefinition):
    """Represent each node as a DOM with summary features."""

    def __init__(
        self,
        input_feature_names: List[str] = [
            "dom_x",
            "dom_y",
            "dom_z",
            "dom_time",
            "charge",
        ],
        id_columns: List[str] = ["dom_x", "dom_y", "dom_z"],
        time_column: str = "dom_time",
        charge_column: str = "charge",
        percentiles: List[int] = [
            0.5,
            1.0,
            2.5,
            5.0,
            10.0,
            25,
            50.0,
            75.0,
            100.0,
        ],
        summaries: List[str] = [
            "sum",
            "mean",
            "std",
            "count",
            "time_percentile",
            "charge_percentile",
            "time_charge_threshold",
        ],
        closest_to_time: Optional[List[float]] = None,
    ) -> None:
        """Construct `NodeAsDOMSummary`.

        Args:
            input_feature_names: List of column names for input features.
            id_columns: List of columns that uniquely identify a DOM.
            time_column: Name of time column.
            charge_column: Name of charge column.
            percentiles: List of percentiles to calculate.
            summaries: List of summaries to calculate.
        """
        self._input_feature_names = input_feature_names

        self._id_index = [
            self._input_feature_names.index(key) for key in id_columns
        ]
        self._charge_index = self._input_feature_names.index(charge_column)
        self._time_index = self._input_feature_names.index(time_column)
        self._time_column = time_column
        self._charge_column = charge_column
        self._percentiles = percentiles
        self._summaries = summaries
        self._closest_to_time = closest_to_time
        assert all(
            [
                summary
                in [
                    "sum",
                    "mean",
                    "std",
                    "count",
                    "time_percentile",
                    "charge_percentile",
                    "time_charge_threshold",
                ]
                for summary in self._summaries
            ]
        ), f"One or more of the summaries is not recognized. The following summaries are recognized: ['sum','mean','std','count','time_percentile','charge_percentile','time_charge_threshold']"
        super().__init__(input_feature_names=self._input_feature_names)

    def _construct_nodes(self, x: torch.Tensor) -> Data:
        cluster_class = cluster_and_pad(
            x=x.numpy(), cluster_columns=self._id_index
        )
        # add first activations of everything except the id_index columns
        # which are already added
        cluster_class.add_first(
            columns=[
                i
                for i in range(len(self._input_feature_names))
                if i not in self._id_index
            ]
        )

        charge_scale_indices = [self._charge_index]
        time_scale_indices = [self._time_index]

        if "sum" in self._summaries:
            cluster_class.add_sum_charge(charge_index=self._charge_index)
            charge_scale_indices.append(len(cluster_class.clustered_x[0]) - 1)
        else:
            cluster_class._calculate_charge_sum(
                charge_index=self._charge_index
            )

        cluster_class._calculate_charge_weights(
            charge_index=self._charge_index
        )

        if "mean" in self._summaries:
            # calculate the weighted time mean
            cluster_class.add_mean(
                columns=[self._time_index],
                weights=cluster_class._charge_weights,
            )
            time_scale_indices.append(len(cluster_class.clustered_x[0]) - 1)

        if "std" in self._summaries:
            # calculate the weighted time std
            cluster_class.add_std(
                columns=[self._time_index],
                weights=cluster_class._charge_weights,
            )
            time_scale_indices.append(len(cluster_class.clustered_x[0]) - 1)

        if "count" in self._summaries:
            cluster_class.add_counts()

        if "time_percentile" in self._summaries:
            cluster_class.add_percentile_summary(
                summarization_indices=[self._time_index],
                percentiles=self._percentiles,
            )
            time_scale_indices = time_scale_indices + list(
                range(
                    len(cluster_class.clustered_x[0]) - len(self._percentiles),
                    len(cluster_class.clustered_x[0]),
                )
            )
        if "charge_percentile" in self._summaries:
            cluster_class.add_percentile_summary(
                summarization_indices=[self._charge_index],
                percentiles=self._percentiles,
            )
            charge_scale_indices = charge_scale_indices + list(
                range(
                    len(cluster_class.clustered_x[0]) - len(self._percentiles),
                    len(cluster_class.clustered_x[0]),
                )
            )

        if "time_charge_threshold" in self._summaries:
            cluster_class.add_charge_threshold_summary(
                summarization_indices=[self._time_index],
                charge_index=self._charge_index,
                percentiles=self._percentiles,
            )
            time_scale_indices = time_scale_indices + list(
                range(
                    len(cluster_class.clustered_x[0]) - len(self._percentiles),
                    len(cluster_class.clustered_x[0]),
                )
            )

        array = cluster_class.clustered_x

        # log10 scale the charge columns
        array[:, charge_scale_indices] = np.log10(
            array[:, charge_scale_indices]
        )
        # scale the time columns
        array[:, time_scale_indices] = (
            array[:, time_scale_indices] - 1e4
        ) / 3e4
        return Data(x=torch.tensor(array))

    def _define_output_feature_names(self, input_feature_names):
        new_feature_names = deepcopy(input_feature_names)
        if "sum" in self._summaries:
            new_feature_names.append("sum_charge")
        if "mean" in self._summaries:
            new_feature_names.append("mean_time")
        if "std" in self._summaries:
            new_feature_names.append("std_time")
        if "count" in self._summaries:
            new_feature_names.append("counts")
        if "time_percentile" in self._summaries:
            for pct in self._percentiles:
                new_feature_names.append(f"time_pct{pct}")
        if "charge_percentile" in self._summaries:
            for pct in self._percentiles:
                new_feature_names.append(f"charge_pct{pct}")
        if "time_charge_threshold" in self._summaries:
            for pct in self._percentiles:
                new_feature_names.append(f"time_charge_threshold_{pct}")
        return new_feature_names


class IceMixNodes(NodeDefinition):
    """Calculate ice properties and perform random sampling.

    Ice properties are calculated based on the z-coordinate of the pulse. For
    each event, a random sampling is performed to keep the number of pulses
    below a maximum number of pulses if n_pulses is over the limit.
    """

    def __init__(
        self,
        input_feature_names: Optional[List[str]] = None,
        max_pulses: int = 768,
        z_name: str = "dom_z",
        hlc_name: Optional[str] = "hlc",
        add_ice_properties: bool = True,
        ice_args: Dict[str, Optional[float]] = {
            "z_offset": None,
            "z_scaling": None,
        },
        sample_pulses: bool = True,
    ) -> None:
        """Construct `IceMixNodes`.

        Args:
            input_feature_names: Column names for input features. Minimum
            required features are z coordinate and hlc column names.
            max_pulses: Maximum number of pulses to keep in the event.
            z_name: Name of the z-coordinate column.
            hlc_name: Name of the `Hard Local Coincidence Check` column.
            add_ice_properties: If True, scattering and absoption length of
            ice in IceCube are added to the feature set based on z coordinate.
            ice_args: Offset and scaling of the z coordinate in the Detector,
            to be able to make similar conversion in the ice data.
            sample_pulses: Enable sampling random pulses. If True and the
            event is longer than the max_length, they will be sampled. If
            False, then only the first max_length pulses will be selected.
        """
        if input_feature_names is None:
            input_feature_names = [
                "dom_x",
                "dom_y",
                "dom_z",
                "dom_time",
                "charge",
                "hlc",
                "rde",
            ]

        if add_ice_properties:
            if z_name not in input_feature_names:
                raise ValueError(
                    f"z name '{z_name}' not found in "
                    f"input_feature_names {input_feature_names}"
                )
            self.all_features = input_feature_names + [
                "scatt_lenght",
                "abs_lenght",
            ]
            self.f_scattering, self.f_absoprtion = ice_transparency(**ice_args)
        else:
            self.all_features = input_feature_names

        super().__init__(input_feature_names=input_feature_names)

        if hlc_name not in input_feature_names:
            self.warning(
                f"hlc name '{hlc_name}' not found in input_feature_names"
                f" '{input_feature_names}', subsampling will be random."
            )
            hlc_name = None

        self.feature_indexes = {
            feat: self.all_features.index(feat) for feat in input_feature_names
        }

        self.input_feature_names = input_feature_names
        self.n_features = len(self.all_features)
        self.max_length = max_pulses
        self.z_name = z_name
        self.hlc_name = hlc_name
        self.add_ice_properties = add_ice_properties
        self.sampling_enabled = sample_pulses

    def _define_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        return self.all_features

    def _add_ice_properties(
        self, graph: torch.Tensor, x: torch.Tensor, ids: List[int]
    ) -> torch.Tensor:

        graph[: len(ids), -2] = torch.tensor(
            self.f_scattering(x[ids, self.feature_indexes[self.z_name]])
        )
        graph[: len(ids), -1] = torch.tensor(
            self.f_absoprtion(x[ids, self.feature_indexes[self.z_name]])
        )
        return graph

    def _pulse_sampler(
        self, x: torch.Tensor, event_length: int
    ) -> torch.Tensor:

        if event_length < self.max_length:
            ids = torch.arange(event_length)
        else:
            ids = torch.randperm(event_length)
            if self.hlc_name is not None:
                auxiliary_n = torch.nonzero(
                    x[:, self.feature_indexes[self.hlc_name]] == 0
                ).squeeze(1)
                auxiliary_p = torch.nonzero(
                    x[:, self.feature_indexes[self.hlc_name]] == 1
                ).squeeze(1)
                ids_n = ids[auxiliary_n][
                    : min(self.max_length, len(auxiliary_n))
                ]
                ids_p = ids[auxiliary_p][
                    : min(self.max_length - len(ids_n), len(auxiliary_p))
                ]

                ids = torch.cat([ids_n, ids_p]).sort().values
            else:
                ids = ids[: self.max_length]

        return ids

    def _construct_nodes(self, x: torch.Tensor) -> torch.Tensor:

        event_length = x.shape[0]
        if self.hlc_name is not None:
            x[:, self.feature_indexes[self.hlc_name]] = torch.logical_not(
                x[:, self.feature_indexes[self.hlc_name]]
            )  # hlc in kaggle was flipped
        if self.sampling_enabled:
            ids = self._pulse_sampler(x, event_length)
        else:
            if event_length < self.max_length:
                ids = torch.arange(event_length)
            else:
                ids = torch.arange(self.max_length)

        event_length = min(self.max_length, event_length)

        graph = torch.zeros([event_length, self.n_features])

        if self.add_ice_properties:
            graph = self._add_ice_properties(graph, x, ids)
            non_ice_features = self.all_features[: self.n_features - 2]
        else:
            non_ice_features = self.all_features

        for idx, feature in enumerate(non_ice_features):
            graph[:event_length, idx] = x[ids, self.feature_indexes[feature]]

        return graph


class ClusterSummaryFeatures(NodeDefinition):
    """Represent pulse maps as clusters with summary features.

    If `cluster_on` is set to the xyz coordinates of optical modules
    e.g. `cluster_on = ['dom_x', 'dom_y', 'dom_z']`, each node will be
    a unique optical module and the pulse information (e.g. charge, time)
    is summarized.
    NOTE: Developed to be used with features
        [dom_x, dom_y, dom_z, charge, time]

    Possible features per cluster:
    - total charge
        feature name: `total_charge`
    - charge accumulated after <X> time units
        feature name: `charge_after_<X>ns`
    - time of first hit in the optical module
        feature name: `time_of_first_hit`
    - time spread per optical module
        feature name: `time_spread`
    - time std per optical module
        feature name: `time_std`
    - time took to collect <X> percent of total charge per cluster
        feature name: `time_after_charge_pct<X>`
    - number of pulses per clusters
        feature name: `counts`

    For more details on some of the features see
    Theo Glauchs thesis (chapter 5.3):
    https://mediatum.ub.tum.de/node?id=1584755
    """

    def __init__(
        self,
        cluster_on: List[str],
        input_feature_names: List[str],
        charge_label: str = "charge",
        time_label: str = "dom_time",
        total_charge: bool = True,
        total_charge_fraction: bool = False,
        charge_after_t: List[int] = [10, 50, 100],
        time_of_first_hit: bool = True,
        time_spread: bool = True,
        time_std: bool = True,
        time_after_charge_pct: List[int] = [1, 3, 5, 11, 15, 20, 50, 80],
        charge_standardization: Union[float, str] = "log",
        time_standardization: float = 1e-3,
        order_in_time: bool = True,
        add_counts: bool = False,
        charge_weighted: bool = False,
        node_limit: Optional[int] = None,
        node_limit_index: Optional[int] = None,
        node_limit_seed: Optional[int] = None,
        node_limit_ascending: bool = False,
    ) -> None:
        """Construct `ClusterSummaryFeatures`.

        Args:
            cluster_on: Names of features to create clusters from.
            input_feature_names: Column names for input features.
            charge_label: Name of the charge column.
            time_label: Name of the time column.
            total_charge: If True, calculates total charge as feature.
            total_charge_fraction: If True, calculates total charge fraction
                (cluster charge / event charge) as feature.
            charge_after_t: List of times at which the accumulated charge
                is calculated as a feature.
            time_of_first_hit: If True, time of first hit is added
                as a feature.
            time_spread: If True, time spread is added as a feature.
            time_std: If True, time std is added as a feature.
            time_after_charge_pct: List of percentiles to calculate time after
                charge.
            charge_standardization: Either a float or 'log'. If a float,
                the features are multiplied by this factor. If 'log', the
                features are transformed to log10 scale.
            time_standardization: Standardization factor for features
                with a time
            order_in_time: If True, clusters are ordered in time.
                    If your data is already ordered in time, you can set this
                    to False to avoid a potential overhead.
                NOTE: Should only be set to False if you are sure that
                    the input data is already ordered in time. Will lead to
                    incorrect results otherwise.
            add_counts: If True, number of log10(event counts per clusters)
                is added as a feature.
            charge_weights: If True, the mean and std of the charge
                distribution is added as a feature.
            node_limit: If set, limits the number of nodes to this number.
            node_limit_index: Index of the feature to sort on when limiting
                the number of nodes.
            node_limit_seed: Seed for random node limiting.

        NOTE: Make sure that either the input data is not already standardized
        or that the `charge_standardization` and `time_standardization`
        parameters are set to 1 to avoid a double standardization.
        """
        # Set member variables
        self._cluster_on = cluster_on
        self._charge_label = charge_label
        self._time_label = time_label
        self._order_in_time = order_in_time

        # Check if charge_standardization is a float or 'log'
        self._charge_standardization = charge_standardization
        self._time_standardization = time_standardization
        self._verify_standardization()

        # feature member variables
        self._total_charge = total_charge
        self._total_charge_fraction = total_charge_fraction
        self._charge_after_t = charge_after_t
        self._time_of_first_hit = time_of_first_hit
        self._time_spread = time_spread
        self._time_std = time_std
        self._time_after_charge_pct = time_after_charge_pct
        self._add_counts = add_counts
        self._charge_weighted = charge_weighted
        self._node_limit = node_limit
        self._node_limit_index = node_limit_index
        self._node_limit_seed = node_limit_seed
        self._node_limit_ascending = node_limit_ascending
        # Base class constructor
        super().__init__(input_feature_names=input_feature_names)
        if self._order_in_time is False:
            self.info(
                "Setting `order_by_time` to False. "
                "Make sure that the input data is already ordered in time."
            )

    def _define_output_feature_names(
        self,
        input_feature_names: List[str],
    ) -> List[str]:
        """Set the output feature names."""
        self.set_indices(input_feature_names)
        new_feature_names = deepcopy(self._cluster_on)
        if self._total_charge:
            new_feature_names.append("total_charge")
        if self._total_charge_fraction:
            new_feature_names.append("total_charge_fraction")
        for t in self._charge_after_t:
            new_feature_names.append(f"charge_after_{t}ns")
        if self._time_of_first_hit:
            new_feature_names.append("time_of_first_hit")
        if self._time_spread:
            new_feature_names.append("time_spread")
        if self._time_std:
            new_feature_names.append("time_std")
        for pct in self._time_after_charge_pct:
            new_feature_names.append(f"time_after_charge_pct{pct}")
        if self._add_counts:
            new_feature_names.append("counts")
        return new_feature_names

    def _construct_nodes(self, x: torch.Tensor) -> Data:
        """Construct nodes from raw node features ´x´."""
        # Cast to Numpy
        x = x.numpy()
        # Shift time to start at 0
        if self._time_idx is not None:
            x[:, self._time_idx] -= np.min(x[:, self._time_idx])
        # Construct clusters with percentile-summarized features
        cluster_class = cluster_and_pad(
            x=x,
            cluster_columns=self._cluster_idx,
            sort_by=[self._time_idx] if self._order_in_time else [],
        )
        # calculate charge weighted median time as reference
        ref_time = cluster_class.reference_time(
            charge_index=self._charge_idx,
            time_index=self._time_idx,
        )

        # add total charge
        if self._total_charge:
            cluster_class.add_sum_charge(charge_index=self._charge_idx)
            cluster_class.clustered_x[:, -1] = self._standardize_features(
                cluster_class.clustered_x[:, -1],
                self._charge_standardization,
            )

        if self._total_charge_fraction:
            event_total_charge = np.nansum(x[:, self._charge_idx])
            cluster_class.add_sum_charge(
                charge_index=self._charge_idx, total_charge=event_total_charge
            )
            cluster_class.clustered_x[:, -1] = self._standardize_features(
                cluster_class.clustered_x[:, -1],
                self._charge_standardization,
            )

        # add charge after t
        if len(self._charge_after_t) > 0:
            cluster_class.add_accumulated_value_after_t(
                time_index=self._time_idx,
                summarization_indices=[self._charge_idx],
                times=self._charge_after_t,
            )
            cluster_class.clustered_x[:, -len(self._charge_after_t) :] = (
                self._standardize_features(
                    cluster_class.clustered_x[:, -len(self._charge_after_t) :],
                    self._charge_standardization,
                )
            )

        # add time of first hit
        if self._time_of_first_hit:
            cluster_class.add_time_first_pulse(
                time_index=self._time_idx,
            )
            cluster_class.clustered_x[:, -1] -= ref_time

            cluster_class.clustered_x[:, -1] = self._standardize_features(
                cluster_class.clustered_x[:, -1],
                self._time_standardization,
            )

        # add time spread
        if self._time_spread:
            cluster_class.add_spread(
                columns=[self._time_idx],
            )
            cluster_class.clustered_x[:, -1] = self._standardize_features(
                cluster_class.clustered_x[:, -1],
                self._time_standardization,
            )

        # add time std
        if self._time_std:
            cluster_class.add_std(
                columns=[self._time_idx],
                weights=(
                    cluster_class._charge_weights
                    if self._charge_weighted
                    else 1
                ),
            )
            cluster_class.clustered_x[:, -1] = self._standardize_features(
                cluster_class.clustered_x[:, -1],
                self._time_standardization,
            )

        # add time after charge percentiles
        if len(self._time_after_charge_pct) > 0:
            cluster_class.add_charge_threshold_summary(
                summarization_indices=[self._time_idx],
                percentiles=self._time_after_charge_pct,
                charge_index=self._charge_idx,
            )
            cluster_class.clustered_x[
                :, -len(self._time_after_charge_pct) :
            ] -= ref_time
            cluster_class.clustered_x[
                :, -len(self._time_after_charge_pct) :
            ] = self._standardize_features(
                cluster_class.clustered_x[
                    :, -len(self._time_after_charge_pct) :
                ],
                self._time_standardization,
            )

        if self._add_counts:
            cluster_class.add_counts()

        if self._node_limit is not None:
            cluster_class.limit_number_of_clusters(
                max_clusters=self._node_limit,
                sort_index=self._node_limit_index,
                ascending=self._node_limit_ascending,
                seed=self._node_limit_seed,
            )  # the user is responsible for knowing which index to sort by

        return torch.tensor(cluster_class.clustered_x)

    def set_indices(self, feature_names: List[str]) -> None:
        """Set the indices for the input features."""
        self._cluster_idx = [
            feature_names.index(column) for column in self._cluster_on
        ]
        self._charge_idx = feature_names.index(self._charge_label)
        self._time_idx = feature_names.index(self._time_label)

    def _standardize_features(
        self,
        x: np.ndarray,
        standardization: Union[float, str],
    ) -> np.ndarray:
        """Standardize the features in the input array."""
        if isinstance(standardization, float):
            return x * standardization
        elif standardization == "log":
            return np.log10(x)
        else:
            # should never happen, but just in case
            raise ValueError(
                f"standardization must be either a float or 'log', "
                f"but got {standardization}"
            )

    def _verify_standardization(
        self,
    ) -> torch.Tensor:
        """Verify settings of standardization of the features."""
        if not isinstance(self._charge_standardization, float):
            if isinstance(self._charge_standardization, str):
                if self._charge_standardization != "log":
                    raise ValueError(
                        f"charge_standardization must be either a float or"
                        f" 'log', but got {self._charge_standardization}"
                    )
            else:
                raise ValueError(
                    f"charge_standardization must be either a float or 'log', "
                    f"but got {self._charge_standardization}"
                )

        if not isinstance(self._time_standardization, float):
            raise ValueError(
                f"time_standardization must be a float, "
                f"but got {self._time_standardization}"
            )
