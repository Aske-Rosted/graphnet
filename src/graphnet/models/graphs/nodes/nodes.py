"""Class(es) for building/connecting graphs."""

from typing import List, Tuple, Optional, Union
from abc import abstractmethod

import torch
from torch_geometric.data import Data

from graphnet.utilities.decorators import final
from graphnet.models import Model
from graphnet.models.graphs.utils import (
    cluster_summarize_with_percentiles,
    identify_indices,
    lex_sort,
    ice_transparency,
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
            max_activations: Maximum number of activations to include in the time series.
        """
        self._keys = keys
        super().__init__(input_feature_names=self._keys)
        self._id_columns = [self._keys.index(key) for key in id_columns]
        self._time_index = self._keys.index(time_column)
        try:
            self._charge_index: Optional[int] = self._keys.index(charge_column)
        except ValueError:
            self.warning(
                "Charge column with name {} not found. Running without.".format(
                    charge_column
                )
            )

            self._charge_index = None

        self._max_activations = max_activations

    def _define_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        return input_feature_names + ["new_node_col"]

    def _construct_nodes(self, x: torch.Tensor) -> Data:
        """Construct nodes from raw node features ´x´."""
        # Cast to Numpy
        x = x.numpy()
        # if there is no charge column add a dummy column of zeros with the same shape as the time column
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
        hlc_name: str = "hlc",
    ) -> None:
        """Construct `IceMixNodes`.

        Args:
            input_feature_names: Column names for input features. Minimum
            required features are z coordinate and hlc column names.
            max_pulses: Maximum number of pulses to keep in the event.
            z_name: Name of the z-coordinate column.
            hlc_name: Name of the `Hard Local Coincidence Check` column.
        """
        super().__init__(input_feature_names=input_feature_names)

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

        if z_name not in input_feature_names:
            raise ValueError(
                f"z name {z_name} not found in "
                f"input_feature_names {input_feature_names}"
            )
        if hlc_name not in input_feature_names:
            raise ValueError(
                f"hlc name {hlc_name} not found in "
                f"input_feature_names {input_feature_names}"
            )

        self.all_features = input_feature_names + [
            "scatt_lenght",
            "abs_lenght",
        ]

        self.feature_indexes = {
            feat: self.all_features.index(feat) for feat in input_feature_names
        }

        self.f_scattering, self.f_absoprtion = ice_transparency()

        self.input_feature_names = input_feature_names
        self.n_features = len(self.all_features)
        self.max_length = max_pulses
        self.z_name = z_name
        self.hlc_name = hlc_name

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
            auxiliary_n = torch.nonzero(
                x[:, self.feature_indexes[self.hlc_name]] == 0
            ).squeeze(1)
            auxiliary_p = torch.nonzero(
                x[:, self.feature_indexes[self.hlc_name]] == 1
            ).squeeze(1)
            ids_n = ids[auxiliary_n][: min(self.max_length, len(auxiliary_n))]
            ids_p = ids[auxiliary_p][
                : min(self.max_length - len(ids_n), len(auxiliary_p))
            ]
            ids = torch.cat([ids_n, ids_p]).sort().values
        return ids

    def _construct_nodes(self, x: torch.Tensor) -> Tuple[Data, List[str]]:

        event_length = x.shape[0]
        x[:, self.feature_indexes[self.hlc_name]] = torch.logical_not(
            x[:, self.feature_indexes[self.hlc_name]]
        )  # hlc in kaggle was flipped
        ids = self._pulse_sampler(x, event_length)
        event_length = min(self.max_length, event_length)

        graph = torch.zeros([event_length, self.n_features])
        for idx, feature in enumerate(
            self.all_features[: self.n_features - 2]
        ):
            graph[:event_length, idx] = x[ids, self.feature_indexes[feature]]

        graph = self._add_ice_properties(graph, x, ids)  # ice properties
        return Data(x=graph)
    

class FirstHitPulses(NodeDefinition):

    def __init__(
        self,
        input_feature_names: Optional[List[str]] = None,
        dom_hit_name: str = "dom_hit",
    ) -> None:
        """Construct `IceMixNodes`.

        Args:
            input_feature_names: Column names for input features. Minimum
            required features are z coordinate and hlc column names.
        """
        super().__init__(input_feature_names=input_feature_names)

        if input_feature_names is None:
            input_feature_names = [
                "dom_x",
                "dom_y",
                "dom_z",
                "dom_time",
                "dom_qtot_exc",
                "saturation_total_time",
                "dom_hit",
            ]

        self.n_features = len(input_feature_names)-1
        self._dom_hit_name = dom_hit_name
        self.all_features = input_feature_names
        self.feature_indexes = {
            feat: self.all_features.index(feat) for feat in input_feature_names
        }

    def _define_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        return input_feature_names[:-1]
    
    def _get_ids(self, x: torch.Tensor) -> torch.Tensor:
        """Get the ids of the pulses that are not in the saturation and calibration errata windows."""
        ids = torch.logical_not(
            x[:, self.feature_indexes["dom_hit"]],
        )
        ids = torch.nonzero(ids).squeeze(1)
        return ids

    def _construct_nodes(self, x: torch.Tensor) -> Tuple[Data, List[str]]:
        
        ids = self._get_ids(x)
        graph = torch.zeros([len(ids), self.n_features])

        final_features = self.all_features[: self.n_features]
        for idx, feature in enumerate(final_features):
            graph[:, idx] = x[ids, self.feature_indexes[feature]]

        return Data(x=graph)
    

class NodesAsPulsesBundle(NodeDefinition):
    
    def __init__(
        self,
        input_feature_names: Optional[List[str]] = None,
        time_name: str = "adjusted_time",
        time_column: str = "time",
    ) -> None:
     
        super().__init__(input_feature_names=input_feature_names)

        # time
        if input_feature_names is None:
            input_feature_names = [
                "dom_x",
                "dom_y",
                "dom_z",
                "adjusted_time",
                "dom_qtot",
            ]

        self.n_features = len(input_feature_names)-1
        self._time_name = time_name
        self._time_column = time_column
        self.all_features = input_feature_names
        self.feature_indexes = {
            feat: self.all_features.index(feat) for feat in input_feature_names
        }

        self.important_features = [feat for feat in input_feature_names if feat.startswith('charge_after_')]

    def _define_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        return input_feature_names[:-1]

    def _construct_nodes(self, x: torch.Tensor) -> Tuple[Data, List[str]]:

        # Reduce Output Size By Only Keeping Relevant Features
        graph = torch.zeros([len(x[:, self.feature_indexes[self._time_name]]), self.n_features])

        # Modify Percentile_Based_Information
        final_features = self.all_features[: self.n_features]
        for idx, feature in enumerate(final_features):
            if feature in self.important_features:
                graph[:, self.feature_indexes[feature]] = (x[:, self.feature_indexes[feature]] - 
                                                (x[:, self.feature_indexes[self._time_name]] + 
                                                x[:, self.feature_indexes[self._time_column]]))
            else:
                graph[:, self.feature_indexes[feature]] = x[:, self.feature_indexes[feature]]

        return Data(x=graph)

class BinnedFeaturesOneNode(NodeDefinition):
    def __init__(
        self,
        input_feature_names: Optional[List[str]] = None,
        charge_name: str = "charge",
        time_name: str = "adjusted_time",
        charge_at_time: List[int] = [10,20,30,40,50,60,75,90],
        first_hit_time: bool = True,
        total_charge: bool = True,
    ) -> None:
        super().__init__(input_feature_names=input_feature_names)

        self.n_features = len(input_feature_names)
        self._dom_hit_name = 'dom_hit'
        self.all_features = input_feature_names
        self.feature_indexes = {
            feat: self.all_features.index(feat) for feat in input_feature_names
        }

        self.charge_at_times = charge_at_time
        print('working')

        
    def _define_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        
        """
        Input Features
        """

        """
        Output Features
        -> Dom x,y,and z coordinates
        -> First Hit Time:
        -> Accumulated_charge: Total charge after specified timing window
        -> Total_Charge on DOM
        """
        
        new_feature_list = deepcopy(input_feature_names)

        new_feature_list.remove('dom_hit')
        #new_feature_list.remove('t_from_leading')
        #new_feature_list.remove('qcumsum')
        new_feature_list.append('accumulated_charge')

        print('running this')

        return new_feature_list

    def _get_ids(self, x: torch.Tensor) -> torch.Tensor:
        """Get the ids of the pulses that are not in the saturation and calibration errata windows."""
        ids = torch.logical_not(
            x[:, self.feature_indexes["dom_hit"]],
        )
        ids = torch.nonzero(ids).squeeze(1)
        return ids
    
    def _get_doms(self, x: torch.Tensor) -> torch.Tensor:

        x = lex_sort(x=x, cluster_columns=[0,1,2,3])

        unique_sensors, self._counts = np.unique(
            x[:, [0,1,2]], axis=0, return_counts=True
        ) 

        contingency_table = np.concatenate(
            [unique_sensors, self._counts.reshape(-1, 1)], axis=1
        )

        self.clustered_x = contingency_table[:, 0 : unique_sensors.shape[1]]

        self._counts = (
            contingency_table[:, self.clustered_x.shape[1] :]
            .flatten()
            .astype(int)
        )

        self._padded_x = np.empty(
            (len(self._counts), max(self._counts), x.shape[1])
        )
        self._padded_x.fill(np.nan)

        for i in range(len(self._counts)):
            self._padded_x[i, : self._counts[i]] = x[: self._counts[i]]
            x = x[self._counts[i] :]

    
    def _calculate_time_first_pulse(self, time_index: int) -> np.ndarray:
        """Calculate the time of the first pulse."""
        assert not hasattr(
            self, "_time_first_pulse"
        ), "Time of first pulse has already been calculated, \
            re-calculation is not allowed"
        
        #print(self._padded_x)
        test_index = 2
        print(self._padded_x[test_index,:,0]*500)
        print(self._padded_x[test_index,:,1]*500)
        print(self._padded_x[test_index,:,2]*500)
        print(self._padded_x[test_index,:,3]*3e4)
        print(self._padded_x[test_index,:,7])
        #print(self._padded_x.shape)
        self._time_first_pulse = np.nanmin(
            self._padded_x[:, :, time_index],
            axis=1,
        )

    def _get_accumulated_charge(self, time_index: int):
        
        tmp_times = (
            np.tile(
                np.array(self.charge_at_times),
                (len(self._time_first_pulse[:, np.newaxis]), 1),
            )
            + self._time_first_pulse[:, np.newaxis]
        )

        # Create a mask for the times
        mask = (
            self._padded_x[:, :, time_index][:, np.newaxis, :]
            >= tmp_times[:, :, np.newaxis]
        )

        selections = np.argmax(
            mask,
            axis=2,
        )
        print(selections)
        selections += (np.arange(len(self._counts)) * self._padded_x.shape[1])[
            :, np.newaxis
        ]
        """
        selections = (
            self._padded_x[:, :, summarization_indices]
            .cumsum(axis=1)
            .reshape(-1, len(summarization_indices))[selections]
        )
        selections = selections.transpose(0, 2, 1).reshape(
            len(self.clustered_x), -1
        )

        # Add the selections to the clustered tensor
        self._add_column(selections, location)

        # update the cluster names
        if self._input_names is not None:
            new_names = [
                self._input_names[i] + "_accumulated_after_" + str(t)
                for i in summarization_indices
                for t in times
            ]
            self._add_column_names(new_names, location)
        """
    
    def _construct_nodes(self, x: torch.tensor) -> Tuple[Data | List[str]]:
        
        doms = x[:, :3]
        unique_doms, inverse_dom_indices = torch.unique(doms, dim=0, return_inverse=True)
        self.time_tensor = torch.tensor(self.charge_at_times)
        T = self.time_tensor.shape[0]
        self._get_doms(x)
        self._calculate_time_first_pulse(time_index=self.feature_indexes['adjusted_time'])


        #ids = self._get_ids(x)
        #graph = torch.zeros([len(ids), self.n_features])

        # Get the Charge Features

        #final_features = self.all_features[: self.n_features]
        #for idx, feature in enumerate(final_features):
        #    graph[:, idx] = x[ids, self.feature_indexes[feature]] 
        
        
        return Data(x=x)

class BinnedPulses(NodeDefinition):

    """Represent each measured pulse of Cherenkov Radiation as a node.

    Exclude Pulses that are not included in the saturation and calibration errata windows
    Bin Pulses in 10 ns bins that describe the ramp up from the first pulse on a dom
    """
    
    def __init__(
        self,
        input_feature_names: Optional[List[str]] = None,
    ) -> None:
        """Construct `IceMixNodes`.

        Args:
            input_feature_names: Column names for input features. Minimum
            required features are z coordinate and hlc column names.
        """
        super().__init__(input_feature_names=input_feature_names)

        if input_feature_names is None:
            input_feature_names = [
                "dom_x",
                "dom_y",
                "dom_z",
                "dom_time",
                "dom_qtot_exc",
                "saturation_total_time",
                "charge",
                "t_from_leading",
                "in_saturation_window",
                "in_calibration_errata",
            ]

        self.n_features = len(input_feature_names)-2

        self.all_features = input_feature_names
        self.feature_indexes = {
            feat: self.all_features.index(feat) for feat in input_feature_names
        }

    def _define_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        return input_feature_names[:-2]
    

"""
Currently Broken
"""
class NodesAsPulsesNoSaturationErrata(NodeDefinition):
    """
    Represent each measured pulse of Cherenkov Radiation as a node.
    Exlude Pulses that are not included in the saturation and calibration errata windows
    """

    def __init__(
        self,
        input_feature_names: Optional[List[str]] = None,
    ) -> None:
        """Construct `IceMixNodes`.

        Args:
            input_feature_names: Column names for input features. Minimum
            required features are z coordinate and hlc column names.
            max_pulses: Maximum number of pulses to keep in the event.
            z_name: Name of the z-coordinate column.
            hlc_name: Name of the `Hard Local Coincidence Check` column.
        """
        super().__init__(input_feature_names=input_feature_names)

        if input_feature_names is None:
            input_feature_names = [
                "dom_x",
                "dom_y",
                "dom_z",
                "dom_time",
                "dom_qtot_exc",
                "saturation_total_time",
                "in_saturation_window",
                "in_calibration_errata",
            ]

        self.n_features = len(input_feature_names)-2

        self.all_features = input_feature_names
        self.feature_indexes = {
            feat: self.all_features.index(feat) for feat in input_feature_names
        }

    def _define_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        return input_feature_names[:-2]
    
    def _get_ids(self, x: torch.Tensor) -> torch.Tensor:
        """Get the ids of the pulses that are not in the saturation and calibration errata windows."""
        ids = torch.logical_and(
            x[:, self.feature_indexes["in_saturation_window"]] == 0,
            x[:, self.feature_indexes["in_calibration_errata"]] == 0,
        )
        ids = torch.nonzero(ids).squeeze(1)
        return ids

    def _construct_nodes(self, x: torch.Tensor) -> Tuple[Data, List[str]]:
        
        ids = self._get_ids(x)
        graph = torch.zeros([len(ids), self.n_features])

        final_features = self.all_features[: self.n_features]
        for idx, feature in enumerate(final_features):
            graph[:, idx] = x[ids, self.feature_indexes[feature]]

        return Data(x=graph)
