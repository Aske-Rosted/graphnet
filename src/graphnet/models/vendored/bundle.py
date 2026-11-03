"""Model classes of Maxwell Nakos' production bundle models (vendored).

Vendored verbatim from GraphNeT 99f030bfd4c396c2f5127b1396e58f4e4048dc1b (the production container lowe_v4.sif; ZeptoByte/graphnet
history), so that his deployed models (`model_config.yml` with these class names) load. Only the imports differ. Not linted (see .pre-commit-config.yaml), so that it stays
comparable to the source; do not edit, re-vendor instead. See README.md in this folder.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.functional import one_hot
from torch_geometric.data import Data

from graphnet.constants import ICECUBE_GEOMETRY_TABLE_DIR
from graphnet.models.data_representation.graphs.nodes.nodes import NodeDefinition
from graphnet.models.detector.detector import Detector
from graphnet.models.task import StandardLearnedTask
from graphnet.training.loss_functions import LossFunction


# src/graphnet/models/detector/icecube.py
class IceCubeBundleAdvanced(Detector):
    """`Detector` class for IceCube-Bundle Rejection."""

    geometry_table_path = os.path.join(
        ICECUBE_GEOMETRY_TABLE_DIR, "icecube86.parquet"
    )
    xyz = ["dom_x", "dom_y", "dom_z"]
    string_id_column = "string"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        feature_map = {
            "dom_x": self._dom_xyz,
            "dom_y": self._dom_xyz,
            "dom_z": self._dom_xyz,
            "time": self._dom_time,
            "adjusted_time": self._adjusted_time,
            "time_from_median": self._adjusted_time,
            "dom_qtot": self._charge,
            "dom_qtot_exc": self._charge,
            "dom_qtot_no_afterpulse": self._charge,
            "dom_qtot_exc_no_afterpulse": self._charge,
            "qcumsum": self._qcumsum,
            "rde": self._rde,
            "pmt_area": self._pmt_area,
            "bright_dom": self._dom_cond,
            "dom_hit": self._dom_cond,
            "saturation_total_time": self._charge,
            "errata_total_time": self._charge,
        }

        charge_after_t_threholds = [
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            40,
            45,
            50,
            60,
            70,
            80,
            90,
            100,
        ]
        time_charge_percentiles = [1, 3, 6, 10, 15, 25, 50, 80]
        extra_features = [
            f"charge_after_{t}" for t in charge_after_t_threholds
        ]
        extra_features_excl = [
            f"charge_after_{t}_excl" for t in charge_after_t_threholds
        ]
        extra_features_sat = [f'charge_after_{t}_sat' for t in charge_after_t_threholds]
        extra_features_percentiles = [
            f"time_charge_{p}" for p in time_charge_percentiles
        ]
        extra_features_percentiles_excl = [
            f"time_charge_{p}_excl" for p in time_charge_percentiles
        ]
        extra_features_percentiles_sat = [f'time_charge_{p}_sat' for p in time_charge_percentiles]

        dict_extra_features = {feat: self._charge for feat in extra_features}
        dict_extra_features_excl = {
            feat: self._charge for feat in extra_features_excl
        }
        dict_extra_features_sat = {feat: self._charge for feat in extra_features_sat}
        dict_extra_features_percentiles = {
            feat: self._adjusted_time for feat in extra_features_percentiles
        }
        dict_extra_features_percentiles_excl = {
            feat: self._adjusted_time
            for feat in extra_features_percentiles_excl
        }
        dict_extra_features_percentiles_sat = {feat: self._adjusted_time for feat in extra_features_percentiles_sat}

        feature_map.update(dict_extra_features)
        feature_map.update(dict_extra_features_excl)
        feature_map.update(dict_extra_features_sat)
        feature_map.update(dict_extra_features_percentiles)
        feature_map.update(dict_extra_features_percentiles_excl)
        feature_map.update(dict_extra_features_percentiles_sat)

        return feature_map

    def _dom_xyz(self, x: torch.tensor) -> torch.tensor:
        return x / 500.0

    def _dom_time(self, x: torch.tensor) -> torch.tensor:
        return (x - 1.0e04) / 3.0e4

    def _adjusted_time(self, x: torch.tensor) -> torch.tensor:
        return x / 3.0e4

    def _qcumsum(self, x: torch.tensor) -> torch.tensor:
        return x / 25

    def _charge(self, x: torch.tensor) -> torch.tensor:

        return torch.log10(x + 1)

    def _rde(self, x: torch.tensor) -> torch.tensor:
        return (x - 1.25) / 0.25

    def _pmt_area(self, x: torch.tensor) -> torch.tensor:
        return x / 0.05

    def _dom_cond(self, x: torch.tensor) -> torch.tensor:
        return x


# src/graphnet/models/data_representation/graphs/nodes/nodes.py
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

        self.n_features = len(input_feature_names) - 1
        self._time_name = time_name
        self._time_column = time_column
        self.all_features = input_feature_names
        self.feature_indexes = {
            feat: self.all_features.index(feat) for feat in input_feature_names
        }

        self.important_features = [
            feat
            for feat in input_feature_names
            if feat.startswith("time_charge_")
        ]

    def _define_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        return input_feature_names[:-1]

    def _construct_nodes(self, x: torch.Tensor) -> Tuple[Data, List[str]]:

        # Reduce Output Size By Only Keeping Relevant Features
        graph = torch.zeros(
            [len(x[:, self.feature_indexes[self._time_name]]), self.n_features]
        )

        # Modify Percentile_Based_Information
        final_features = self.all_features[: self.n_features]
        for idx, feature in enumerate(final_features):
            if feature in self.important_features:
                graph[:, self.feature_indexes[feature]] = x[
                    :, self.feature_indexes[feature]
                ] - (
                    x[:, self.feature_indexes[self._time_name]]
                    + x[:, self.feature_indexes[self._time_column]]
                )
            else:
                graph[:, self.feature_indexes[feature]] = x[
                    :, self.feature_indexes[feature]
                ]
                
        return graph


# src/graphnet/models/task/reconstruction.py
class LateralSpreadReconstruction(StandardLearnedTask):

    default_target_labels = ["leading_rms_MCPE"]
    default_prediction_labels = ["leading_rms_MCPE_pred"]
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        # Leave as is
        return torch.nn.functional.softplus(x, beta=0.05)


# src/graphnet/training/loss_functions.py
class RMSEAdjustedLoss(LossFunction):

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:

        assert prediction.dim() == 2
        assert prediction.size() == target.size()

        elements = torch.mean(
            (prediction - target) ** 2 / ((1 + target) ** (0.5)), dim=-1
        )
        return elements


class FocalCrossEntropyLoss(LossFunction):
    """Compute cross-entropy loss for classification tasks.

    Predictions are an [N, num_class]-matrix of logits (i.e., non-softmax'ed
    probabilities), and targets are an [N,1]-matrix with integer values in
    (0, num_classes - 1).
    """

    def __init__(
        self,
        options: Union[int, List[Any], Dict[Any, int]],
        ratio: float = 1,
        gamma: float = 2,
        alpha: float = 1,
        *args: Any,
        **kwargs: Any,
    ):
        """Construct CrossEntropyLoss."""
        # Base class constructor
        super().__init__(*args, **kwargs)

        # Member variables
        self._gamma = gamma
        self._alpha = alpha
        self._options = options
        self._nb_classes: int
        if isinstance(self._options, int):
            assert self._options in [torch.int32, torch.int64]
            assert (
                self._options >= 2
            ), f"Minimum of two classes required. Got {self._options}."
            self._nb_classes = options  # type: ignore
        elif isinstance(self._options, list):
            self._nb_classes = len(self._options)  # type: ignore
        elif isinstance(self._options, dict):
            self._nb_classes = len(
                np.unique(list(self._options.values()))
            )  # type: ignore
        else:
            raise ValueError(
                f"Class options of type {type(self._options)} not supported"
            )

        self._loss = nn.CrossEntropyLoss(
            weight=torch.Tensor([ratio, 1]), reduction="none"
        )

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Transform outputs to angle and prepare prediction."""
        if isinstance(self._options, int):
            # Integer number of classes: Targets are expected to be in
            # (0, nb_classes - 1).

            # Target integers are positive
            assert torch.all(target >= 0)

            # Target integers are consistent with the expected number of class.
            assert torch.all(target < self._options)

            assert target.dtype in [torch.int32, torch.int64]
            target_integer = target

        elif isinstance(self._options, list):
            # List of classes: Mapping target classes in list onto
            # (0, nb_classes - 1). Example:
            #    Given options: [1, 12, 13, ...]
            #    Yields: [1, 13, 12] -> [0, 2, 1, ...]
            target_integer = torch.tensor(
                [self._options.index(value) for value in target]
            )

        elif isinstance(self._options, dict):
            # Dictionary of classes: Mapping target classes in dict onto
            # (0, nb_classes - 1). Example:
            #     Given options: {1: 0, -1: 0, 12: 1, -12: 1, ...}
            #     Yields: [1, -1, -12, ...] -> [0, 0, 1, ...]
            target_integer = torch.tensor(
                [self._options[int(value)] for value in target]
            )

        else:
            assert False, "Shouldn't reach here."

        target_one_hot: Tensor = one_hot(target_integer, self._nb_classes).to(
            prediction.device
        )

        cross_loss = self._loss(prediction.float(), target_one_hot.float())

        a_t = self._alpha * target_one_hot.float() + (1 - self._alpha) * (
            1 - target_one_hot.float()
        )
        # p_t = prediction.float()*target + (1-prediction.float())*(1-target_one_hot.float())
        p_t = torch.exp(-cross_loss)
        return (1 - p_t) ** self._gamma * cross_loss
