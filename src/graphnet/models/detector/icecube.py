"""IceCube-specific `Detector` class(es)."""

from typing import Dict, Callable
import torch
import os

from graphnet.models.detector.detector import Detector
from graphnet.constants import ICECUBE_GEOMETRY_TABLE_DIR


class IceCubeBundle(Detector):
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
            "dom_time": self._dom_time,
            "adjusted_time": self._adjusted_time,
            "dom_qtot": self._charge,
            "dom_qtot_exc": self._charge,
            "t25": self._percentile_time,
            "t50": self._percentile_time,
            "t100": self._percentile_time,
            "q100": self._charge,
            "q250": self._charge,
            "q500": self._charge,
            "rde": self._rde,
            "pmt_area": self._pmt_area,
            "bright_dom": self._dom_cond,
            "is_saturated_dom": self._dom_cond,
            "is_errata_dom": self._dom_cond,
        }

        return feature_map

    def _dom_xyz(self, x: torch.tensor) -> torch.tensor:
        return x / 500.0

    def _dom_time(self, x: torch.tensor) -> torch.tensor:
        return (x - 1.0e04) / 3.0e4

    def _adjusted_time(self, x: torch.tensor) -> torch.tensor:
        return x / 3.0e4

    def _percentile_time(self, x: torch.tensor) -> torch.tensor:
        cond = x != -100
        x[cond] = x[cond] / 3.0e4
        return x

    def _charge(self, x: torch.tensor) -> torch.tensor:

        cond = x != -100
        x[cond] = torch.log10(x[cond] + 1)
        return x

    def _rde(self, x: torch.tensor) -> torch.tensor:
        return (x - 1.25) / 0.25

    def _pmt_area(self, x: torch.tensor) -> torch.tensor:
        return x / 0.05

    def _dom_cond(self, x: torch.tensor) -> torch.tensor:
        return x


class IceCubeBundleNew(Detector):
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
            "dom_time": self._dom_time,
            "adjusted_time": self._adjusted_time,
            "dom_qtot": self._charge,
            "dom_qtot_exc": self._charge,
            "qcumsum": self._qcumsum,
            "rde": self._rde,
            "pmt_area": self._pmt_area,
            "bright_dom": self._dom_cond,
            "dom_hit": self._dom_cond,
            "saturation_total_time": self._charge,
            "in_saturation_window": self._identity,
            "in_calibration_errata": self._identity,
            "t_from_leading": self._identity,
        }

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
        extra_features_percentiles = [
            f"time_charge_{p}" for p in time_charge_percentiles
        ]
        extra_features_percentiles_excl = [
            f"time_charge_{p}_excl" for p in time_charge_percentiles
        ]

        dict_extra_features = {feat: self._charge for feat in extra_features}
        dict_extra_features_excl = {
            feat: self._charge for feat in extra_features_excl
        }
        dict_extra_features_percentiles = {
            feat: self._adjusted_time for feat in extra_features_percentiles
        }
        dict_extra_features_percentiles_excl = {
            feat: self._adjusted_time
            for feat in extra_features_percentiles_excl
        }

        feature_map.update(dict_extra_features)
        feature_map.update(dict_extra_features_excl)
        feature_map.update(dict_extra_features_percentiles)
        feature_map.update(dict_extra_features_percentiles_excl)

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


class IceCube86(Detector):
    """`Detector` class for IceCube-86."""

    geometry_table_path = os.path.join(
        ICECUBE_GEOMETRY_TABLE_DIR, "icecube86.parquet"
    )
    xyz = ["dom_x", "dom_y", "dom_z"]
    string_id_column = "string"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        # feature_map = {
        #    "dom_x": self._dom_xyz,
        #    "dom_y": self._dom_xyz,
        #    "dom_z": self._dom_xyz,
        #    "dom_time": self._dom_time,
        #    "charge": self._charge,
        #    "rde": self._rde,
        #    "pmt_area": self._pmt_area,
        #    "hlc": self._identity,
        # }

        # feature_map = {
        #    "dom_x": self._dom_xyz,
        #    "dom_y": self._dom_xyz,
        #    "dom_z": self._dom_xyz,
        #    "dom_min_time": self._dom_time,
        #    "dom_qtot": self._charge,
        #    "rde": self._rde,
        #    "pmt_area": self._pmt_area,
        # }
        feature_map = {
            "dom_x": self._dom_xyz,
            "dom_y": self._dom_xyz,
            "dom_z": self._dom_xyz,
            "adjusted_time": self._dom_time,
            "dom_qtot": self._charge,
            "rde": self._rde,
            "pmt_area": self._pmt_area,
            "is_bright_dom": self._identity,
        }

        return feature_map

    def _dom_xyz(self, x: torch.tensor) -> torch.tensor:
        return x / 500.0

    def _dom_time(self, x: torch.tensor) -> torch.tensor:
        return (x - 1.0e04) / 3.0e4

    def _charge(self, x: torch.tensor) -> torch.tensor:
        return torch.log10(x)

    def _rde(self, x: torch.tensor) -> torch.tensor:
        return (x - 1.25) / 0.25

    def _pmt_area(self, x: torch.tensor) -> torch.tensor:
        return x / 0.05


class IceCubeKaggle(Detector):
    """`Detector` class for Kaggle Competition."""

    geometry_table_path = os.path.join(
        ICECUBE_GEOMETRY_TABLE_DIR, "icecube86.parquet"
    )
    xyz = ["x", "y", "z"]
    string_id_column = "string"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        feature_map = {
            "x": self._xyz,
            "y": self._xyz,
            "z": self._xyz,
            "time": self._time,
            "charge": self._charge,
            "auxiliary": self._identity,
        }
        return feature_map

    def _xyz(self, x: torch.tensor) -> torch.tensor:
        return x / 500.0

    def _time(self, x: torch.tensor) -> torch.tensor:
        return (x - 1.0e04) / 3.0e4

    def _charge(self, x: torch.tensor) -> torch.tensor:
        return torch.log10(x) / 3.0


class IceCubeDeepCore(IceCube86):
    """`Detector` class for IceCube-DeepCore."""

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        feature_map = {
            "dom_x": self._dom_xy,
            "dom_y": self._dom_xy,
            "dom_z": self._dom_z,
            "dom_time": self._dom_time,
            "charge": self._identity,
            "rde": self._rde,
            "pmt_area": self._pmt_area,
            "hlc": self._identity,
        }
        return feature_map

    def _dom_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 100.0

    def _dom_z(self, x: torch.tensor) -> torch.tensor:
        return (x + 350.0) / 100.0

    def _dom_time(self, x: torch.tensor) -> torch.tensor:
        return ((x / 1.05e04) - 1.0) * 20.0

    def _rde(self, x: torch.tensor) -> torch.tensor:
        return (x - 1.25) / 0.25

    def _pmt_area(self, x: torch.tensor) -> torch.tensor:
        return x / 0.05


class IceCubeUpgrade(Detector):
    """`Detector` class for IceCube-Upgrade."""

    geometry_table_path = os.path.join(
        ICECUBE_GEOMETRY_TABLE_DIR, "icecube_upgrade.parquet"
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
            "dom_time": self._dom_time,
            "charge": self._charge,
            "rde": self._identity,
            "pmt_area": self._pmt_area,
            "string": self._string,
            "pmt_number": self._pmt_number,
            "dom_number": self._dom_number,
            "pmt_dir_x": self._identity,
            "pmt_dir_y": self._identity,
            "pmt_dir_z": self._identity,
            "dom_type": self._dom_type,
            "hlc": self._identity,
        }

        return feature_map

    def _dom_time(self, x: torch.tensor) -> torch.tensor:
        return (x / 2e04) - 1.0

    def _charge(self, x: torch.tensor) -> torch.tensor:
        return torch.log10(x) / 2.0

    def _string(self, x: torch.tensor) -> torch.tensor:
        return (x - 50.0) / 50.0

    def _pmt_number(self, x: torch.tensor) -> torch.tensor:
        return x / 20.0

    def _dom_number(self, x: torch.tensor) -> torch.tensor:
        return (x - 60.0) / 60.0

    def _dom_type(self, x: torch.tensor) -> torch.tensor:
        return x / 130.0

    def _dom_xyz(self, x: torch.tensor) -> torch.tensor:
        return x / 500.0

    def _pmt_area(self, x: torch.tensor) -> torch.tensor:
        return x / 0.05
