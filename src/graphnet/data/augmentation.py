"""Random augmentations of the input rows of an event.

A `DataAugmentation` changes the raw input array of an event, e.g. its
pulses, each time the event is read, before the `DataRepresentation` turns
it into the model input. Node features computed from the rows (e.g. by
`ClusterSummaryFeatures`) are therefore recomputed from the augmented rows.

Augmentations are given to a `Dataset` (`augmentations=[...]`). When a
`DatasetConfig` defines several selections, they are only applied to the
selections whose name contains "train" (the selections that are also
shuffled), so validation and test events are left unchanged.
"""

import os
from abc import abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from numpy.random import Generator, SeedSequence, default_rng

from graphnet.models import Model

# One generator per process and seed, shared by all augmentations, so that
# copies of an augmentation (e.g. one per file of an `EnsembleDataset`) do
# not repeat each other's draws.
_GENERATORS: Dict[Optional[int], Generator] = {}
_GENERATORS_PID: Optional[int] = None


def _process_generator(seed: Optional[int]) -> Generator:
    """Return the generator of the current process for `seed`."""
    global _GENERATORS_PID
    pid = os.getpid()
    if _GENERATORS_PID != pid:
        _GENERATORS.clear()
        _GENERATORS_PID = pid
    if seed not in _GENERATORS:
        entropy = [torch.initial_seed()]
        if seed is not None:
            entropy.append(seed)
        _GENERATORS[seed] = default_rng(SeedSequence(entropy))
    return _GENERATORS[seed]


class DataAugmentation(Model):
    """Base class for random augmentations of the input rows of an event.

    Each time an event is read, the augmentation is applied with probability
    `apply_probability`; otherwise the event is returned unchanged.

    Random numbers come from one generator per process (shared by all
    augmentations with the same `seed`), seeded from `seed` and from torch's
    seed of the process (`torch.initial_seed()`, which differs between
    `DataLoader` workers). Workers thus draw independently, and the draws
    are reproducible when torch is seeded, e.g. with
    `pytorch_lightning.seed_everything(..., workers=True)`.
    """

    def __init__(
        self,
        apply_probability: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        """Construct `DataAugmentation`.

        Args:
            apply_probability: Probability that an event is augmented each
                time it is read. Defaults to 1.
            seed: Optional seed, combined with torch's seed of the process.
        """
        super().__init__(name=__name__, class_name=self.__class__.__name__)
        if not 0.0 <= apply_probability <= 1.0:
            raise ValueError(
                f"`apply_probability` must be in [0, 1], "
                f"got {apply_probability}."
            )
        self._apply_probability = apply_probability
        self._seed = seed
        self._column_cache: Dict[Tuple[str, ...], Dict[str, int]] = {}

    @property
    def rng(self) -> Generator:
        """Return the random number generator of the current process."""
        return _process_generator(self._seed)

    def forward(  # type: ignore[override]
        self, input_features: np.ndarray, input_feature_names: List[str]
    ) -> np.ndarray:
        """Return the augmented rows of one event.

        Args:
            input_features: Rows of the event, shape `[n_rows, n_features]`.
            input_feature_names: Name of each column of `input_features`.

        Returns:
            The augmented rows (a new array), or `input_features` itself if
            the event is not augmented this time.
        """
        if len(input_features) == 0:
            return input_features
        if self.rng.random() >= self._apply_probability:
            return input_features
        return self._augment(input_features, input_feature_names)

    @abstractmethod
    def _augment(
        self, input_features: np.ndarray, input_feature_names: List[str]
    ) -> np.ndarray:
        """Augment the rows of one event."""

    def _column(self, name: str, input_feature_names: List[str]) -> int:
        """Return the column index of feature `name`."""
        key = tuple(input_feature_names)
        if key not in self._column_cache:
            self._column_cache[key] = {
                feature: index for index, feature in enumerate(key)
            }
        try:
            return self._column_cache[key][name]
        except KeyError:
            raise ValueError(
                f"{self.__class__.__name__} needs feature '{name}', which "
                f"is not among the input features {input_feature_names}."
            ) from None


class _DropoutAugmentation(DataAugmentation):
    """Shared settings of the dropout augmentations."""

    def __init__(
        self,
        drop_rate: Union[float, Sequence[float]],
        charge_feature: Optional[str] = "charge",
        preserve_charge_sum: bool = True,
        apply_probability: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(apply_probability=apply_probability, seed=seed)
        if isinstance(drop_rate, (int, float)):
            low, high = float(drop_rate), float(drop_rate)
        else:
            if len(drop_rate) != 2:
                raise ValueError(
                    "`drop_rate` must be a number or a (low, high) range, "
                    f"got {drop_rate}."
                )
            low, high = float(drop_rate[0]), float(drop_rate[1])
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError(
                f"`drop_rate` must satisfy 0 <= low <= high <= 1, "
                f"got {drop_rate}."
            )
        if preserve_charge_sum and charge_feature is None:
            raise ValueError("`preserve_charge_sum` needs a `charge_feature`.")
        self._drop_rate = (low, high)
        self._charge_feature = charge_feature
        self._preserve_charge_sum = preserve_charge_sum

    def _sample_drop_rate(self) -> float:
        """Draw the drop rate of one event."""
        low, high = self._drop_rate
        if low == high:
            return low
        return float(self.rng.uniform(low, high))

    def _keep(
        self,
        input_features: np.ndarray,
        input_feature_names: List[str],
        keep: np.ndarray,
    ) -> np.ndarray:
        """Return the kept rows, rescaling the charge if requested."""
        kept = input_features[keep]
        if self._preserve_charge_sum:
            assert self._charge_feature is not None
            column = self._column(self._charge_feature, input_feature_names)
            kept_sum = kept[:, column].sum()
            if kept_sum > 0:
                scale = input_features[:, column].sum() / kept_sum
                kept[:, column] = kept[:, column] * scale
        return kept


class RandomRowDropout(_DropoutAugmentation):
    """Randomly remove rows (e.g. pulses) of an event.

    The drop rate `p` of an event is `drop_rate`, or drawn uniformly from
    `drop_rate = (low, high)` for each event. With `mode="uniform"`, every
    row is removed with probability `p`. With `mode="thinning"`, a row is
    removed with probability `p ** max(q, 1)`, `q` being the row's charge in
    photoelectrons: the probability that all of its photons are lost if
    each photon is lost with probability `p`, as for a lower sensor
    efficiency. Bright rows are then rarely removed.

    With `preserve_charge_sum`, the charge of the remaining rows is scaled
    so that the total charge of the event is unchanged.

    If fewer than `min_rows` rows would remain, the event is returned
    unchanged.
    """

    def __init__(
        self,
        drop_rate: Union[float, Sequence[float]],
        mode: str = "uniform",
        charge_feature: Optional[str] = "charge",
        preserve_charge_sum: bool = True,
        min_rows: int = 1,
        apply_probability: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        """Construct `RandomRowDropout`.

        Args:
            drop_rate: Probability `p` of removing a row, or a `(low, high)`
                range from which `p` is drawn uniformly for each event.
            mode: "uniform" (every row removed with probability `p`) or
                "thinning" (a row of charge `q` removed with probability
                `p ** max(q, 1)`). Defaults to "uniform".
            charge_feature: Name of the charge column, used by "thinning"
                and `preserve_charge_sum`. Defaults to "charge".
            preserve_charge_sum: Scale the charge of the remaining rows so
                that the total charge of the event is unchanged. Defaults to
                True.
            min_rows: Minimum number of rows to keep. Defaults to 1.
            apply_probability: Probability that an event is augmented each
                time it is read. Defaults to 1.
            seed: Optional seed, combined with torch's seed of the process.
        """
        super().__init__(
            drop_rate=drop_rate,
            charge_feature=charge_feature,
            preserve_charge_sum=preserve_charge_sum,
            apply_probability=apply_probability,
            seed=seed,
        )
        if mode not in ("uniform", "thinning"):
            raise ValueError(
                f"`mode` must be 'uniform' or 'thinning', got '{mode}'."
            )
        if mode == "thinning" and charge_feature is None:
            raise ValueError("`mode='thinning'` needs a `charge_feature`.")
        if min_rows < 0:
            raise ValueError(f"`min_rows` must be >= 0, got {min_rows}.")
        self._mode = mode
        self._min_rows = min_rows

    def _augment(
        self, input_features: np.ndarray, input_feature_names: List[str]
    ) -> np.ndarray:
        p = self._sample_drop_rate()
        n_rows = len(input_features)
        if self._mode == "uniform":
            drop_probability = np.full(n_rows, p)
        else:
            assert self._charge_feature is not None
            column = self._column(self._charge_feature, input_feature_names)
            charge = np.maximum(input_features[:, column], 1.0)
            drop_probability = p**charge
        keep = self.rng.random(n_rows) >= drop_probability
        if keep.sum() < self._min_rows:
            return input_features
        return self._keep(input_features, input_feature_names, keep)


class RandomSensorDropout(_DropoutAugmentation):
    """Randomly remove all rows of some sensors of an event.

    Sensors are identified by the values of `sensor_features` (e.g. the
    sensor position, or a sensor id). Each sensor with rows in the event is
    removed with probability `p`, which is `drop_rate`, or drawn uniformly
    from `drop_rate = (low, high)` for each event.

    With `preserve_charge_sum`, the charge of the remaining rows is scaled
    so that the total charge of the event is unchanged.

    If fewer than `min_sensors` sensors would remain, the event is returned
    unchanged.
    """

    def __init__(
        self,
        drop_rate: Union[float, Sequence[float]],
        sensor_features: List[str],
        charge_feature: Optional[str] = "charge",
        preserve_charge_sum: bool = True,
        min_sensors: int = 1,
        apply_probability: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        """Construct `RandomSensorDropout`.

        Args:
            drop_rate: Probability `p` of removing a sensor, or a
                `(low, high)` range from which `p` is drawn uniformly for
                each event.
            sensor_features: Names of the columns that identify a sensor,
                e.g. `["dom_x", "dom_y", "dom_z"]`.
            charge_feature: Name of the charge column, used by
                `preserve_charge_sum`. Defaults to "charge".
            preserve_charge_sum: Scale the charge of the remaining rows so
                that the total charge of the event is unchanged. Defaults to
                True.
            min_sensors: Minimum number of sensors to keep. Defaults to 1.
            apply_probability: Probability that an event is augmented each
                time it is read. Defaults to 1.
            seed: Optional seed, combined with torch's seed of the process.
        """
        super().__init__(
            drop_rate=drop_rate,
            charge_feature=charge_feature,
            preserve_charge_sum=preserve_charge_sum,
            apply_probability=apply_probability,
            seed=seed,
        )
        if len(sensor_features) == 0:
            raise ValueError("`sensor_features` must not be empty.")
        if min_sensors < 0:
            raise ValueError(f"`min_sensors` must be >= 0, got {min_sensors}.")
        self._sensor_features = list(sensor_features)
        self._min_sensors = min_sensors

    def _augment(
        self, input_features: np.ndarray, input_feature_names: List[str]
    ) -> np.ndarray:
        p = self._sample_drop_rate()
        columns = [
            self._column(name, input_feature_names)
            for name in self._sensor_features
        ]
        _, sensor_of_row = np.unique(
            input_features[:, columns], axis=0, return_inverse=True
        )
        sensor_of_row = sensor_of_row.reshape(-1)
        n_sensors = int(sensor_of_row.max()) + 1
        keep_sensor = self.rng.random(n_sensors) >= p
        if keep_sensor.sum() < self._min_sensors:
            return input_features
        keep = keep_sensor[sensor_of_row]
        return self._keep(input_features, input_feature_names, keep)
