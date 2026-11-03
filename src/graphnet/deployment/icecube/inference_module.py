"""IceCube I3InferenceModule.

Contains functionality for writing model predictions to i3 files.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    TYPE_CHECKING,
    Union,
)

import numpy as np
import torch
from torch_geometric.data import Data, Batch

from graphnet.utilities.config import ModelConfig
from graphnet.deployment import DeploymentModule
from graphnet.deployment.deployment_module import (
    add_time_references,
    concatenate_task_outputs,
)
from graphnet.data.extractors.icecube import I3FeatureExtractor
from graphnet.models import Model
from graphnet.utilities.imports import has_icecube_package

if has_icecube_package() or TYPE_CHECKING:
    from icecube.icetray import (
        I3Frame,
    )  # pyright: reportMissingImports=false
    from icecube.dataclasses import (
        I3Direction,
        I3Double,
        I3MapStringDouble,
        I3Particle,
        I3Position,
    )  # pyright: reportMissingImports=false


@dataclass
class _DeployedModel:
    """One model of an `I3InferenceModule` and how its output is written."""

    name: str
    model: Model
    features: List[str]
    columns: List[str]
    reference_times: Dict[str, str]


@dataclass
class _ModelOutput:
    """The output of one model for one frame."""

    name: str
    values: Dict[str, float]  # prediction column -> value
    references: Dict[str, float]  # reference attribute -> value
    has_pulses: bool


def _model_references(model: Model) -> Dict[str, str]:
    """Return prediction label -> reference attribute declared by the tasks.

    These are added back by the model itself (`Task(reference_labels=...)`).
    """
    references = {}
    for task in getattr(model, "_tasks", []):
        for column, attribute in getattr(task, "_reference_columns", []):
            references[task._prediction_labels[column]] = attribute
    return references


class I3InferenceModule(DeploymentModule):
    """General class for inference on i3 frames.

    Every task of a model is written, each prediction column as an
    `I3Double` named `<model_name>_<column>`. Several models can be run in
    one module (lists of `model_config`, `state_dict` and `model_name`):
    the pulses are extracted once per frame, and each model builds its own
    data representation from them.

    Time predictions relative to an event-level reference are written as
    absolute times: either the model adds the reference itself (tasks with
    `reference_labels`), or `reference_times` names the graph attribute to
    add. The references used are also written, as `<model_name>_<attribute>`.
    """

    def __init__(
        self,
        pulsemap_extractor: Union[
            List[I3FeatureExtractor], I3FeatureExtractor
        ],
        model_config: Union[
            ModelConfig, str, Sequence[Union[ModelConfig, str]]
        ],
        state_dict: Union[str, Sequence[str]],
        model_name: Union[str, List[str]],
        gcd_file: str,
        features: Optional[List[str]] = None,
        prediction_columns: Optional[
            Union[Sequence[str], Sequence[Optional[Sequence[str]]]]
        ] = None,
        pulsemap: Optional[str] = None,
        reference_times: Optional[
            Union[Dict[str, str], List[Optional[Dict[str, str]]]]
        ] = None,
        device: str = "cpu",
        overwrite: bool = False,
        requirements: Optional[Callable[["I3Frame"], bool]] = None,
        record_runtimes: bool = False,
        num_threads: Optional[int] = None,
        inference_speed_check: Optional[bool] = None,
    ):
        """General class for inference on I3Frames (physics).

        Arguments:
            pulsemap_extractor: The extractor used to extract the pulsemap.
            model_config: The ModelConfig (or path to it) that summarizes the
                            model used for inference; a list for several
                            models.
            state_dict: Path to state_dict containing the learned weights
                (or a Lightning `.ckpt`); a list for several models.
            model_name: The name used for the model. Will help define the
                        named entry in the I3Frame. E.g. "dynedge". A list
                        (one name per model) for several models.
            gcd_file: path to associated gcd file.
            features: the features of the pulsemap that the models are
                expecting. Defaults to each model's input feature names.
            prediction_columns: column names for the predictions of the model
                (all tasks, in task order). Will help define the named entry
                in the I3Frame. E.g. ['energy_reco']. Optional; for several
                models a list with one entry (or None) per model.
            pulsemap: the pulsmap that the model is expecting as input.
            reference_times: Prediction columns whose values are relative to
                an event-level time reference, mapped to the graph attribute
                that holds the reference, e.g. `{"HEPTime": "reference_time"}`
                for a time target relative to the `reference_time_attribute`
                of `ClusterSummaryFeatures`. The reference is added back, so
                these columns are written as absolute times, and the
                reference itself is written as `<model_name>_<attribute>`.
                Only for models that do not add the reference themselves
                (tasks with `reference_labels`). For several models, a list
                with one entry (or None) per model.
            device: The computational device to run the models on.
            overwrite: If True, replace frame objects that already exist
                under the same key; otherwise keep them (with a warning).
            requirements: Optional function of the frame; frames for which it
                returns False are passed on without inference.
            record_runtimes: If True, write the time spent per frame on data
                representation, inference and writing (and their total) as an
                `I3MapStringDouble` named `<first model_name>_Speed`.
            num_threads: If given, `torch.set_num_threads(num_threads)` is
                called before each frame's inference (other modules in the
                tray may change the number of threads).
            inference_speed_check: Deprecated name of `record_runtimes`.
        """
        super().__init__(
            model_config=model_config,
            state_dict=state_dict,
            prediction_columns=prediction_columns,
            device=device,
        )
        if inference_speed_check is not None:
            self.warning(
                "`inference_speed_check` is deprecated; use "
                "`record_runtimes`."
            )
            record_runtimes = record_runtimes or inference_speed_check
        # Checks
        assert isinstance(gcd_file, str), "gcd_file must be string"
        names = model_name if isinstance(model_name, list) else [model_name]
        assert len(names) == len(self.models), "Give one model_name per model."
        assert len(set(names)) == len(names), "model names must be unique."
        if reference_times is None or isinstance(reference_times, dict):
            references_per_model = [reference_times] * len(self.models)
        else:
            references_per_model = list(reference_times)
        assert len(references_per_model) == len(
            self.models
        ), "Give one reference_times entry (or None) per model."

        # Set Member Variables
        if isinstance(pulsemap_extractor, list):
            self._i3_extractors = pulsemap_extractor
        else:
            self._i3_extractors = [pulsemap_extractor]
        self._deployed = [
            self._deploy(name, model, columns, features, references)
            for name, model, columns, references in zip(
                names,
                self.models,
                self.prediction_columns_per_model,
                references_per_model,
            )
        ]
        self.model_name = model_name
        self._graph_definition = self.model._graph_definition
        self._features = self._deployed[0].features
        self._pulsemap = pulsemap
        self._gcd_file = gcd_file
        self._overwrite = overwrite
        self._requirements = requirements
        self._record_runtimes = record_runtimes
        self._num_threads = num_threads

        # Set GCD file for pulsemap extractor
        for i3_extractor in self._i3_extractors:
            i3_extractor.set_gcd(i3_file="", gcd_file=self._gcd_file)

    def _deploy(
        self,
        name: str,
        model: Model,
        columns: List[str],
        features: Optional[List[str]],
        reference_times: Optional[Dict[str, str]],
    ) -> _DeployedModel:
        """Check how a model's output is written, and describe it."""
        reference_times = dict(reference_times or {})
        unknown = set(reference_times) - set(columns)
        assert not unknown, f"reference_times: unknown columns {unknown}"
        added_by_model = _model_references(model)
        twice = set(reference_times) & set(added_by_model)
        if twice:
            raise ValueError(
                f"{name}: the model already adds the references of {twice} "
                "(reference_labels); `reference_times` would add them twice."
            )
        node_reference = getattr(
            model._graph_definition._node_definition,
            "_reference_time_attribute",
            None,
        )
        used = set(reference_times.values()) | set(added_by_model.values())
        if node_reference is not None and node_reference not in used:
            self.warning(
                f"{name}: the node definition adds the time reference "
                f"'{node_reference}' to each graph, but neither the model "
                "(reference_labels) nor `reference_times` uses it: time "
                "predictions relative to it are written as relative times."
            )
        if features is None:
            features = model._graph_definition._input_feature_names
        # references added by the model are written too (traceability)
        return _DeployedModel(
            name=name,
            model=model,
            features=list(features),
            columns=list(columns),
            reference_times=reference_times,
        )

    def __call__(self, frame: I3Frame) -> bool:
        """Write predictions from model(s) to frame."""
        if self._requirements is not None and not self._requirements(frame):
            return True
        if self._num_threads is not None:
            torch.set_num_threads(self._num_threads)
        start = perf_counter()
        # Extract the pulses once for all models
        feature_dicts = [extractor(frame) for extractor in self._i3_extractors]
        graphs = [
            self._create_graph(feature_dicts, deployed)
            for deployed in self._deployed
        ]
        represented = perf_counter()

        # Inference; relative times -> absolute times
        outputs = []
        for deployed, data in zip(self._deployed, graphs):
            predictions = self._apply_model(data=data, deployed=deployed)
            predictions, references = add_time_references(
                predictions, data, deployed.columns, deployed.reference_times
            )
            for column, attribute in _model_references(deployed.model).items():
                if column in deployed.columns:
                    references[attribute] = (
                        float(data[attribute].reshape(-1)[0])
                        if data is not None
                        else float("nan")
                    )
            outputs.append(
                _ModelOutput(
                    name=deployed.name,
                    values=dict(zip(deployed.columns, predictions[0])),
                    references=references,
                    has_pulses=data is not None,
                )
            )
        inferred = perf_counter()

        # Submit to frame
        self._write(frame=frame, outputs=outputs)
        written = perf_counter()
        if self._record_runtimes:
            self._put(
                frame,
                self._runtimes_key(),
                I3MapStringDouble(
                    {
                        "data_repr_time": represented - start,
                        "inference_time": inferred - represented,
                        "write_time": written - inferred,
                        "total_time": written - start,
                    }
                ),
            )
        return True

    def _runtimes_key(self) -> str:
        return self._deployed[0].name + "_Speed"

    def _write(self, frame: I3Frame, outputs: List[_ModelOutput]) -> None:
        """Write each model's output as I3Doubles `<model_name>_<column>`."""
        for output in outputs:
            self._write_doubles(frame, output.name, output.values)
            self._write_doubles(frame, output.name, output.references)

    def _write_doubles(
        self, frame: I3Frame, name: str, values: Dict[str, float]
    ) -> None:
        for column, value in values.items():
            self._put(frame, name + "_" + column, I3Double(float(value)))

    def _put(self, frame: I3Frame, key: str, value: Any) -> None:
        """Put `value` in the frame, respecting `overwrite`."""
        if key in frame:
            if not self._overwrite:
                self.warning(
                    f"{key} already exists in frame and overwrite is False;"
                    " keeping the existing object."
                )
                return
            frame.Delete(key)
        frame.Put(key, value)

    def _apply_model(
        self, data: Optional[Data], deployed: _DeployedModel
    ) -> np.ndarray:
        """Apply model to `Data`: all tasks, shape [1, n_columns].

        An event without pulses gets NaN in every column.
        """
        n_columns = len(deployed.columns)
        if data is None:
            self.warning(
                f"Event without pulses - padding the {deployed.name} "
                "predictions with NaN."
            )
            return np.full((1, n_columns), np.nan)
        predictions = concatenate_task_outputs(
            self._inference(data, deployed.model)
        )
        if predictions.shape != (1, n_columns):
            raise ValueError(
                f"{deployed.name}: predictions have shape "
                f"{predictions.shape}, but there are {n_columns} prediction "
                f"columns {deployed.columns}."
            )
        return predictions

    def _create_data_representation(self, frame: I3Frame) -> Optional[Batch]:
        """Process Physics I3Frame into the (first) model's graph."""
        feature_dicts = [extractor(frame) for extractor in self._i3_extractors]
        return self._create_graph(feature_dicts, self._deployed[0])

    def _extract_feature_array_from_frame(self, frame: I3Frame) -> np.ndarray:
        """Apply the I3FeatureExtractors to the I3Frame.

        Arguments:
            frame: Physics I3Frame (PFrame)

        Returns:
            array with pulsemap, the (first) model's features as columns
        """
        return self._feature_array(
            [extractor(frame) for extractor in self._i3_extractors],
            self._features,
        )

    def _add_to_frame(self, frame: I3Frame, data: Dict[str, Any]) -> I3Frame:
        """Add every field in data to I3Frame (respecting `overwrite`).

        Arguments:
            frame: I3Frame (physics)
            data: Dictionary containing content that will be written to frame.

        Returns:
            frame: Same I3Frame as input, but with the new entries
        """
        assert isinstance(
            data, dict
        ), f"data must be of type dict. Got {type(data)}"
        for key, value in data.items():
            self._put(frame, key, value)
        return frame

    @staticmethod
    def _feature_array(
        feature_dicts: List[Dict[str, Any]], features: List[str]
    ) -> np.ndarray:
        """Stack the extracted pulses of all extractors, `features` columns."""
        return np.concatenate(
            [
                np.array([feature_dict[key] for key in features]).T.reshape(
                    -1, len(features)
                )
                for feature_dict in feature_dicts
            ],
            axis=0,
        )

    def _create_graph(
        self, feature_dicts: List[Dict[str, Any]], deployed: _DeployedModel
    ) -> Optional[Batch]:
        """Build a model's graph from the extracted pulses (None if empty)."""
        input_features = self._feature_array(feature_dicts, deployed.features)
        if len(input_features) == 0:
            return None
        data = deployed.model._graph_definition(
            input_features=input_features,
            input_feature_names=deployed.features,
        )
        return Batch.from_data_list([data]).to(self._device)


class I3ParticleInferenceModule(I3InferenceModule):
    """I3InferenceModule that also writes the reconstruction as I3Particle.

    For each model, the configured direction, position, time and energy
    columns are written as an `I3Particle` named `<model_name>_particle`
    instead of as I3Doubles; all other columns are written as I3Doubles.
    """

    def __init__(
        self,
        direction: List[str],
        position: List[str],
        time: str,
        energy: str,
        shape: Optional[Any] = None,
        **kwargs: Any,
    ):
        """Construct `I3ParticleInferenceModule`.

        Arguments:
            direction: Prediction columns of the direction: three (Cartesian
                components, as `I3Direction(x, y, z)`) or two (zenith,
                azimuth).
            position: Prediction columns of the position (x, y, z).
            time: Prediction column of the (absolute) time at `position`.
            energy: Prediction column of the energy.
            shape: `I3Particle.ParticleShape` of the particles. Defaults to
                `InfiniteTrack`.
            **kwargs: Arguments of `I3InferenceModule`.
        """
        super().__init__(**kwargs)
        assert len(direction) in (2, 3), "direction needs 2 or 3 columns"
        assert len(position) == 3, "position needs 3 columns"
        self._direction = list(direction)
        self._position = list(position)
        self._time = time
        self._energy = energy
        self._particle_columns = (
            self._direction
            + self._position
            + [
                time,
                energy,
            ]
        )
        for deployed in self._deployed:
            missing = set(self._particle_columns) - set(deployed.columns)
            assert (
                not missing
            ), f"{deployed.name}: no prediction columns {missing}"
        self._shape = (
            shape
            if shape is not None
            else I3Particle.ParticleShape.InfiniteTrack
        )

    def _write(self, frame: I3Frame, outputs: List[_ModelOutput]) -> None:
        for output in outputs:
            values = {
                c: v
                for c, v in output.values.items()
                if c not in self._particle_columns
            }
            self._put(frame, output.name + "_particle", self._particle(output))
            self._write_doubles(frame, output.name, values)
            self._write_doubles(frame, output.name, output.references)

    def _particle(self, output: _ModelOutput) -> "I3Particle":
        values = output.values
        particle = I3Particle()
        particle.dir = I3Direction(*[values[c] for c in self._direction])
        particle.pos = I3Position(*[values[c] for c in self._position])
        particle.time = values[self._time]
        particle.energy = values[self._energy]
        particle.shape = self._shape
        finite = np.isfinite([values[c] for c in self._particle_columns]).all()
        if not output.has_pulses:
            particle.fit_status = I3Particle.FitStatus.InsufficientHits
        elif not finite:
            particle.fit_status = I3Particle.FitStatus.GeneralFailure
        else:
            particle.fit_status = I3Particle.FitStatus.OK
        return particle


class I3MultipleModelInferenceModule(I3InferenceModule):
    """I3InferenceModule writing all models' outputs into one map.

    The outputs go into one `I3MapStringDouble` named `key_name`: a model
    with a single prediction column under `<model_name>`, otherwise each
    column under `<model_name>_<column>` (and its references under
    `<model_name>_<attribute>`). With `first_column_only`, every model's
    first prediction column only, under `<model_name>`.
    """

    def __init__(
        self,
        key_name: str,
        first_column_only: bool = False,
        multiple_models: Optional[bool] = None,
        **kwargs: Any,
    ):
        """Construct `I3MultipleModelInferenceModule`.

        Arguments:
            key_name: Name of the `I3MapStringDouble` in the frame. Runtimes
                (`record_runtimes`) are written as `<key_name>_Speed`.
            first_column_only: If True, write only each model's first
                prediction column (of its first task), under `<model_name>`;
                e.g. the first-class probability of a classifier. This is
                the map layout of the earlier `multiple_models=True` modules.
            multiple_models: Deprecated: True means `first_column_only`.
            **kwargs: Arguments of `I3InferenceModule`.
        """
        super().__init__(**kwargs)
        if multiple_models is not None:
            self.warning(
                "`multiple_models` is deprecated: several models are always "
                "supported; `multiple_models=True` now means "
                "`first_column_only=True`."
            )
            first_column_only = first_column_only or multiple_models
        self._key_name = key_name
        self._first_column_only = first_column_only

    def _runtimes_key(self) -> str:
        return self._key_name + "_Speed"

    def _write(self, frame: I3Frame, outputs: List[_ModelOutput]) -> None:
        scores: Dict[str, float] = {}
        for output in outputs:
            if self._first_column_only:
                scores[output.name] = float(next(iter(output.values.values())))
                continue
            if len(output.values) == 1 and not output.references:
                scores[output.name] = float(*output.values.values())
                continue
            for column, value in {
                **output.values,
                **output.references,
            }.items():
                scores[output.name + "_" + column] = float(value)
        self._put(frame, self._key_name, I3MapStringDouble(scores))
