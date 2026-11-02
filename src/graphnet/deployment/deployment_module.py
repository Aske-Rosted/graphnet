"""Class(es) for deploying GraphNeT models in icetray as I3Modules."""

from abc import abstractmethod
from typing import Any, List, Union, Dict, Optional, Sequence, Tuple

import numpy as np
from torch import Tensor, load
from torch_geometric.data import Data, Batch

from graphnet.models import Model
from graphnet.utilities.config import ModelConfig
from graphnet.utilities.logging import Logger


def add_time_references(
    predictions: np.ndarray,
    data: Optional[Data],
    prediction_columns: List[str],
    reference_times: Dict[str, str],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Add the event's time references to relative time predictions.

    Arguments:
        predictions: Predictions of one event, shape [1, n_columns].
        data: The event's graph (batch of one), or None for an event
            without pulses (then the references are NaN).
        prediction_columns: Column names of `predictions`.
        reference_times: Prediction column -> graph attribute holding the
            time reference of that column.

    Returns:
        The predictions with the references added (float64), and the
        references by attribute name.
    """
    predictions = np.array(predictions, dtype=np.float64)
    references: Dict[str, float] = {}
    for column, attribute in reference_times.items():
        if attribute not in references:
            references[attribute] = (
                float(data[attribute].reshape(-1)[0])
                if data is not None
                else float("nan")
            )
        predictions[:, prediction_columns.index(column)] += references[
            attribute
        ]
    return predictions, references


def concatenate_task_outputs(outputs: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate the outputs of all tasks of a model, per event.

    Arguments:
        outputs: One array per task, each of shape [n_events] or
            [n_events, n_columns].

    Returns:
        Array of shape [n_events, total number of columns], in task order.
    """
    return np.concatenate(
        [np.asarray(o).reshape(len(o), -1) for o in outputs], axis=1
    )


class DeploymentModule(Logger):
    """Base DeploymentModule for GraphNeT.

    Contains standard methods for loading models doing inference with them.
    Experiment-specific implementations may overwrite methods and should define
    `__call__`.

    One or several models can be deployed: pass lists of `model_config` and
    `state_dict` (and of `prediction_columns`) for several. `self.model` and
    `self.prediction_columns` are those of the first model.
    """

    def __init__(
        self,
        model_config: Union[
            ModelConfig, str, Sequence[Union[ModelConfig, str]]
        ],
        state_dict: Union[
            Dict[str, Tensor], str, Sequence[Union[Dict[str, Tensor], str]]
        ],
        device: str = "cpu",
        prediction_columns: Optional[
            Union[Sequence[str], Sequence[Optional[Sequence[str]]]]
        ] = None,
    ):
        """Construct DeploymentModule.

        Arguments:
            model_config: A model configuration file, or a list of them.
            state_dict: A state dict for the model (or a path to it, e.g. a
                Lightning `.ckpt`), or a list of them.
            device: The computational device to use. Defaults to "cpu".
            prediction_columns: Column names for each column in model output
                (all tasks, in task order); for several models, a list with
                one entry (or None) per model. Defaults to the models'
                prediction labels.
        """
        super().__init__(name=__name__, class_name=self.__class__.__name__)
        configs: List[Any]
        state_dicts: List[Any]
        columns: List[Any]
        if isinstance(model_config, (list, tuple)):
            configs = list(model_config)
            assert isinstance(state_dict, (list, tuple)) and len(
                state_dict
            ) == len(configs), "Give one state_dict per model_config."
            state_dicts = list(state_dict)
            columns = list(prediction_columns or [None] * len(configs))
            assert len(columns) == len(
                configs
            ), "Give one prediction_columns entry (or None) per model."
        else:
            configs = [model_config]
            state_dicts = [state_dict]
            columns = [prediction_columns]

        self._device = device
        self.models: List[Model] = []
        self.prediction_columns_per_model: List[List[str]] = []
        for config, weights, model_columns in zip(
            configs, state_dicts, columns
        ):
            model = self._load_model(model_config=config, state_dict=weights)
            # Set model to inference mode and move it to the device
            model.inference()
            model.train(mode=False)
            model.to(device)
            self.models.append(model)
            self.prediction_columns_per_model.append(
                self._resolve_prediction_columns(model_columns, model)
            )
        self.model = self.models[0]
        self.prediction_columns = self.prediction_columns_per_model[0]

    @abstractmethod
    def __call__(self, input_data: Any) -> Any:
        """Define here how the module acts on a file/data stream."""

    def _load_model(
        self,
        model_config: Union[ModelConfig, str],
        state_dict: Union[Dict[str, Tensor], str],
    ) -> Model:
        """Load `Model` from config and insert learned weights."""
        model = Model.from_config(model_config, trust=True)
        if isinstance(state_dict, str) and state_dict.endswith(".ckpt"):
            # Lightning checkpoints hold more than tensors; map to the CPU so
            # that GPU-trained checkpoints load anywhere
            ckpt = load(state_dict, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(state_dict)
        return model

    def _resolve_prediction_columns(
        self,
        prediction_columns: Optional[Union[Sequence[str], str]],
        model: Optional[Model] = None,
    ) -> List[str]:
        if prediction_columns is None:
            return list((model or self.model).prediction_labels)
        if isinstance(prediction_columns, str):
            return [prediction_columns]
        return list(prediction_columns)

    def _inference(
        self, data: Union[Data, Batch], model: Optional[Model] = None
    ) -> List[np.ndarray]:
        """Apply a model to a single event or batch of events `data`.

        Args:
            data: A `Data` or ``Batch` object -
                  either a single output of a `GraphDefinition` or a batch of
                  them.
            model: The model to apply; defaults to the (first) model.

        Returns:
            A List of numpy arrays, each representing the output from the
            `Task`s that the model contains.
        """
        # Perform inference
        output = (model or self.model)(data=data)
        # Loop over tasks in model and transform to numpy
        for k in range(len(output)):
            output[k] = output[k].detach().cpu().numpy()
        return output
