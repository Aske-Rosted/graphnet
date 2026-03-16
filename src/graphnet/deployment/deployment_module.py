"""Class(es) for deploying GraphNeT models in icetray as I3Modules."""

from abc import abstractmethod
from typing import Any, List, Union, Dict

import numpy as np
<<<<<<< HEAD
from torch import Tensor, load
=======
from torch import Tensor, load, device
>>>>>>> bundle_reco
from torch_geometric.data import Data, Batch

from graphnet.models import Model
from graphnet.utilities.config import ModelConfig
from graphnet.utilities.logging import Logger


class DeploymentModule(Logger):
    """Base DeploymentModule for GraphNeT.

    Contains standard methods for loading models doing inference with them.
    Experiment-specific implementations may overwrite methods and should define
    `__call__`.
    """

    def __init__(
        self,
        model_config: Union[List[ModelConfig], List[str], ModelConfig, str],
        state_dict: Union[Dict[str, Tensor], str],
        device: str = "cpu",
        prediction_columns: Union[List[str], None] = None,
        multiple_models: bool = False,
    ):
        """Construct DeploymentModule.

        Arguments:
            model_config: A model configuration file.
            state_dict: A state dict for the model.
            device: The computational device to use. Defaults to "cpu".
            prediction_columns: Column names for each column in model output.
        """
        super().__init__(name=__name__, class_name=self.__class__.__name__)
        # Set Member Variables
        self.multiple_models = multiple_models
        if multiple_models == False:
            self.model = self._load_model(
                model_config=model_config, state_dict=state_dict
            )
        elif multiple_models == True:
            self.models = []
            for mc, sd in zip(model_config, state_dict):
                self.models.append(self._load_model(model_config=mc, state_dict=sd))

        self.prediction_columns = self._resolve_prediction_columns(
            prediction_columns
        )
        if multiple_models == False:
            # Set model to inference mode.
            self.model.inference()
            self.model.train(mode=False)

            # Move model to device
            self.model.to(device)

        elif multiple_models == True:
            for model in self.models:
                model.inference()
                model.train(mode=False)
                model.to(device)

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
            ckpt = load(state_dict, map_location=device('cpu'))
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(state_dict)
        return model

    def _resolve_prediction_columns(
        self, prediction_columns: Union[List[str], None]
    ) -> List[str]:
        if self.multiple_models == False:
            if prediction_columns is not None:
                if isinstance(prediction_columns, str):
                    prediction_columns = [prediction_columns]
                else:
                    prediction_columns = prediction_columns
            else:
                prediction_columns = self.model.prediction_labels
            return prediction_columns
        elif self.multiple_models == True:
            resolved_prediction_columns = []
            for i, model in enumerate(self.models):
                if prediction_columns is not None:
                    if isinstance(prediction_columns[i], str):
                        resolved_prediction_columns.append([prediction_columns[i]])
                    else:
                        resolved_prediction_columns.append(prediction_columns[i])
                else:
                    resolved_prediction_columns.append(model.prediction_labels)
            return resolved_prediction_columns

    def _inference(self, data: Union[Data, Batch]) -> List[np.ndarray]:
        """Apply model to a single event or batch of events `data`.

        Args:
            data: A `Data` or ``Batch` object -
                  either a single output of a `GraphDefinition` or a batch of
                  them.

        Returns:
            A List of numpy arrays, each representing the output from the
            `Task`s that the model contains.
        """
        if self.multiple_models == False:
            # Perform inference
            output = self.model(data=data)
            # Loop over tasks in model and transform to numpy
            for k in range(len(output)):
                output[k] = output[k].detach().numpy()
            return output
        elif self.multiple_models == True:
            outputs = []
            for _, model in enumerate(self.models):
                output = model(data=data[_])
                for k in range(len(output)):
                    output[k] = output[k].detach().numpy()
                outputs.append(output)
            return outputs
