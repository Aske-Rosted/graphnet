"""IceCube I3InferenceModule.

Contains functionality for writing model predictions to i3 files.
"""

from typing import List, Union, Optional, TYPE_CHECKING, Dict, Any

import numpy as np
from torch_geometric.data import Data, Batch

from torch.cuda import OutOfMemoryError
from time import sleep
import torch
import gc
import os
from time import time


from graphnet.utilities.config import ModelConfig
from graphnet.deployment import DeploymentModule
from graphnet.data.extractors.icecube import I3PulseExtractor
from graphnet.utilities.imports import has_icecube_package

if has_icecube_package() or TYPE_CHECKING:
    from icecube.icetray import (
        I3Frame,
    )  # pyright: reportMissingImports=false
    from icecube import (
        dataclasses,
    )
    from icecube.dataclasses import (
        I3Double,
        I3Particle,
        I3Direction,
        I3Position,
    )  # pyright: reportMissingImports=false


class I3InferenceModule(DeploymentModule):
    """General class for inference on i3 frames."""

    def __init__(
        self,
        pulsemap_extractor: Union[
            List[I3PulseExtractor], I3PulseExtractor
        ],
        model_config: Union[List[ModelConfig], List[str], ModelConfig, str],
        state_dict: Union[List[str], str],
        model_name: Union[List[str], str],
        num_threads: int = 1,
        gcd_file: Optional[str] = None,
        features: Optional[List[str]] = None,
        prediction_columns: Optional[Union[List[str], None]] = None,
        pulsemap: Optional[str] = None,
        multiple_models: bool = False,
        key_name: Optional[str] = None,
        requirements: Optional[callable] = None,
        device: Optional[str] = "cpu",
        batch_size: Optional[int] = 1,
        skip: bool = False,
        inference_speed_check: bool = False,
    ):
        """General class for inference on I3Frames (physics).

        Arguments:
            pulsemap_extractor: The extractor used to extract the pulsemap.
            model_config: The ModelConfig (or path to it) that summarizes the
                            model used for inference.
                          
            state_dict: Path to state_dict containing the learned weights.
            model_name: The name used for the model. Will help define the
                        named entry in the I3Frame. E.g. "dynedge".
            gcd_file: path to associated gcd file.
            features: the features of the pulsemap that the model is expecting.
            prediction_columns: column names for the predictions of the model.
                               Will help define the named entry in the I3Frame.
                                E.g. ['energy_reco']. Optional.
            pulsemap: the pulsmap that the model is expecting as input.
            multiple_models: process multiple models with the same feature set at once.
            key_name: The name used for the key in the I3Frame. Will help define the
                     named entry in the I3Frame. E.g. "dynedge_predictions".
        """
        super().__init__(
            model_config=model_config,
            state_dict=state_dict,
            prediction_columns=prediction_columns,
            device=device,
            multiple_models=multiple_models,
        )
        # Checks
        if gcd_file is not None:
            assert isinstance(gcd_file, str), "gcd_file must be string"
        else:
            self.warning("No GCD file provided. " "Expected to be set later")

        # Set Member Variables
        if isinstance(pulsemap_extractor, list):
            self._i3_extractors = pulsemap_extractor
        else:
            self._i3_extractors = [pulsemap_extractor]
        
        # All
        if self.multiple_models == True:
            self.features_list = []
            for model in self.models:
                self.features_list.append(model._graph_definition._input_feature_names)
        elif features is None:
            features = self.model._graph_definition._input_feature_names

        if self.multiple_models == True:
            self._graph_definitions = [model._graph_definition for model in self.models]
            self._graph_definitions = [graph_definition.to(device) for graph_definition in self._graph_definitions]
        else:
            self._graph_definition = self.model._graph_definition
            self._graph_definition.to(device)
        
        
        self._pulsemap = pulsemap
        self._gcd_file = gcd_file
        self.model_name = model_name
        self._features = features
        self._requirements = requirements
        self._device = device
        self._batch_size = batch_size
        self._skip = skip
        self._num_threads = num_threads
        self._inference_speed_check = inference_speed_check
        self._multiple_models = multiple_models
        self._key_name = key_name
        # Set GCD file for pulsemap extractor
        if gcd_file is not None:
            for i3_extractor in self._i3_extractors:
                i3_extractor.set_gcd(i3_file="", gcd_file=self._gcd_file)

    def __call__(self, frame: I3Frame) -> bool:
        """Write predictions from model to frame."""
        # Check requirements
        torch.set_num_threads(self._num_threads)
        if not self._requirements is None:
            if not self._check_requirements(frame=frame):
                if self._skip:
                    return False
                predictions = np.repeat(
                    [np.nan], len(self.prediction_columns)
                ).reshape(-1, len(self.prediction_columns))
                dim = self._check_dimensions(predictions=predictions)
                data = self._create_dictionary(
                    dim=dim, predictions=predictions
                )
                self._add_to_frame(frame=frame, data=data)
                return True
        # inference
        memory_watch = False
        if self._inference_speed_check is True:
            # create log file if it does not exist
            data_repr_start = time()
        try:
            if not self.multiple_models:
                data = self._create_data_representation(frame=frame).to(
                    self._device
                )
                if self._inference_speed_check is True:
                    data_repr_end = time()
                    data_repr_time = data_repr_end - data_repr_start
                    inference_start = time()
                predictions = self._apply_model(data=data)
            else:
                features = self._extract_feature_array_from_frame(frame=frame)
                if self._inference_speed_check is True:
                    data_repr_end = time()
                    data_repr_time = data_repr_end - data_repr_start
                    inference_start = time()
                model_input_data = []
                for _,graph_definition in enumerate(self._graph_definitions):
                    data = graph_definition(
                        input_features=features[_],
                        input_feature_names=self.features_list[_],
                    )
                    model_input_data.append(Batch.from_data_list([data]))

                predictions = self._apply_model(data=model_input_data)

            if self._inference_speed_check is True:
                inference_end = time()
                inference_time = inference_end - inference_start
                self._logger.info(
                    f"Data representation time: {data_repr_time:.4f} s\n"
                    f"Inference time: {inference_time:.4f} s\n"
                )

        except OutOfMemoryError:
            self.error(
                "Out of memory error. "
                "Please reduce batch size or model size."
                "trying to run on cpu."
            )
            save_device = self._device
            self._device = "cpu"
            self.model.to(self._device)
            data = self._create_data_representation(frame=frame)
            if self._inference_speed_check is True:
                data_repr_end = time()
                data_repr_time = data_repr_end - data_repr_start
                inference_start = time()
            predictions = self._apply_model(data=data)
            if self._inference_speed_check is True:
                inference_end = time()
                inference_time = inference_end - inference_start
                self._logger.info(
                    f"Data representation time: {data_repr_time:.4f} s\n"
                    f"Inference time: {inference_time:.4f} s\n"
                )
            self._device = save_device
            memory_watch = True
        del data

        if self._inference_speed_check is True:
            write_start = time()
        # Check dimensions of predictions and prediction columns
        dim = self._check_dimensions(predictions=predictions)

        # Build Dictionary from predictions
        data = self._create_dictionary(dim=dim, predictions=predictions)
        del predictions
        del dim

        # Submit Dictionary to frame
        self._add_to_frame(frame=frame, data=data)
        del data
        if self._inference_speed_check is True:
            write_end = time()
            write_time = write_end - write_start
            self._logger.info(f"Write time: {write_time:.4f} s\n")
            total_time = data_repr_time + inference_time + write_time
            self._logger.info(f"Total time: {total_time:.4f} s\n")
        self._clean_device()

        if memory_watch:
            self.warning("Memory watch triggered. Trying to return to device.")
            self.model.to(self._device)
        return True
    

    def _check_dimensions(self, predictions: np.ndarray) -> int:
        if len(predictions.shape) > 1:
            dim = predictions.shape[1]
        else:
            dim = len(predictions)
        try:
            assert dim == len(self.prediction_columns)
        except AssertionError as e:
            self.error(
                f"predictions have shape {dim} but"
                f"prediction columns have [{self.prediction_columns}]"
            )
            raise e

        # assert predictions.shape[0] == 1
        return dim

    def _create_dictionary(
        self, dim: int, predictions: np.ndarray
    ) -> Dict[str, Any]:
        """Transform predictions into a dictionary."""
        data = {}
        for i in range(dim):
            data[self.model_name + "_" + self.prediction_columns[i]] = (
                I3Double(float(predictions[i]))
            )

            # try:
            #     assert len(predictions[:, i]) == 1
            #     data[
            #         self.model_name + "_" + self.prediction_columns[i]
            #     ] = I3Double(float(predictions[:, i][0]))
            # except IndexError:
            #     data[
            #         self.model_name + "_" + self.prediction_columns[i]
            #     ] = I3Double(predictions[0])
        return data

    def _apply_model(self, data: Data) -> np.ndarray:
        """Apply model to `Data` and case-handling."""
        if data is not None:
            predictions = self._inference(data)
            if isinstance(predictions, list):
                predictions = np.concatenate(
                    [pred.flatten() for pred in predictions]
                )
                # warn_bool = len(predictions) > 1
                # predictions = predictions[0]
                # if warn_bool:
                #     self.warning(
                #         f"{self.__class__.__name__} assumes one Task "
                #         f"but got {len(predictions)}. Only the first will"
                #         " be used."
                #     )
        else:
            self.warning(
                "At least one event has no pulses "
                " - padding {self.prediction_columns} with NaN."
            )
            predictions = np.repeat(
                [np.nan], len(self.prediction_columns)
            ).reshape(-1, len(self.prediction_columns))
        return predictions

    def _create_data_representation(self, frame: I3Frame) -> Data:
        """Process Physics I3Frame into graph."""
        # Extract features
        input_features = self._extract_feature_array_from_frame(frame)
        # Prepare graph data
        if len(input_features) > 0:
            data = self._graph_definition(
                input_features=input_features,
                input_feature_names=self._features,
            )
            return Batch.from_data_list([data.to(self._device)])
        else:
            return None

    def _extract_feature_array_from_frame(self, frame: I3Frame) -> np.array:
        """Apply the I3FeatureExtractors to the I3Frame.

        Arguments:
            frame: Physics I3Frame (PFrame)

        Returns:
            array with pulsemap
        """
        features = None
        for i3extractor in self._i3_extractors:
            feature_dict = i3extractor(frame)
            if self.multiple_models == False:
                features_pulsemap = np.array(
                    [feature_dict[key] for key in self._features]
                ).T
                if features is None:
                    features = features_pulsemap
                else:
                    features = np.concatenate(
                        (features, features_pulsemap), axis=0
                    )
                return features
            if self.multiple_models == True:
                features_array = []
                for feature_list in self.features_list:
                    features = None
                    features_pulsemap = np.array(
                        [feature_dict[key] for key in feature_list]
                    ).T
                    if features is None:
                        features = features_pulsemap
                    else:
                        features = np.concatenate(
                            (features, features_pulsemap), axis=0
                        )
                    features_array.append(features)
            return features_array


    def _add_to_frame(self, frame: I3Frame, data: Dict[str, Any]) -> None:
        """Add every field in data to I3Frame.

        Arguments:
            frame: I3Frame (physics)
            data: Dictionary containing content that will be written to frame.

        Returns:
            frame: Same I3Frame as input, but with the new entries
        """
        assert isinstance(
            data, dict
        ), f"data must be of type dict. Got {type(data)}"
        for key in data.keys():
            if key not in frame:
                frame.Put(key, data[key])
        return

    def _check_requirements(self, frame: I3Frame) -> bool:
        """Check if requirements are met."""
        for requirement in self._requirements:
            res = requirement(frame)
            if not res:
                return res
        return True

    def _clean_device(self) -> None:
        """Clean up the device."""
        if "cuda" in self._device:
            torch.cuda.empty_cache()


class I3ParticleInferenceModule(I3InferenceModule):
    """I3InferenceModule for I3Particle data."""

    def __init__(
        self,
        directions: List[str],
        time: str,
        energy: str,
        positions: List[str],
        shift_time: bool = False,
        **kwargs,
    ):
        """Initialize the I3ParticleInferenceModule."""
        super().__init__(**kwargs)

        self._directions = [
            self.model_name + "_" + dirs for dirs in directions
        ]
        assert (
            len(self._directions) == 2 or len(self._directions) == 3
        ), "directions must be a list of 2 or 3 elements"
        self._time = self.model_name + "_" + time
        self._energy = self.model_name + "_" + energy
        self._positions = [self.model_name + "_" + pos for pos in positions]
        self._shift_time = shift_time
        assert (
            len(self._positions) == 3
        ), "positions must be a list of 3 elements"

    def _get_min_time(self, frame: I3Frame) -> float:
        """Get the minimum time of the first pulse in the frame."""
        min_time = np.inf
        doms = frame[self._pulsemap].apply(frame).values()
        # seach for the minimum time
        for dom in doms:
            if dom[0].time < min_time:
                min_time = dom[0].time

        return min_time

    def _add_to_frame(self, frame, data):
        """Create the I3Particle and add it to the frame."""

        particle = I3Particle()

        directions = [data[k].value for k in self._directions]
        # drop the directions from the data dictionary
        for dirs in self._directions:
            del data[dirs]

        positions = [data[k].value for k in self._positions]
        # drop the positions from the data dictionary
        for pos in self._positions:
            del data[pos]

        particle.dir = I3Direction(*directions)

        if self._shift_time:
            # Shift time to be relative to the first pulse
            shift_time = (
                self._get_min_time(frame)
                - frame["CVStatistics"].min_pulse_time
            )
            particle.time = data[self._time].value + shift_time
        else:
            particle.time = data[self._time].value
        # drop the time from the data dictionary
        if self._time in data:
            del data[self._time]

        particle.energy = data[self._energy].value
        # drop the energy from the data dictionary
        if self._energy in data:
            del data[self._energy]
        # Set the position of the particle
        particle.pos = I3Position(*positions)
        particle.shape = I3Particle.ParticleShape.InfiniteTrack
        particle_name = self.model_name + "_particle"
        particle.fit_status = I3Particle.FitStatus.OK
        # Add the particle to the frame
        if particle_name not in frame:
            frame.Put(particle_name, particle)

        # for all the other values in data, add them to an I3Dictionary
        super()._add_to_frame(frame=frame, data=data)
        return
