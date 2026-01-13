"""I3Extractor class(es) for extracting specific, reconstructed features."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional
from .i3extractor import I3Extractor
from graphnet.data.extractors.icecube.utilities.frames import (
    get_om_keys_and_pulseseries,
)
from graphnet.utilities.imports import has_icecube_package

if has_icecube_package() or TYPE_CHECKING:
    from icecube import (
        icetray,
        dataclasses,
    )  # pyright: reportMissingImports=false

import numpy as np


class I3PulseLevelExtractor(I3Extractor):
    """Base class for extracting specific, reconstructed features."""

    def __init__(
        self,
        pulsemap: str,
        exclude: list = [None],
        extractor_name: Optional[str] = None,
    ):
        """Construct I3PulseLevelExtractor.

        Args:
            pulsemap: Name of the pulse (series) map for which to extract
                reconstructed features.
            exclude: List of keys to exclude from the extracted data.
            extractor_name: Name of the extractor.
        """
        # Member variable(s)
        self._pulsemap = pulsemap
        if extractor_name is None:
            extractor_name = pulsemap

        # Base class constructor
        super().__init__(extractor_name, exclude=exclude)


class I3FeatureExtractor(I3PulseLevelExtractor):
    """Old class now contained in I3PulseLevelExtractor."""

    def __init__(self, pulsemap: str, exclude: list = [None]):
        """Construct I3FeatureExtractor.

        Args:
            pulsemap: Name of the pulse (series) map for which to extract
                reconstructed features.
            exclude: List of keys to exclude from the extracted data.
        """
        self.warning_once(
            "I3FeatureExtractor is deprecated and will be removed in a future release. Please use I3PulseLevelExtractor instead."
        )
        super().__init__(pulsemap, exclude=exclude)


class I3FeatureExtractorIceCube86(I3PulseLevelExtractor):
    """Class for extracting reconstructed features for IceCube-86."""

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, List[Any]]:
        """Extract reconstructed features from `frame`.

        Args:
            frame: Physics (P) I3-frame from which to extract reconstructed
                features.

        Returns:
            Dictionary of reconstructed features for all pulses in `pulsemap`,
                in pure-python format.
        """
        padding_value: float = -1.0
        output: Dict[str, List[Any]] = {
            "charge": [],
            "dom_time": [],
            "width": [],
            "dom_x": [],
            "dom_y": [],
            "dom_z": [],
            "pmt_area": [],
            "rde": [],
            "is_bright_dom": [],
            "is_bad_dom": [],
            "is_saturated_dom": [],
            "is_errata_dom": [],
            "event_time": [],
            "hlc": [],
            "awtd": [],
            "string": [],
            "pmt_number": [],
            "dom_number": [],
            "dom_type": [],
        }
        # Get OM data
        if self._pulsemap in frame:
            om_keys, data = get_om_keys_and_pulseseries(
                frame,
                self._pulsemap,
                self._calibration,
            )
        else:
            self.warning_once(f"Pulsemap {self._pulsemap} not found in frame.")
            return output

        # Added these :
        bright_doms = None
        bad_doms = None
        saturation_windows = None
        calibration_errata = None
        if "BrightDOMs" in frame:
            bright_doms = frame.Get("BrightDOMs")

        if "BadDomsList" in frame:
            bad_doms = frame.Get("BadDomsList")

        if "SaturationWindows" in frame:
            saturation_windows = frame.Get("SaturationWindows")

        if "CalibrationErrata" in frame:
            calibration_errata = frame.Get("CalibrationErrata")

        event_time = frame["I3EventHeader"].start_time.mod_julian_day_double

        for om_key in om_keys:
            # Common values for each OM
            x = self._gcd_dict[om_key].position.x
            y = self._gcd_dict[om_key].position.y
            z = self._gcd_dict[om_key].position.z
            area = self._gcd_dict[om_key].area
            rde = self._get_relative_dom_efficiency(
                frame, om_key, padding_value
            )

            string = om_key[0]
            dom_number = om_key[1]
            pmt_number = om_key[2]
            dom_type = self._gcd_dict[om_key].omtype

            # DOM flags
            if bright_doms:
                is_bright_dom = 1 if om_key in bright_doms else 0
            else:
                is_bright_dom = int(padding_value)

            if bad_doms:
                is_bad_dom = 1 if om_key in bad_doms else 0
            else:
                is_bad_dom = int(padding_value)

            if saturation_windows:
                is_saturated_dom = 1 if om_key in saturation_windows else 0
            else:
                is_saturated_dom = int(padding_value)

            if calibration_errata:
                is_errata_dom = 1 if om_key in calibration_errata else 0
            else:
                is_errata_dom = int(padding_value)

            # Loop over pulses for each OM
            pulses = data[om_key]
            for pulse in pulses:
                output["charge"].append(
                    getattr(pulse, "charge", padding_value)
                )
                output["dom_time"].append(
                    getattr(pulse, "time", padding_value)
                )
                output["width"].append(getattr(pulse, "width", padding_value))
                output["pmt_area"].append(area)
                output["rde"].append(rde)
                output["dom_x"].append(x)
                output["dom_y"].append(y)
                output["dom_z"].append(z)
                # ID's
                output["string"].append(string)
                output["pmt_number"].append(pmt_number)
                output["dom_number"].append(dom_number)
                output["dom_type"].append(dom_type)
                # DOM flags
                output["is_bright_dom"].append(is_bright_dom)
                output["is_bad_dom"].append(is_bad_dom)
                output["is_saturated_dom"].append(is_saturated_dom)
                output["is_errata_dom"].append(is_errata_dom)
                output["event_time"].append(event_time)

                # Pulse flags
                flags = getattr(pulse, "flags", padding_value)
                if flags == padding_value:
                    output["hlc"].append(padding_value)
                    output["awtd"].append(padding_value)
                else:
                    output["hlc"].append((pulse.flags >> 0) & 0x1)  # bit 0
                    output["awtd"].append(self._parse_awtd_flag(pulse))

        return output

    def _get_relative_dom_efficiency(
        self, frame: "icetray.I3Frame", om_key: int, padding_value: float
    ) -> float:
        if (
            "I3Calibration" in frame
        ):  # Not available for e.g. mDOMs in IceCube Upgrade
            rde = frame["I3Calibration"].dom_cal[om_key].relative_dom_eff
        else:
            try:
                assert self._calibration is not None
                rde = self._calibration.dom_cal[om_key].relative_dom_eff
            except:  # noqa: E722
                rde = padding_value
        return rde

    def _parse_awtd_flag(
        self, pulse: Any, fadc_min_width_ns: float = 6.0
    ) -> bool:
        """Parse awtd flag from pulse width.

        Returns True if the pulse was readout using the awtd digitizer.

        Method by Tom Stuttard.

        Notes from Tom:
        Function to read the bits of the pulse flags and unpack them into
        meaningful info Using pulse width rather than flags to separate FADC vs
        ATWD pulses, due to a known issue.
        https://github.com/icecube/icetray/issues/2721 Note that the issue
        states to use 8ns, but I have found that actually 6ns is correct.
        """
        # Use pulse width to check whether a pulse is
        # (a) FADC-only, or
        # includes ATWD (and probably also FADC)
        return pulse.width < (fadc_min_width_ns * icetray.I3Units.ns)


class I3PulseOriginLabels(I3PulseLevelExtractor):
    """Class for extracting MCPE labels for IceCube-86."""

    def __init__(
        self,
        pulsemap: str,
        exclude: list = [None],
        extractor_name: str = "PulseOrigin",
        time_window: float = 10.0,
        mctree: str = "I3MCTree",
        mcpe_map: str = "I3MCPESeriesMapWithoutNoise",
        mcpe_map_id: str = "I3MCPESeriesMapParticleIDMap",
    ):
        """Construct I3PulseOriginLabels.

        Args:
            pulsemap: Name of the pulse (series) map for which to extract
                reconstructed features.
            exclude: List of keys to exclude from the extracted data.
            extractor_name: Name of the extractor.
            time_window: Time window (in ns) around each pulse to consider
                MCPEs for label assignment.
            mctree: Name of the MCTree in the I3 frame.
            mcpe_map: Name of the MCPE series map in the I3 frame.
            mcpe_map_id: Name of the MCPE series map particle ID map in the I3 frame.
        """
        super().__init__(pulsemap, exclude, extractor_name)
        self._time_window = time_window
        self._mctree = mctree
        self._mcpe_map = mcpe_map
        self._mcpe_map_id = mcpe_map_id

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, List[Any]]:
        """Extract MCPE labels from `frame`.

        Args:
            frame: Physics (P) I3-frame from which to extract MCPE labels.

        Returns:
            Dictionary of MCPE labels for all pulses in `pulsemap`,
                in pure-python format.
        """
        output: Dict[str, List[Any]] = {
            "charge": [],
            "dom_time": [],
            "dom_x": [],
            "dom_y": [],
            "dom_z": [],
            "neutrino_fraction": [],
            "neutrino_npe_fraction": [],
            "npe": [],
            "pulse_count": [],
            "noise_hit": [],
            "trackness": [],
            "overlap_count": [],
            "min_time_delta": [],
        }

        # Get OM data
        if self._pulsemap in frame:
            om_keys, data = get_om_keys_and_pulseseries(
                frame,
                self._pulsemap,
                self._calibration,
            )
        else:
            self.warning_once(f"Pulsemap {self._pulsemap} not found in frame.")
            return output

        for om_key in om_keys:
            # Loop over pulses for each OM
            pulses = data[om_key]
            pulse_times, pulse_charges = self._get_pulse_info(pulses)
            npe_list, times, nu_bool, track_like_list = self._get_mcpe_info(
                frame, om_key
            )
            time_distance_matrix = pulse_times[:, None] - times[None, :]
            weight_matrix = self._get_gaussian_weight(time_distance_matrix) * (
                np.abs(time_distance_matrix) <= self._time_window
            )

            with np.errstate(invalid="ignore"):
                weight_matrix /= np.sum(weight_matrix, axis=0, keepdims=True)
                weight_matrix = np.nan_to_num(weight_matrix, nan=0.0)
                pulse_counts = np.sum(weight_matrix, axis=1)
                neutrino_fractions = (weight_matrix @ nu_bool) / pulse_counts
                neutrino_npe_fractions = (
                    weight_matrix @ (npe_list * nu_bool)
                ) / (weight_matrix @ npe_list)
                total_npe = weight_matrix @ npe_list
                trackness = weight_matrix @ track_like_list / pulse_counts
                min_time_deltas = (
                    np.min(np.abs(time_distance_matrix), axis=1)
                    if len(times) > 0
                    else np.array([np.nan] * len(pulse_times))
                )

                # Determine how many other mcpe's overlap with any other pulse for a given pulse
                weight_matrix_binary = (weight_matrix > 0).astype(float)
                overlap_counts = weight_matrix_binary @ weight_matrix_binary.T
                np.fill_diagonal(overlap_counts, 0)
                overlap_counts = np.sum(overlap_counts, axis=1)

            output["neutrino_fraction"].extend(neutrino_fractions.tolist())
            output["neutrino_npe_fraction"].extend(
                neutrino_npe_fractions.tolist()
            )
            output["npe"].extend(total_npe.tolist())
            output["pulse_count"].extend(pulse_counts.tolist())
            output["charge"].extend(pulse_charges.tolist())
            output["dom_time"].extend(pulse_times.tolist())
            output["dom_x"].extend(
                [self._gcd_dict[om_key].position.x] * len(pulse_times)
            )
            output["dom_y"].extend(
                [self._gcd_dict[om_key].position.y] * len(pulse_times)
            )
            output["dom_z"].extend(
                [self._gcd_dict[om_key].position.z] * len(pulse_times)
            )
            output["noise_hit"].extend((pulse_counts == 0).tolist())
            output["min_time_delta"].extend(min_time_deltas.tolist())
            output["trackness"].extend(trackness.tolist())
            output["overlap_count"].extend(overlap_counts.tolist())

        return output

    def _get_mcpe_info(
        self, frame: "icetray.I3Frame", om_key: "icetray.OMKey"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Determine the neutrino fraction of a pulse.

        Args:
            pulse: The pulse for which to determine the neutrino fraction.
            mcpe_map: Name of the MCPE series map in the I3 frame.

        Returns:
            Neutrino fraction of the pulse.
        """
        times: list[float] = []
        nu_bool: list[bool] = []
        npe_list: list[float] = []
        track_like_list: list[float] = []
        try:
            mcpe_info = frame[self._mcpe_map][om_key]
        except KeyError:
            return (
                np.array(npe_list),
                np.array(times),
                np.array(nu_bool),
                np.array(track_like_list),
            )
        for i, mcpe in enumerate(mcpe_info):
            try:
                nu_primary = (
                    frame[self._mctree].get_primary(mcpe.ID).is_neutrino
                )
                track_like = frame[self._mctree].get_particle(mcpe.ID).is_track
                nu_bool.append(nu_primary)
            except RuntimeError as e:
                # backup to using the mcpe id map to figure out the parent type, if any part of the mcpe has a neutrino as the primary, we count it as a neutrino mcpe this is a choice, but the information about which part of the mcpe corresponds to which primary is lost.
                if "particle not found" in str(e):
                    ids = [
                        id_p
                        for id_p, indexval in frame[self._mcpe_map_id][
                            om_key
                        ].items()
                        if i in indexval
                    ]
                    bool_val = any(
                        [
                            frame[self._mctree].get_primary(id_p).is_neutrino
                            for id_p in ids
                        ]
                    )
                    track_like = [
                        frame[self._mctree].get_particle(id_p).is_track
                        for id_p in ids
                    ]
                    track_like = sum(track_like) / len(track_like)
                    nu_bool.append(bool_val)
                else:
                    raise e

            times.append(mcpe.time)
            npe_list.append(mcpe.npe)
            track_like_list.append(track_like)

        return (
            np.array(npe_list),
            np.array(times),
            np.array(nu_bool),
            np.array(track_like_list),
        )

    def _get_pulse_info(
        self, pulses: List["icetray.I3RecoPulse"]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create an nd array of pulse times, charge."""
        times, charges = np.array([[p.time, p.charge] for p in pulses]).T
        return times, charges

    def _get_gaussian_weight(
        self, time_distance_matrix: np.ndarray
    ) -> np.ndarray:
        """Create gaussian weight matrix based on time distance matrix."""
        return np.exp(
            -0.5 * (time_distance_matrix / (self._time_window / 2)) ** 2
        )


class I3FeatureExtractorIceCubeDeepCore(I3FeatureExtractorIceCube86):
    """Class for extracting reconstructed features for IceCube-DeepCore."""


class I3FeatureExtractorIceCubeUpgrade(I3FeatureExtractorIceCube86):
    """Class for extracting reconstructed features for IceCube-Upgrade."""

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, List[Any]]:
        """Extract reconstructed features from `frame`.

        Args:
            frame: Physics (P) I3-frame from which to extract reconstructed
                features.

        Returns:
            Dictionary of reconstructed features for all pulses in `pulsemap`,
                in pure-python format.
        """
        output: Dict[str, List[Any]] = {
            "pmt_dir_x": [],
            "pmt_dir_y": [],
            "pmt_dir_z": [],
        }

        # Add features from IceCube86
        output_icecube86 = super().__call__(frame)
        output.update(output_icecube86)

        # Get OM data
        if self._pulsemap in frame:
            om_keys, data = get_om_keys_and_pulseseries(
                frame,
                self._pulsemap,
                self._calibration,
            )
        else:
            self.warning_once(f"Pulsemap {self._pulsemap} not found in frame.")
            return output

        for om_key in om_keys:
            # Common values for each OM
            pmt_dir_x = self._gcd_dict[om_key].orientation.x
            pmt_dir_y = self._gcd_dict[om_key].orientation.y
            pmt_dir_z = self._gcd_dict[om_key].orientation.z

            # Loop over pulses for each OM
            pulses = data[om_key]
            for _ in pulses:
                output["pmt_dir_x"].append(pmt_dir_x)
                output["pmt_dir_y"].append(pmt_dir_y)
                output["pmt_dir_z"].append(pmt_dir_z)

        return output


class I3PulseNoiseTruthFlagIceCubeUpgrade(I3FeatureExtractorIceCube86):
    """Feature extractor class with pulse noise truth flag added."""

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, List[Any]]:
        """Extract reconstructed features from `frame`.

        Args:
            frame: Physics (P) I3-frame from which to extract reconstructed
                features.

        Returns:
            Dictionary of reconstructed features for all pulses in `pulsemap`,
                in pure-python format.
        """
        output: Dict[str, List[Any]] = {
            "truth_flag": [],
        }

        # Add features from IceCubeUpgrade
        output_icecube_upgrade = super().__call__(frame)
        output.update(output_icecube_upgrade)

        # Get OM data
        if self._pulsemap in frame:
            om_keys, data = get_om_keys_and_pulseseries(
                frame,
                self._pulsemap,
                self._calibration,
            )
        else:
            self.warning_once(f"Pulsemap {self._pulsemap} not found in frame.")
            return output

        for om_key in om_keys:
            # Loop over pulses for each OM
            pulses = data[om_key]
            for truth_flag in pulses:
                output["truth_flag"].append(truth_flag)

        return output
