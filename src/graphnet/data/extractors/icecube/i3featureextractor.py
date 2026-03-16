#!/usr/bin/env python
"""I3Extractor class(es) for extracting specific, reconstructed features."""


from typing import TYPE_CHECKING, Any, Dict, List
from .i3extractor import I3Extractor
from graphnet.data.extractors.icecube.utilities.frames import (
    get_om_keys_and_pulseseries,
)
from graphnet.utilities.imports import has_icecube_package

if has_icecube_package() or TYPE_CHECKING:
    from icecube import icetray  # pyright: reportMissingImports=false
    from icecube import phys_services
    from icecube import dataclasses

import pandas as pd
from collections import defaultdict 

class I3FeatureExtractorLegacy(I3Extractor):
    """Base class for extracting specific, reconstructed features."""

    def __init__(self, pulsemap: str, quantiles_time: List[Any], quantiles_charge: List[Any], 
                 is_data: bool = False):
        """Construct I3FeatureExtractorLegacy.

        Args:
            pulsemap: Name of the pulse (series) map for which to extract
                reconstructed features.
            quantiles_time: 
            quantiles_charge
        """
        # Member variable(s)
        self._pulsemap = pulsemap
        self._quantiles_time = quantiles_time
        self._quantiles_charge = quantiles_charge
        self._is_data = is_data

        # Base class constructor
        super().__init__(pulsemap)

class I3FeatureExtractorLegacyIceCube(I3FeatureExtractorLegacy):
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

class I3FeatureExtractorLegacyIceCube86(I3FeatureExtractorLegacy):
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
            "adjusted_time": [],
            "width": [],
            "dom_qtot": [],
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
            "r": [],
            "residual": [],
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

        #print(frame["I3EventHeader"].event_id, frame["I3EventHeader"].sub_event_id, "graphnet run")
        #print(len(om_keys), 'graphnet')

        if self._is_data == False:
            leading = self._get_leading_particle(frame=frame)
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
                #print(pulses)

            for _,pulse in enumerate(pulses):

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

                if self._is_data == False:
                    output['r'].append(
                        phys_services.I3Calculator.closest_approach_distance(leading, self._gcd_dict[om_key].position)
                        )
                    output['residual'].append(
                        phys_services.I3Calculator.time_residual(leading, self._gcd_dict[om_key].position, getattr(pulse, "time", padding_value)) 
                    )
                else:
                    output['r'].append(padding_value)
                    output['residual'].append(padding_value)

                # Pulse flags
                flags = getattr(pulse, "flags", padding_value)
                if flags == padding_value:
                    output["hlc"].append(padding_value)
                    output["awtd"].append(padding_value)
                else:
                    output["hlc"].append((pulse.flags >> 0) & 0x1)  # bit 0
                    output["awtd"].append(self._parse_awtd_flag(pulse))

        # Convert Event Info Into Dataframe
        evt_pulses = pd.DataFrame({"charge": output['charge'],
                                    "dom_time": output['dom_time'],
                                    "width": output['width'],
                                    "dom_x": output['dom_x'],
                                    "dom_y": output['dom_y'],
                                    "dom_z": output['dom_z'],
                                    "pmt_area": output['pmt_area'],
                                    "rde": output['rde'],
                                    "is_bright_dom": output['is_bright_dom'],
                                    "is_bad_dom": output['is_bad_dom'],
                                    "is_saturated_dom": output['is_saturated_dom'],
                                    "is_errata_dom": output['is_errata_dom'],
                                    "event_time": output['event_time'],
                                    "hlc": output['hlc'],
                                    "awtd": output['awtd'],
                                    "string": output['string'],
                                    "pmt_number": output['pmt_number'],
                                    "dom_number": output['dom_number'],
                                    "dom_type": output['dom_type'],
                                    "r": output['r'],
                                    "residual": output['residual'],},)
        
        # Produce Quantile Information of Each DOM
        t_quantiles = evt_pulses.groupby(["string", "dom_number"])['dom_time'].quantile(self._quantiles_time).unstack().reset_index()
        for quant in self._quantiles_time:
            t_quantiles = t_quantiles.rename(columns={quant: f't{int(1000*quant)}'})

        evt_pulses['qcumsum'] = evt_pulses.groupby(["string", "dom_number"])['charge'].cumsum()	
        q_quantiles = evt_pulses.groupby(["string", "dom_number"])['qcumsum'].quantile(self._quantiles_charge).unstack().reset_index()
        evt_pulses.drop(columns=['qcumsum'], inplace=True)
        for quant in self._quantiles_charge:
            q_quantiles = q_quantiles.rename(columns={quant: f'q{int(1000*quant)}'})	
        q_total = evt_pulses.groupby(["string", "dom_number"], as_index=False)['charge'].sum()
        # Extrac the Minimum Pulse Time of Each Dom
        min_times = evt_pulses.loc[evt_pulses.groupby(["string", "dom_number"], as_index=True)['dom_time'].idxmin()]

        min_times = min_times.merge(t_quantiles, on = ["string", "dom_number"])
        min_times = min_times.merge(q_quantiles, on = ["string", "dom_number"])

        min_times['adjusted_time'] = min_times["dom_time"] - min_times["dom_time"].min()
 
        total_pulses = evt_pulses.groupby(["string", "dom_number"], as_index=False)['charge'].size()

        min_times['dom_qtot'] = q_total['charge']
        min_times['dom_qtot_exc'] = q_total['charge']
        min_times['total_pulses'] = total_pulses['size']

        bright_doms = min_times['dom_qtot']/frame['Homogenized_QTot_New'].value >= .4

        min_times['bright_dom'] = bright_doms.to_numpy(dtype=float)

        bad_doms = (min_times['is_errata_dom'] == 1) | (min_times['is_saturated_dom'] == 1)
        t_name_keys = [f't{int(1000*quant)}' for quant in self._quantiles_time]
        q_name_keys = [f'q{int(1000*quant)}' for quant in self._quantiles_charge]

        for t_name in t_name_keys:
            min_times[t_name] = min_times[t_name] - min_times["dom_time"].min()

        # Remove This
        #min_times.loc[bad_doms, t_name_keys] = -100
        #min_times.loc[bad_doms, q_name_keys] = -100
        min_times.loc[bad_doms, 'dom_qtot_exc'] = -100


        #print(min_times)
        output = min_times.to_dict(orient='list')
        #print(min_times)
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

    # Error Getting this for a certain set
    def _get_leading_particle(
            self, frame: "icetray.I3Frame",
        ):

        try:
            tracklist = frame['MMCTrackList']

            max_energy = -1
            max_particle = tracklist[0]
            for particle in tracklist:
                if particle.Ei > max_energy:
                    max_energy = particle.Ei
                    max_particle = particle

            return max_particle.particle
        except:
            print('no mmctracklist')
            mctree = frame['I3MCTree_preMuonProp']
            return mctree[1]