from typing import TYPE_CHECKING, Any, Dict, List
from .i3extractor import I3Extractor
from graphnet.data.extractors.icecube.utilities.frames import (
    get_om_keys_and_pulseseries,
)
from graphnet.utilities.imports import has_icecube_package
from graphnet.data.extractors.icecube.utilities.pulselabeling import *

if has_icecube_package() or TYPE_CHECKING:
    from icecube import icetray  # pyright: reportMissingImports=false
    from icecube import phys_services
    from icecube import dataclasses

import pandas as pd
import numpy as np
from collections import defaultdict


class I3PulseExtractorNugen(I3Extractor):
    """Base class for labeling pulses from laterally spread muons in moun
    bundles."""

    def __init__(
        self,
        pulsemap: str,
        quantiles_time: List[Any] = [0.25, 0.5, 0.75],
        quantiles_charge: List[Any] = [0.25, 0.75, 0.95],
        dom_time_max: float = 100,
        dom_charge_max: float = 50,
    ):
        """Construct I3FeatureExtractor.

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

        # Base class constructor
        super().__init__(pulsemap)


"""
Chain
-> get a list of the muons
-> compute mc pulse array
-> label pulses
-> 
"""


class I3PulseExtractorNugenIceCube86(I3PulseExtractorNugen):
    """Class processing through and labeling pulses."""

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
            "time": [],
            "adjusted_time": [],
            "width": [],
            "dom_qtot": [],
            "dom_x": [],
            "dom_y": [],
            "dom_z": [],
            "dom_hit": [],
            "pmt_area": [],
            "rde": [],
            "r_primary": [],
            "timing_residual_primary": [],
            "is_bright_dom": [],
            "is_bad_dom": [],
            "in_saturation_window": [],
            "in_calibration_errata": [],
            "saturation_total_time": [],
            "errata_total_time": [],
            "hlc": [],
            "awtd": [],
            "string": [],
            "pmt_number": [],
            "dom_number": [],
            "dom_type": [],
            "hit_type": [],
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

        # print(frame["I3EventHeader"].event_id, frame["I3EventHeader"].sub_event_id, "graphnet run")
        # print(len(om_keys), 'graphnet')

        # Process MCPulses Information

        max_energy = -1
        try:
            tracklist = frame["MMCTrackList"]

            max_particle = tracklist[0]
            for particle in tracklist:
                if particle.Ei > max_energy:
                    max_energy = particle.Ei
                    max_particle = particle

            leading = max_particle.particle

        except:
            leading = frame["PolyplopiaPrimary"]

        for om_key in om_keys:
            # Common values for each OM
            x = self._gcd_dict[om_key].position.x
            y = self._gcd_dict[om_key].position.y
            z = self._gcd_dict[om_key].position.z
            area = self._gcd_dict[om_key].area
            rde = self._get_relative_dom_efficiency(
                frame, om_key, padding_value
            )

            r_primary = phys_services.I3Calculator.closest_approach_distance(
                leading, self._gcd_dict[om_key].position
            )
            t_primary = (
                leading.time
                + phys_services.I3Calculator.cherenkov_time(
                    leading, self._gcd_dict[om_key].position
                )
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

            saturation_start = None
            saturation_stop = None
            if saturation_windows:
                if om_key in saturation_windows:
                    is_saturated_dom = 1
                    saturation_start = saturation_windows[om_key][0].start
                    saturation_stop = saturation_windows[om_key][0].stop
                else:
                    is_saturated_dom = 0
            else:
                is_saturated_dom = int(0)

            errata_start = None
            errata_stop = None
            if calibration_errata:
                if om_key in calibration_errata:
                    is_errata_dom = 1
                    errata_start = calibration_errata[om_key][0].start
                    errata_stop = calibration_errata[om_key][0].stop
                else:
                    is_errata_dom = 0
            else:
                is_errata_dom = int(0)

            # Loop over pulses for each OM
            pulses = data[om_key]
            # print(pulses)

            for _, pulse in enumerate(pulses):

                output["charge"].append(
                    getattr(pulse, "charge", padding_value)
                )

                time = getattr(pulse, "time", padding_value)
                output["time"].append(time)

                output["width"].append(getattr(pulse, "width", padding_value))
                output["pmt_area"].append(area)
                output["rde"].append(rde)
                output["dom_x"].append(x)
                output["dom_y"].append(y)
                output["dom_z"].append(z)
                output["dom_hit"].append(_)
                output["hit_type"].append("leading")

                # ID's
                output["string"].append(string)
                output["pmt_number"].append(pmt_number)
                output["dom_number"].append(dom_number)
                output["dom_type"].append(dom_type)

                # DOM flags
                output["is_bright_dom"].append(is_bright_dom)
                output["is_bad_dom"].append(is_bad_dom)
                # output["is_saturated_dom"].append(is_saturated_dom)
                # output["is_errata_dom"].append(is_errata_dom)
                # output["event_time"].append(event_time)

                # Specific Saturation Information
                if saturation_start is not None:
                    output["in_saturation_window"].append(
                        1 if saturation_start <= time <= saturation_stop else 0
                    )
                    output["saturation_total_time"].append(
                        saturation_stop - saturation_start
                    )
                else:
                    output["in_saturation_window"].append(0)
                    output["saturation_total_time"].append(0)
                if errata_start is not None:
                    output["in_calibration_errata"].append(
                        1 if errata_start <= time <= errata_stop else 0
                    )
                    output["errata_total_time"].append(
                        errata_stop - errata_start
                    )
                else:
                    output["in_calibration_errata"].append(0)
                    output["errata_total_time"].append(0)

                # Residual Information
                output["r_primary"].append(r_primary)
                output["timing_residual_primary"].append(
                    pulse.time - t_primary
                )

                # Pulse flags
                flags = getattr(pulse, "flags", padding_value)
                if flags == padding_value:
                    output["hlc"].append(padding_value)
                    output["awtd"].append(padding_value)
                else:
                    output["hlc"].append((pulse.flags >> 0) & 0x1)  # bit 0
                    output["awtd"].append(self._parse_awtd_flag(pulse))

        # Convert Event Info Into Dataframe
        evt_pulses = pd.DataFrame(
            {
                "charge": output["charge"],
                "time": output["time"],
                "width": output["width"],
                "dom_x": output["dom_x"],
                "dom_y": output["dom_y"],
                "dom_z": output["dom_z"],
                "dom_hit": output["dom_hit"],
                "pmt_area": output["pmt_area"],
                "rde": output["rde"],
                "is_bright_dom": output["is_bright_dom"],
                "is_bad_dom": output["is_bad_dom"],
                "hlc": output["hlc"],
                "awtd": output["awtd"],
                "string": output["string"],
                "pmt_number": output["pmt_number"],
                "dom_number": output["dom_number"],
                "dom_type": output["dom_type"],
                "r": output["r_primary"],
                "residual": output["timing_residual_primary"],
                "in_saturation_window": output["in_saturation_window"],
                "in_calibration_errata": output["in_calibration_errata"],
                "saturation_total_time": output["saturation_total_time"],
                "errata_total_time": output["errata_total_time"],
                "hit_type": output["hit_type"],
            },
        )

        # Produce Quantile Information of Each DOM
        t_quantiles = (
            evt_pulses.groupby(["string", "dom_number"])["time"]
            .quantile(self._quantiles_time)
            .unstack()
            .reset_index()
        )
        for quant in self._quantiles_time:
            t_quantiles = t_quantiles.rename(
                columns={quant: f"t{int(1000*quant)}"}
            )

        evt_pulses["qcumsum"] = evt_pulses.groupby(["string", "dom_number"])[
            "charge"
        ].cumsum()
        q_quantiles = (
            evt_pulses.groupby(["string", "dom_number"])["qcumsum"]
            .quantile(self._quantiles_charge)
            .unstack()
            .reset_index()
        )
        for quant in self._quantiles_charge:
            q_quantiles = q_quantiles.rename(
                columns={quant: f"q{int(1000*quant)}"}
            )
        evt_pulses["dom_qtot"] = evt_pulses.groupby(["string", "dom_number"])[
            "charge"
        ].transform("sum")
        # Extrac the Minimum Pulse Time of Each Dom
        # min_times = evt_pulses.loc[evt_pulses.groupby(["string", "dom_number"], as_index=True)['time'].idxmin()]
        earliest_hits = evt_pulses.groupby(["string", "dom_number"])[
            "time"
        ].transform("min")

        evt_pulses["t_from_leading"] = evt_pulses["time"] - earliest_hits
        min_times = evt_pulses[
            (evt_pulses["t_from_leading"] < 100) | (evt_pulses["qcumsum"] < 25)
        ]

        # min_times = min_times.merge(t_quantiles, on = ["string", "dom_number"])
        # min_times = min_times.merge(q_quantiles, on = ["string", "dom_number"])

        min_times["adjusted_time"] = (
            min_times["time"] - min_times["time"].min()
        )

        bright_doms = (
            min_times["dom_qtot"] / frame["Homogenized_QTot_New"].value >= 0.4
        )

        min_times["bright_dom"] = bright_doms.to_numpy(dtype=float)
        # bad_doms = (min_times['is_errata_dom'] == 1) | (min_times['is_saturated_dom'] == 1)
        # t_name_keys = [f't{int(1000*quant)}' for quant in self._quantiles_time]
        # q_name_keys = [f'q{int(1000*quant)}' for quant in self._quantiles_charge]
        #
        # for t_name in t_name_keys:
        #    min_times[t_name] = min_times[t_name] - min_times["time"].min()

        # reco_pulses_labeled = label_reco_pulses_newer(
        #    reco_pulses=min_times,
        #    mc_pulses=mc_labeled_pulses,
        # )
        #
        # self.set_residual_labels(
        #    frame,
        #    time = 100,
        #    charge = 50,
        #    reco_pulses = reco_pulses_labeled,
        # )

        output = min_times.to_dict(orient="list")
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
        self,
        frame: "icetray.I3Frame",
    ):

        try:
            tracklist = frame["MMCTrackList"]

            max_energy = -1
            max_particle = tracklist[0]
            for particle in tracklist:
                if particle.Ei > max_energy:
                    max_energy = particle.Ei
                    max_particle = particle

            return max_particle.particle
        except:
            print("no mmctracklist")
            mctree = frame["I3MCTree_preMuonProp"]
            return mctree[1]

    def get_leading_muon(
        self,
        bundle_muons,
        pulses,
    ):

        most_charge = -10
        most_energy = -10

        for muon in bundle_muons:

            muon_label = muon.minor_id
            muon_energy = muon.energy

            muon_pulses = pulses.loc[pulses["muon_id"] == muon_label]
            total_charge = muon_pulses["charge"].to_numpy().sum()
            # print(f'Total Charge: {total_charge}, Muon: {muon_label}')
            if total_charge > most_charge:
                leading_charge = muon
                most_charge = total_charge

            if muon_energy > most_energy:
                leading_energy = muon
                most_energy = muon_energy

        return leading_charge, leading_energy

    def get_mc_pulse_info(
        self,
        frame,
        geo,
    ):

        make_labeled_pulses(frame, geo)

        bundle_muons = get_all_bundle_muons(frame)
        event_pulses = make_labeled_pulses(frame, geo)

        # Selecting the "Leading" Options for the Muons
        leading_energy = get_leading_muon_energy(bundle_muons=bundle_muons)

        try:
            leading_charge = get_leading_muon_charge(
                bundle_muons, event_pulses
            )
        except:
            leading_charge = leading_energy

        leading_primary = frame["PolyplopiaPrimary"]

        frame["leading_charge"] = leading_charge
        frame["leading_energy"] = leading_energy

        leading_muons = [leading_charge, leading_energy, leading_primary]

        event_pulses = compute_residual_information(
            frame,
            event_pulses,
            geo,
            leading_muons,
        )

        return event_pulses, leading_muons
