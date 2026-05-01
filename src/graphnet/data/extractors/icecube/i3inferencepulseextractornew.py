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
pd.options.mode.chained_assignment = None
import numpy as np
from collections import defaultdict


class I3InferencePulseExtractorNew(I3Extractor):
    """Base class for labeling pulses from laterally spread muons in moun
    bundles."""

    def __init__(
        self,
        pulsemap: str,
        charges_after_t: List[Any] = [
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
        ],
        time_charge_percentiles: float = [1, 3, 6, 10, 15, 25, 50, 80],
        pulse_labeling: bool = True,
        afterpulse_cutoff: float = 4000.0,  # 4 microseconds
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
        # Charge Accumulated After T Seconds
        self._charges_after_t = charges_after_t
        # Time Passed when Charge Reaches Threshold
        self._time_charge_percentiles = time_charge_percentiles
        self._pulse_labeling = pulse_labeling
        # Cutoff For Pulses from DOM First Hit
        self._afterpulse_cutoff = afterpulse_cutoff

        self._drop_at_end = [
            "in_saturation_window",
            "in_calibration_errata",
            "saturation_start_time",
            "saturation_stop_time",
            "errata_start_time",
            "errata_stop_time",
            "qcumsum",
        ]
        # Base class constructor
        super().__init__(pulsemap)


"""
Chain
-> get a list of the muons
-> compute mc pulse array
-> label pulses
-> 
"""


class I3InferencePulseExtractorNewIceCube86(I3InferencePulseExtractorNew):
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
            "dom_qtot_excl": [],  # DOM Excluded Saturation/Calibration Errata Windows
            "dom_x": [],
            "dom_y": [],
            "dom_z": [],
            "dom_hit": [],
            "pmt_area": [],
            "rde": [],
            "r_charge": [],
            "r_energy": [],
            "r_primary": [],
            "timing_residual_charge": [],
            "timing_residual_energy": [],
            "timing_residual_primary": [],
            # "is_bright_dom": [],
            "is_bad_dom": [],
            "in_saturation_window": [],
            "in_calibration_errata": [],
            "saturation_start_time": [],
            "saturation_stop_time": [],
            "saturation_total_time": [],
            "errata_start_time": [],
            "errata_stop_time": [],
            "errata_total_time": [],
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

                # ID's
                output["string"].append(string)
                output["pmt_number"].append(pmt_number)
                output["dom_number"].append(dom_number)
                output["dom_type"].append(dom_type)

                # DOM flags
                # output["is_bright_dom"].append(is_bright_dom)
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
                    output["saturation_start_time"].append(saturation_start)
                    output["saturation_stop_time"].append(saturation_stop)
                else:
                    output["in_saturation_window"].append(0)
                    output["saturation_total_time"].append(0)
                    output["saturation_start_time"].append(-1)
                    output["saturation_stop_time"].append(-1)
                if errata_start is not None:
                    output["in_calibration_errata"].append(
                        1 if errata_start <= time <= errata_stop else 0
                    )
                    output["errata_total_time"].append(
                        errata_stop - errata_start
                    )
                    output["errata_start_time"].append(errata_start)
                    output["errata_stop_time"].append(errata_stop)
                else:
                    output["in_calibration_errata"].append(0)
                    output["errata_total_time"].append(0)
                    output["errata_start_time"].append(-1)
                    output["errata_stop_time"].append(-1)

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
                # "is_bright_dom": output['is_bright_dom'],
                "is_bad_dom": output["is_bad_dom"],
                "hlc": output["hlc"],
                "awtd": output["awtd"],
                "string": output["string"],
                "pmt_number": output["pmt_number"],
                "dom_number": output["dom_number"],
                "dom_type": output["dom_type"],
                "in_saturation_window": output["in_saturation_window"],
                "saturation_start_time": output["saturation_start_time"],
                "saturation_stop_time": output["saturation_stop_time"],
                "in_calibration_errata": output["in_calibration_errata"],
                "errata_start_time": output["errata_start_time"],
                "errata_stop_time": output["errata_stop_time"],
                "saturation_total_time": output["saturation_total_time"],
                "errata_total_time": output["errata_total_time"],
            },
        )

        # Produce Quantile Information of Each DOM

        # Compute Charge of T Seconds

        evt_pulses["qcumsum"] = evt_pulses.groupby(["string", "dom_number"])[
            "charge"
        ].cumsum()

        evt_pulses["dom_qtot"] = evt_pulses.groupby(["string", "dom_number"])[
            "charge"
        ].transform("sum")

        evt_pulses["charge_temp"] = evt_pulses["charge"]

        evt_pulses.loc[
            (evt_pulses["in_saturation_window"].astype(int) == 1)
            | (evt_pulses["in_calibration_errata"].astype(int) == 1),
            "charge_temp",
        ] = 0
        evt_pulses["dom_qtot_exc"] = evt_pulses.groupby(
            ["string", "dom_number"]
        )["charge_temp"].transform("sum")
        # Extrac the Minimum Pulse Time of Each Dom
        # min_times = evt_pulses.loc[evt_pulses.groupby(["string", "dom_number"], as_index=True)['time'].idxmin()]
        earliest_hits = evt_pulses.groupby(["string", "dom_number"])[
            "time"
        ].transform("min")

        evt_pulses["t_from_leading"] = evt_pulses["time"] - earliest_hits

        afterpulses_removed = evt_pulses[
            (evt_pulses["t_from_leading"] < self._afterpulse_cutoff)
        ]

        afterpulses_removed["dom_qtot_no_afterpulse"] = (
            afterpulses_removed.groupby(["string", "dom_number"])[
                "charge"
            ].transform("sum")
        )
        afterpulses_removed["dom_qtot_exc_no_afterpulse"] = (
            afterpulses_removed.groupby(["string", "dom_number"])[
                "charge_temp"
            ].transform("sum")
        )

        # Add Charges After T ns
        for time_cut in self._charges_after_t:
            afterpulses_removed = self._charge_after_time(
                afterpulses_removed,
                time_cutoff=time_cut,
            )

        # Add Time at Charge Percentile
        for quant in self._time_charge_percentiles:
            afterpulses_removed = self._time_at_charge_percentile(
                afterpulses_removed,
                quantile=quant,
            )

        for time_cut in self._charges_after_t:
            afterpulses_removed = self._charge_after_time(
                afterpulses_removed,
                time_cutoff=time_cut,
                charge_key="charge_temp",
                name="_excl",
            )

        # Add Time at Charge Percentile
        for quant in self._time_charge_percentiles:
            afterpulses_removed = self._time_at_charge_percentile(
                afterpulses_removed,
                quantile=quant,
                charge_key="charge_temp",
                name="_excl",
            )

        median_pulse_time = self._weighted_quantile(
            afterpulses_removed["time"].values,
            afterpulses_removed["charge"].values,
            quantile=50,
        )

        afterpulses_removed["time_from_median"] = (
            afterpulses_removed["time"] - median_pulse_time
        )

        afterpulses_removed = afterpulses_removed.drop(["charge_temp"], axis=1)

        #
        # min_times = evt_pulses[(evt_pulses['t_from_leading'] < self._dom_time_max) | (evt_pulses['qcumsum'] < self._dom_charge_max)]
        # Restrict to first hits
        min_times = afterpulses_removed.loc[
            evt_pulses.groupby(["string", "dom_number"], as_index=True)[
                "time"
            ].idxmin()
        ]

        min_times["adjusted_time"] = (
            min_times["time"] - min_times["time"].min()
        )

        bright_doms = min_times["dom_qtot"] / frame["HQTOT"].value >= 0.4

        min_times["bright_dom"] = bright_doms.to_numpy(dtype=float)

        # bad_doms = (min_times['is_errata_dom'] == 1) | (min_times['is_saturated_dom'] == 1)
        # t_name_keys = [f't{int(1000*quant)}' for quant in self._quantiles_time]
        # q_name_keys = [f'q{int(1000*quant)}' for quant in self._quantiles_charge]
        #
        # for t_name in t_name_keys:
        #    min_times[t_name] = min_times[t_name] - min_times["time"].min()

        reco_pulses_final = min_times

        reco_pulses_final = reco_pulses_final.drop(self._drop_at_end, axis=1)
        output = reco_pulses_final.to_dict(orient="list")

        # frame['NumberStrings'] = dataclasses.I3Double(len(np.unique(evt_pulses['string'].values)))
        # frame['NumberDOMs'] = dataclasses.I3Double(len(np.unique(evt_pulses[['dom_x', 'dom_y', 'dom_z']].values)))
        # evt_pulses = evt_pulses[evt_pulses['hlc'] == 1]  # Only keep HLC pulses
        # frame['NumberStringsHLC'] = dataclasses.I3Double(len(np.unique(evt_pulses['string'].values)))
        # frame['NumberDOMsHLC'] = dataclasses.I3Double(len(np.unique(evt_pulses[['dom_x', 'dom_y', 'dom_z']].values)))
        del reco_pulses_final, evt_pulses
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

    def _charge_after_time(
        self,
        pulses: pd.DataFrame,
        time_cutoff: float = 10.0,
        charge_key: str = "charge",
        name="",
    ) -> pd.DataFrame:
        """Calculate total charge after a certain time window."""

        pulses[f"charge_after_{time_cutoff}{name}"] = (
            pulses[pulses["t_from_leading"] <= time_cutoff]
            .groupby(["string", "dom_number"])[charge_key]
            .transform("sum")
        )
        return pulses

    def _weighted_quantile(
        self,
        x,
        weights,
        quantile: float,
    ):
        """Compute the weighted quantile of a 1D array.

        Args:
            x: 1D array of values.
            weights: 1D array of weights.
            quantile: Quantile to compute (between 0 and 1).

        Returns:
            Weighted quantile value.
        """
        sorter = np.argsort(x)
        x_sorted = x[sorter]
        weights_sorted = weights[sorter]
        cumulative_weights = np.cumsum(weights_sorted)
        cutoff = weights_sorted.sum() * (quantile / 100)

        return x_sorted[cumulative_weights >= cutoff][0]

    def _time_at_charge_percentile(
        self,
        pulses: pd.DataFrame,
        quantile: float = 0.5,
        charge_key: str = "charge",
        name="",
    ) -> pd.DataFrame:

        test = pulses.groupby(["string", "dom_number"]).apply(
            lambda x: self._weighted_quantile(
                x["time"].values,
                x[charge_key].values,
                quantile=quantile,
            )
        )
        pulses[f"time_charge_{quantile}{name}"] = list(
            zip(pulses["string"], pulses["dom_number"])
        )
        pulses[f"time_charge_{quantile}{name}"] = pulses[
            f"time_charge_{quantile}{name}"
        ].map(test)
        return pulses

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

    def _get_leading_nugen(
        self,
        frame,
    ):

        leading_primary, leading_energy = self._get_leading_charged_particles(
            frame,
        )
        leading_charge = leading_energy

        return [leading_charge, leading_energy, leading_primary]

    def _get_leading_charged_particles(
        self,
        frame,
    ):

        primary = frame["PolyplopiaPrimary"]
        pdg = frame["PolyplopiaPrimary"].pdg_encoding
        full_mctree = frame["I3MCTree"]
        mctree = frame["I3MCTree_preMuonProp"]
        if np.abs(pdg) in [12, 14, 16]:
            current = mctree[1]
            while mctree.number_of_children(current) > 0:
                current = mctree.first_child(current)

            primary.pos.x = current.pos.x
            primary.pos.y = current.pos.y
            primary.pos.z = current.pos.z
        else:
            current = mctree[frame["PolyplopiaPrimary"]]
            highest_energy = -1
            bundle_particles = mctree.get_daughters(current)
            for particle in bundle_particles:
                if (
                    particle.type_string in ["MuPlus", "MuMinus"]
                    and particle.location_type_string == "InIce"
                ):
                    if particle.energy > highest_energy:
                        highest_energy = particle.energy
                        current = particle

        tracklist = frame["MMCTrackList"]

        e_initial = 0
        for track in tracklist:
            # if full_mctree.is_in_subtree(primary, track.particle) == True: # Cleaning Coincidence Hits
            if track.Ei > e_initial:
                e_initial = track.Ei
                current = track.particle

        return primary, current

    def get_mc_pulse_info(
        self,
        frame,
        geo,
    ):

        make_labeled_pulses(frame, geo)

        bundle_muons = get_all_bundle_muons(frame)
        event_pulses = make_labeled_pulses(frame, geo)

        # Selecting the "Leading" Options for the Muons
        leading_primary, leading_energy = self._get_leading_charged_particles(
            frame,
        )

        try:
            leading_charge = get_leading_muon_charge(
                bundle_muons, event_pulses
            )
        except:
            leading_charge = leading_energy

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

    def make_multiplicity_information(
        self,
        frame,
        mc_pulses,
    ):

        make_multiplicity_statistics(
            frame,
            mc_pulses,
        )

    def label_training_targets(
        self,
        leading_muon,
        pulses,
    ):
        """If the hit type is leading, label it as a 0.

        If the hit type is lateral and has a negative residual, label it as a
        1. If the hit type is lateral and has a positive residual, label it 10.
        If the hit type is coincident, label it as a 2. If the hit type is
        noise, label it as a 3.
        """

        mapping = {
            "leading": 0,
            "lateral": 1,
            "coincidence": 2,
            "noise": 3,
        }

        pulses[f"label_{leading_muon}"] = pulses[
            f"hit_type_{leading_muon}"
        ].map(mapping)

        pulses[f"label_{leading_muon}"] = np.where(
            (pulses[f"label_{leading_muon}"] == 1)
            & (pulses[f"residual_{leading_muon}"] >= 0),
            10,
            pulses[f"label_{leading_muon}"],
        )

        return pulses.drop(
            ["hit_type_charge", "hit_type_energy", "hit_type_primary"], axis=1
        )
