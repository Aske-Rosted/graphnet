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
    from icecube import simclasses
    from icecube import recclasses

import polars as pl

import numpy as np
from collections import defaultdict 

class I3PulseExtractorNew(I3Extractor):
    """Base class for labeling pulses from laterally spread muons in moun bundles."""

    def __init__(self, pulsemap: str, 
                 charges_after_t: List[Any] = [5,10,15,20,25,30,35,40,45,50,60,70,80,90,100],
                 time_charge_percentiles: float = [1,3,6,10,15,25,50,80],
                 pulse_labeling: bool = True,
                 afterpulse_cutoff: float = 4000.0, # 4 microseconds
                 training_data: bool = True,
                 exclude_saturation: bool = False,
                 exclude_errata: bool = False,
                 #exclude_bright_doms: bool = False,
                 #exclude_bad_doms: bool = False, 
                ):
        """Construct I3FeatureExtractor.

        Args:
            pulsemap: Name of the pulse (series) map for which to extract
                reconstructed features.
            charges_after_t: List of time cutoffs (in ns) for which to compute charge
            summurized pulses. 
            time_charge_percentiles: List of charge percentile thresholds (summarized)
                for which the time at which the charge reaches that percentile will be computed.
            pulse_labeling: Whether to label pulses based on their hit type (leading, lateral, coincident, noise)
            afterpulse_cutoff: Time cutoff (in ns) after the leading pulse for which to exclude pulses as potential afterpulses.
            training_data: Whether the data being processed is training data. False if inference is being done.
            exclude_saturation: Whether to exclude pulses that fall within saturation windows.
            exclude_errata: Whether to exclude pulses that fall within calibration errata windows.
            exclude_bright_doms: Whether to exclude pulses from bright DOMs. (Probably A Better Input for NodesAsPulses)
            exclude_bad_doms: Whether to exclude pulses from bad DOMs. (Probably A Better Input for NodesAsPulses)
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

        self._training_data = training_data
        self._exclude_saturation = exclude_saturation
        self._exclude_errata = exclude_errata
        #self._exclude_bright_doms = exclude_bright_doms
        #self._exclude_bad_doms = exclude_bad_doms

        self._drop_at_end = [
            'in_saturation_window',
            'in_calibration_errata',
            'saturation_start_time',
            'saturation_stop_time',
            'errata_start_time',
            'errata_stop_time',
            'qcumsum',
        ]
        # Base class constructor
        super().__init__(pulsemap)

class I3PulseExtractorNewIceCube86(I3PulseExtractorNew):
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
            "width": [],
            "dom_x": [],
            "dom_y": [],
            "dom_z": [],
            "dom_hit": [],
            "pmt_area": [],
            "rde": [],
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

        if self._pulse_labeling:
            output_pulse_labeling = {
                "r_charge": [],
                "r_energy": [],
                "r_primary": [],
                "timing_residual_charge": [],
                "timing_residual_energy": [],
                "timing_residual_primary": [],
            }
            output.update(output_pulse_labeling)
        
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


        # Process MCPulses Information
        particle_pdg = frame['PolyplopiaPrimary'].pdg_encoding
        if self._training_data:
            if (np.abs(particle_pdg) not in [12,14,16]) & (self._pulse_labeling):
                mc_labeled_pulses, leading = self.get_mc_pulse_info(
                    frame, 
                    geo = self._gcd_dict,
                )

                make_multiplicity_statistics(
                    frame,
                    event_pulses=mc_labeled_pulses,
                )

            else:
                leading = self._get_leading_nugen(
                    frame,
                )

        for om_key in om_keys:
            # Common values for each OM
            x = self._gcd_dict[om_key].position.x
            y = self._gcd_dict[om_key].position.y
            z = self._gcd_dict[om_key].position.z
            area = self._gcd_dict[om_key].area
            rde = self._get_relative_dom_efficiency(
                frame, om_key, padding_value
            )

            if self._pulse_labeling:
                r = phys_services.I3Calculator.closest_approach_distance(leading[0], self._gcd_dict[om_key].position)
                r_energy = phys_services.I3Calculator.closest_approach_distance(leading[1], self._gcd_dict[om_key].position)
                r_primary = phys_services.I3Calculator.closest_approach_distance(leading[2], self._gcd_dict[om_key].position)

                t_charge = leading[0].time + phys_services.I3Calculator.cherenkov_time(leading[0],self._gcd_dict[om_key].position)
                t_energy = leading[1].time + phys_services.I3Calculator.cherenkov_time(leading[1],self._gcd_dict[om_key].position)
                t_primary = leading[2].time + phys_services.I3Calculator.cherenkov_time(leading[2],self._gcd_dict[om_key].position) 

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
                #print(pulses)

            for _,pulse in enumerate(pulses):

                output["charge"].append(
                    getattr(pulse, "charge", padding_value)
                )

                time = getattr(pulse, "time", padding_value)
                output["time"].append(
                    time
                )

                output["width"].append(getattr(pulse, "width", padding_value))
                output["pmt_area"].append(area)
                output["rde"].append(rde)
                output["dom_x"].append(x)
                output["dom_y"].append(y)
                output["dom_z"].append(z)
                output['dom_hit'].append(_)

                # ID's
                output["string"].append(string)
                output["pmt_number"].append(pmt_number)
                output["dom_number"].append(dom_number)
                output["dom_type"].append(dom_type)

                # DOM flags
                #output["is_bright_dom"].append(is_bright_dom)
                output["is_bad_dom"].append(is_bad_dom)
                #output["is_saturated_dom"].append(is_saturated_dom)
                #output["is_errata_dom"].append(is_errata_dom)
                #output["event_time"].append(event_time)

                # Specific Saturation Information
                if saturation_start is not None:
                    output["in_saturation_window"].append(1 if saturation_start <= time <= saturation_stop else 0)
                    output["saturation_total_time"].append(saturation_stop - saturation_start)
                    output["saturation_start_time"].append(saturation_start)
                    output["saturation_stop_time"].append(saturation_stop)
                else:
                    output["in_saturation_window"].append(0)
                    output["saturation_total_time"].append(0)
                    output["saturation_start_time"].append(-1)
                    output["saturation_stop_time"].append(-1)
                if errata_start is not None:
                    output["in_calibration_errata"].append(1 if errata_start <= time <= errata_stop else 0)
                    output["errata_total_time"].append(errata_stop - errata_start)
                    output["errata_start_time"].append(errata_start)
                    output["errata_stop_time"].append(errata_stop)
                else:
                    output["in_calibration_errata"].append(0)
                    output["errata_total_time"].append(0)
                    output["errata_start_time"].append(-1)
                    output["errata_stop_time"].append(-1)

                # Residual Information
                if self._pulse_labeling:
                    output['r_charge'].append(r)
                    output['r_energy'].append(r_energy)
                    output['r_primary'].append(r_primary)
                    output['timing_residual_charge'].append(pulse.time - t_charge)
                    output['timing_residual_energy'].append(pulse.time - t_energy)
                    output['timing_residual_primary'].append(pulse.time - t_primary)

                # Pulse flags
                flags = getattr(pulse, "flags", padding_value)
                if flags == padding_value:
                    output["hlc"].append(padding_value)
                    output["awtd"].append(padding_value)
                else:
                    output["hlc"].append((pulse.flags >> 0) & 0x1)  # bit 0
                    output["awtd"].append(self._parse_awtd_flag(pulse))

        # Convert Dictionary to Dataframe for Easier Manipulation
        # print(frame['I3EventHeader'].event_id, frame['HQTOT'].value)
        evt_pulses = pl.DataFrame(output, strict=False)

        # Summarized Charge Information
        evt_pulses = evt_pulses.with_columns(
            qcumsum=pl.col("charge").cum_sum().over(["string", "dom_number"]),
            dom_qtot=pl.col("charge").sum().over(["string", "dom_number"]),
            first_hit=pl.col("time").min().over(["string", "dom_number"]),
        )
    
        evt_pulses = evt_pulses.with_columns(
            charge_temp = pl.col("charge")
        )

        mask = pl.lit(False)

        if self._exclude_saturation or self._training_data:
            mask = mask | (pl.col("in_saturation_window").cast(pl.Int32) == 1)

            evt_pulses = evt_pulses.with_columns(
                charge_sat=pl.when(mask).then(0).otherwise(pl.col('charge'))
            )

        if self._exclude_errata or self._training_data:
            mask = mask | (pl.col("in_calibration_errata").cast(pl.Int32) == 1)

            evt_pulses = evt_pulses.with_columns(
                charge_temp=pl.when(mask).then(0).otherwise(pl.col('charge'))
            )
        
        evt_pulses = evt_pulses.with_columns(
            t_from_leading = pl.col("time") - pl.col("time").min().over(["string", "dom_number"]),
        )

        afterpulses_removed = evt_pulses.filter(pl.col('t_from_leading') < self._afterpulse_cutoff)

        afterpulses_removed = afterpulses_removed.with_columns(
            dom_qtot_no_afterpulse=pl.col("charge").sum().over(["string", "dom_number"]),
            dom_qtot_exc_no_afterpulse=pl.col("charge_temp").sum().over(["string", "dom_number"]),
        )

        """
        For training data, process all combinations of summarized features:
        -> No Exclusion
        -> Exclude Saturation Windows
        -> Exclude Both
        """

        if (not self._exclude_saturation and not self._exclude_errata) or self._training_data:
            """
            All Pulses, Process Conditionally Unless Processing Training Data.
            """
            
            afterpulses_removed = self._charge_after_time(
                afterpulses_removed,
                time_cutoffs=self._charges_after_t,
            ) 

            # Add Time at Charge Percentile
            afterpulses_removed = self._time_at_charge_percentile(
                afterpulses_removed,
                quantiles=self._time_charge_percentiles,
            )

        if (self._exclude_saturation and self._exclude_errata) or self._training_data:
            """
            All Exclusions, Process Conditionally Unless Processing Training Data.
            """
            afterpulses_removed = self._charge_after_time(
                afterpulses_removed,
                time_cutoffs=self._charges_after_t,
                charge_key = 'charge_temp',
                name = '_excl',
            ) 

            # Add Time at Charge Percentile
            afterpulses_removed = self._time_at_charge_percentile(
                afterpulses_removed,
                quantiles=self._time_charge_percentiles,
                charge_key = 'charge_temp',
                name = '_excl',
            )

        if (self._exclude_saturation and not self._exclude_errata) or self._training_data:
            """
            Only Exclude Saturation, Process Conditionally Unless Processing Training Data.
            """
            afterpulses_removed = self._charge_after_time(
                afterpulses_removed,
                time_cutoffs=self._charges_after_t,
                charge_key = 'charge_sat',
                name = '_sat',
            ) 

            # Add Time at Charge Percentile
            afterpulses_removed = self._time_at_charge_percentile(
                afterpulses_removed,
                quantiles=self._time_charge_percentiles,
                charge_key = 'charge_sat',
                name = '_sat',
            )

        median_pulse_time = (
            afterpulses_removed.select(
                pl.col('time').sort_by("time")
                .filter(
                    pl.col('charge').sort_by("time").cum_sum() >= (pl.col('charge').sum() * 0.5)
                )
                .first()
            ).item()
        )

        afterpulses_removed = afterpulses_removed.with_columns(
            time_from_median = pl.col('time') - median_pulse_time,
        )

        afterpulses_removed = afterpulses_removed.drop("charge_temp")

        min_times = (
            afterpulses_removed.sort("time")
            .group_by(["string", "dom_number"])
            .first()
        )

        min_times = min_times.with_columns(
            adjusted_time=pl.col("time") - pl.col("time").min()
        )

        min_times = min_times.with_columns(
            bright_dom = (pl.col('dom_qtot')/frame['HQTOT'].value).ge(0.4).cast(pl.Int32)
        )

        if (np.abs(particle_pdg) not in [12,14,16]) and self._pulse_labeling:

            reco_pulses_labeled = label_reco_pulses(
                reco_pulses=min_times,
                mc_pulses=mc_labeled_pulses,
            )

            hit_types = ['charge', 'energy', 'primary']

            for hit_type in hit_types:
                reco_pulses_final = self.label_training_targets(
                    leading_muon=hit_type,
                    pulses=reco_pulses_labeled,
                )

        else:
            reco_pulses_final = min_times

        if self._training_data:
            reco_pulses_final = reco_pulses_final.drop([
                'r_charge', 'r_energy', 'r_primary', 'timing_residual_primary', 'timing_residual_charge',
            ])
        reco_pulses_final = reco_pulses_final.drop(self._drop_at_end)
        output = reco_pulses_final.to_dict(as_series = False)

        frame['NumberStrings'] = dataclasses.I3Double(
            float(evt_pulses.select(pl.col("string").n_unique()).item())
        )
        frame['NumberDOMs'] = dataclasses.I3Double(
           float(evt_pulses.select(pl.struct(["dom_x", "dom_y", "dom_z"]).n_unique()).item())
        )

        hlc_pulses = evt_pulses.filter(pl.col("hlc") == 1)

        frame['NumberStringsHLC'] = dataclasses.I3Double(
            float(hlc_pulses.select(pl.col("string").n_unique()).item())
        )
        frame['NumberDOMsHLC'] = dataclasses.I3Double(
            float(hlc_pulses.select(pl.struct(["dom_x", "dom_y", "dom_z"]).n_unique()).item())
        )

        del reco_pulses_final, evt_pulses, hlc_pulses
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
        pulses: pl.DataFrame,
        time_cutoffs: list[float] = [10.0],
        charge_key: str = 'charge',
        name = '',
    ) -> pl.DataFrame:
        """
        Calculate total charge for different time windows.
        """
        
        operations = [
            pl.col(charge_key)
            .filter(pl.col('t_from_leading') <= time_cutoff).sum()
            .over(['string', 'dom_number'])
            .fill_null(0)
            .alias(f"charge_after_{time_cutoff}{name}")
            for time_cutoff in time_cutoffs
        ]

        return pulses.with_columns(operations)
    
    def _time_at_charge_percentile(
        self,
        pulses: pl.DataFrame,
        quantiles: list[float] = [5],
        charge_key: str = 'charge',
        name = '',
    ) -> pl.DataFrame:
        """
        Calculate time at which charge reaches a certain percentile for each DOM.
        """

        operations = [
            pl.col('time').sort_by('time')
            .filter(
                pl.col(charge_key).sort_by('time').cum_sum()
                .over(['string', 'dom_number']) >= 
                (pl.col(charge_key).sum().over(['string', 'dom_number']) * q / 100)
            )
            .first()
            .over(['string', 'dom_number'])
            .alias(f'time_charge_{q}{name}')
            for q in quantiles
        ]   

        return pulses.with_columns(operations)
        
    
    # Error Getting this for a certain set
    def _get_leading_particle(
        self,
        frame: "icetray.I3Frame",
    ):

        primary = frame['PolyplopiaPrimary']
        pdg = frame['PolyplopiaPrimary'].pdg_encoding
        mctree = frame['I3MCTree_preMuonProp']

        # Handling Neutrino Leading Particles
        if np.abs(pdg) in [12, 14, 16]:
            current = mctree[1]
            while mctree.number_of_children(current) > 0:
                current = mctree.first_child(current)

            primary.pos.x = current.pos.x
            primary.pos.y = current.pos.y
            primary.pos.z = current.pos.z

        # Handling Corsika Events
        else:
            current = mctree[frame['PolyplopiaPrimary']]
            highest_energy = -1
            bundle_particles = mctree.get_daughters(current)
            for particle in bundle_particles:
                if (particle.type_string in ['MuPlus', 'MuMinus'] and particle.location_type_string == 'InIce'):
                    if particle.energy > highest_energy:
                        highest_energy = particle.energy
                        current = particle
        
        return current
        
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
        
        primary = frame['PolyplopiaPrimary']
        pdg = frame['PolyplopiaPrimary'].pdg_encoding
        full_mctree = frame['I3MCTree']
        mctree = frame['I3MCTree_preMuonProp']
        if np.abs(pdg) in [12, 14, 16]:
            current = mctree[1]
            while mctree.number_of_children(current) > 0:
                current = mctree.first_child(current)

            primary.pos.x = current.pos.x
            primary.pos.y = current.pos.y
            primary.pos.z = current.pos.z
        else:
            current = mctree[frame['PolyplopiaPrimary']]
            highest_energy = -1
            bundle_particles = mctree.get_daughters(current)
            for particle in bundle_particles:
                if (particle.type_string in ['MuPlus', 'MuMinus'] and particle.location_type_string == 'InIce'):
                    if particle.energy > highest_energy:
                        highest_energy = particle.energy
                        current = particle

        tracklist = frame['MMCTrackList']

        e_initial = 0
        for track in tracklist:
            #if full_mctree.is_in_subtree(primary, track.particle) == True: # Cleaning Coincidence Hits
            if track.Ei > e_initial:
                e_initial = track.Ei
                current = track.particle
        
        return primary, current
    
    
    def get_mc_pulse_info(
        self,
        frame,
        geo,
    ):
        
        bundle_muons = get_all_bundle_muons(frame) 
        event_pulses = make_labeled_pulses(frame, geo)


        # Selecting the "Leading" Options for the Muons
        leading_primary, leading_energy = self._get_leading_charged_particles(
            frame,
        )

        try:
            leading_charge = get_leading_muon_charge(
                bundle_muons,
                event_pulses
            )
        except:
            leading_charge = leading_energy

        frame['leading_charge'] = leading_charge
        frame['leading_energy'] = leading_energy

        leading_muons = [leading_charge, leading_energy, leading_primary]

        event_pulses = compute_residual_information(
            frame,
            event_pulses,
            geo,
            leading_muons,
        )

        return event_pulses, leading_muons

    def label_training_targets(
        self,
        leading_muon,
        pulses: pl.DataFrame,
    ): 
        """
        If the hit type is leading, label it as a 0.
        If the hit type is lateral, label it as a 1.
        If the hit type is coincident, label it as a 2.
        If the hit type is noise, label it as a 3.
        """

        mapping = {
            'leading': 0,
            'lateral': 1,
            'coincidence': 2,
            'noise': 3,
        }

        pulses = (
            pulses.with_columns(
                pl.col(f'hit_type_{leading_muon}').replace(mapping).alias(f'label_{leading_muon}')
            )
            .drop(['hit_type_charge', 'hit_type_energy', 'hit_type_primary'])
        )
        
        return pulses





        
    