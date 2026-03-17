"""I3Extractor class(es) for extracting truth-level information."""

import numpy as np
import matplotlib.path as mpath
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .i3extractor import I3Extractor
from .utilities.frames import (
    frame_is_montecarlo,
    frame_is_noise,
)
from graphnet.utilities.imports import has_icecube_package
from .utilities.mctree_processing import make_shower_and_stochasticity_info

parent_dir = "/data/user/mnakos/EHE_Globalfit"
import sys

sys.path.append(parent_dir)

from .utilities.track_topologies import (
    compute_skimming_event,
    get_topology_metrics,
    compute_dom_positions,
    make_outer_boundary,
)

if has_icecube_package() or TYPE_CHECKING:
    from icecube import (
        dataclasses,
        icetray,
        phys_services,
        simclasses,
    )  # pyright: reportMissingImports=false


class I3BundleExtractor(I3Extractor):
    """Class for extracting truth-level information."""

    def __init__(
        self,
        name: str = "truth",
        borders: Optional[List[np.ndarray]] = None,
        mctree: Optional[str] = "I3MCTree_preMuonProp",
        sim_type: Optional[str] = "corsika",
    ):
        """Construct I3BundleExtractor.

        Args:
            name: Name of the `I3Extractor` instance.
            borders: Array of boundaries of the detector volume as ((x,y),z)-
                coordinates, for identifying, e.g., particles starting and
                stopping within the detector. Defaults to hard-coded boundary
                coordinates.
            mctree: Str of which MCTree to use for truth values.
        """
        # Base class constructor
        super().__init__(name)

        if borders is None:
            border_xy = np.array(
                [
                    (-256.1400146484375, -521.0800170898438),
                    (-132.8000030517578, -501.45001220703125),
                    (-9.13000011444092, -481.739990234375),
                    (114.38999938964844, -461.989990234375),
                    (237.77999877929688, -442.4200134277344),
                    (361.0, -422.8299865722656),
                    (405.8299865722656, -306.3800048828125),
                    (443.6000061035156, -194.16000366210938),
                    (500.42999267578125, -58.45000076293945),
                    (544.0700073242188, 55.88999938964844),
                    (576.3699951171875, 170.9199981689453),
                    (505.2699890136719, 257.8800048828125),
                    (429.760009765625, 351.0199890136719),
                    (338.44000244140625, 463.7200012207031),
                    (224.5800018310547, 432.3500061035156),
                    (101.04000091552734, 412.7900085449219),
                    (22.11000061035156, 509.5),
                    (-101.05999755859375, 490.2200012207031),
                    (-224.08999633789062, 470.8599853515625),
                    (-347.8800048828125, 451.5199890136719),
                    (-392.3800048828125, 334.239990234375),
                    (-437.0400085449219, 217.8000030517578),
                    (-481.6000061035156, 101.38999938964844),
                    (-526.6300048828125, -15.60000038146973),
                    (-570.9000244140625, -125.13999938964844),
                    (-492.42999267578125, -230.16000366210938),
                    (-413.4599914550781, -327.2699890136719),
                    (-334.79998779296875, -424.5),
                ]
            )
            border_z = np.array([-512.82, 524.56])
            self._borders = [border_xy, border_z]
        else:
            self._borders = borders
        self._mctree = mctree
        self._sim_type = sim_type
        self._dom_list = compute_dom_positions()
        self._outer_boundar = make_outer_boundary(self._dom_list)

    def __call__(
        self, frame: "icetray.I3Frame", padding_value: Any = -1
    ) -> Dict[str, Any]:
        """Extract truth-level information."""
        is_mc = frame_is_montecarlo(frame, self._mctree)
        is_noise = False
        # is_noise = frame_is_noise(frame, self._mctree)
        # sim_type = self._find_data_type(is_mc, self._i3_file)
        sim_type = self._sim_type
        """
        Primary: Primary Corsika/Neutrino
        Leading: Highest Energy Charged Particle
        Charge: Highest Charge Deposited Charged Particle (Corsika)
        -> For NuGen, same as leading
        """

        output = {
            "azimuth": padding_value,
            "zenith": padding_value,
            "pid": padding_value,
            "event_time": frame["I3EventHeader"].start_time.utc_daq_time,
            "interaction_type": padding_value,
            "RunID": frame["I3EventHeader"].run_id,
            "SubrunID": frame["I3EventHeader"].sub_run_id,
            "EventID": frame["I3EventHeader"].event_id,
            "SubEventID": frame["I3EventHeader"].sub_event_id,
            "true_starting": padding_value,  # True Starting Vertex Inside Detector Volume
            "closest_approach_x": padding_value,  # Closest Approach X Coordinate
            "closest_approach_y": padding_value,  # Closest Approach Y Coordinate
            "closest_approach_z": padding_value,  # Closest Approach Z Coordinate
            "topology_primary_high": padding_value,
            "topology_primary_low": padding_value,
            "topology_leading_high": padding_value,
            "topology_leading_low": padding_value,
            "charge": frame["HQTOT"].value,
            "fraction_coincidence": padding_value,
            "muon_multiplicity": padding_value,
            "muon_multiplicity_cylinder": padding_value,
            "deposited_muon_multiplicity": padding_value,
            "deposited_muon_multiplicity_residual_primary": padding_value,
            "deposited_muon_multiplicity_residual_energy": padding_value,
            "deposited_muon_multiplicity_residual_charge": padding_value,
            "primary_rms_energy": padding_value,
            "primary_rms3_energy": padding_value,
            "primary_rms_MCPE": padding_value,
            "primary_rms3_MCPE": padding_value,
            "primary_most_lateral_deposit": padding_value,
            "leading_rms_energy": padding_value,
            "leading_rms3_energy": padding_value,
            "leading_rms_MCPE": padding_value,
            "leading_rms3_MCPE": padding_value,
            "leading_most_lateral_deposit": padding_value,
            # "primary_stochasticity_std_energy": padding_value,
            # "primary_stochasticity_pomean_energy": padding_value,
            # "primary_stochasticity_pomedian_energy": padding_value,
            # "primary_stochasticity_std_MCPE": padding_value,
            # "primary_stochasticity_pomean_MCPE": padding_value,
            # "primary_stochasticity_pomedian_MCPE": padding_value,
            "leading_stochasticity_std_energy": padding_value,
            "leading_stochasticity_pomean_energy": padding_value,
            "leading_stochasticity_pomedian_energy": padding_value,
            "leading_stochasticity_std_MCPE": padding_value,
            "leading_stochasticity_pomean_MCPE": padding_value,
            "leading_stochasticity_pomedian_MCPE": padding_value,
            "primary_length_deposited": padding_value,
            "leading_length_deposited": padding_value,
            "length_in_detector": padding_value,
            "length_in_detector_100": padding_value,
            "length_in_detector_200": padding_value,
            "hitstrings": frame["NumberStrings"].value,
            "hitdoms": frame["NumberDOMs"].value,
            "hitstringsHLC": frame["NumberStringsHLC"].value,
            "hitdomsHLC": frame["NumberDOMsHLC"].value,
            # "flat_spectrum_weight": self._generation_spectrum_correction(frame),
        }

        # Only InIceSplit P frames contain ML appropriate I3RecoPulseSeriesMap etc.
        # At low levels i3files contain several other P frame splits (e.g NullSplit),
        # we remove those here.
        if frame["I3EventHeader"].sub_event_stream not in [
            "InIceSplit",
            "Final",
        ]:
            return output

        """
        Gather Basic Event Information
        """
        primary_particle, interaction_type = self._get_basic_event_information(
            frame,
        )

        primary, leading = self._get_leading_particle(
            frame,
        )

        output.update(
            {
                "primary_energy": primary_particle.energy,
                "azimuth": primary.dir.azimuth,
                "zenith": primary.dir.zenith,
                # "azi_leading": leading.dir.azimuth, # Azimuth of the Highest Energy Charged Lepton in Event
                # "zen_leading": leading.dir.zenith, # Zenith of the Highest Energy Charged Lepton in Event
                "pid": primary_particle.pdg_encoding,
                "interaction_type": interaction_type,
            }
        )
        """Gather Starting Event Information."""
        topology_info = self._event_topologies(
            frame,
        )

        output.update(
            topology_info,
        )
        """Gather Muon Multiplicity Information."""
        try:
            muon_multiplicity_dict = self._get_muon_multiplicity(
                frame,
                primary_particle.pdg_encoding,
            )
            output.update(
                muon_multiplicity_dict,
            )
        except:
            print("Muon Multiplicity Information Not Found")
        """
        Gather Muon Stochasticity and Lateral Spread Information
        """

        self._get_lateral_and_stochasticity_info(
            frame,
        )

        output.update(
            {
                "primary_rms_energy": frame["PrimaryShowerProfile_energy"][
                    "lateral_rms"
                ],
                "primary_rms3_energy": frame["PrimaryShowerProfile_energy"][
                    "lateral_rms3"
                ],
                "primary_rms_MCPE": frame["PrimaryShowerProfile_MCPEs"][
                    "lateral_rms"
                ],
                "primary_rms3_MCPE": frame["PrimaryShowerProfile_MCPEs"][
                    "lateral_rms3"
                ],
                "primary_most_lateral_deposit": frame[
                    "PrimaryShowerProfile_MCPEs"
                ]["most_lateral_deposit"],
                "leading_rms_energy": frame["LeadingShowerProfile_energy"][
                    "lateral_rms"
                ],
                "leading_rms3_energy": frame["LeadingShowerProfile_energy"][
                    "lateral_rms3"
                ],
                "leading_rms_MCPE": frame["LeadingShowerProfile_MCPEs"][
                    "lateral_rms"
                ],
                "leading_rms3_MCPE": frame["LeadingShowerProfile_MCPEs"][
                    "lateral_rms3"
                ],
                "leading_most_lateral_deposit": frame[
                    "LeadingShowerProfile_MCPEs"
                ]["most_lateral_deposit"],
                # "primary_stochasticity_std_energy": frame['PrimaryShowerProfile_energy']['stochasticity_std'],
                # "primary_stochasticity_pomean_energy": frame['PrimaryShowerProfile_energy']['stochasticity_ratio_above_mean'],
                # "primary_stochasticity_pomedian_energy": frame['PrimaryShowerProfile_energy']['stochasticity_ratio_above_median'],
                # "primary_stochasticity_std_MCPE": frame['PrimaryShowerProfile_MCPEs']['stochasticity_std'],
                # "primary_stochasticity_pomean_MCPE": frame['PrimaryShowerProfile_MCPEs']['stochasticity_ratio_above_mean'],
                # "primary_stochasticity_pomedian_MCPE": frame['PrimaryShowerProfile_MCPEs']['stochasticity_ratio_above_median'],
                "leading_stochasticity_std_energy": frame[
                    "LeadingShowerProfile_energy"
                ]["stochasticity_std"],
                "leading_stochasticity_pomean_energy": frame[
                    "LeadingShowerProfile_energy"
                ]["stochasticity_ratio_above_mean"],
                "leading_stochasticity_pomedian_energy": frame[
                    "LeadingShowerProfile_energy"
                ]["stochasticity_ratio_above_median"],
                "leading_stochasticity_std_MCPE": frame[
                    "LeadingShowerProfile_MCPEs"
                ]["stochasticity_std"],
                "leading_stochasticity_pomean_MCPE": frame[
                    "LeadingShowerProfile_MCPEs"
                ]["stochasticity_ratio_above_mean"],
                "leading_stochasticity_pomedian_MCPE": frame[
                    "LeadingShowerProfile_MCPEs"
                ]["stochasticity_ratio_above_median"],
                "primary_length_deposited": frame["ShowerLengthDeposited"][
                    "primary_length_deposited"
                ],
                "leading_length_deposited": frame["ShowerLengthDeposited"][
                    "leading_length_deposited"
                ],
                "fraction_coincidence": frame["fraction_coincidence"].value,
            }
        )

        return output

    def _get_leading_particle(
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

    def _get_basic_event_information(
        self,
        frame,
    ):
        primary_particle = frame["PolyplopiaPrimary"]

        if np.abs(primary_particle.pdg_encoding) in [12, 14, 16]:
            # Particle Is a Neutrino
            interaction_type = frame["I3MCWeightDict"]["InteractionType"]
        else:
            # No Relevant Interaction Type
            interaction_type = -1

        return primary_particle, interaction_type

    def _event_topologies(
        self,
        frame,
        threshold="low",
    ):
        """
        Topology Definition
        Starting Vertex inside Detector Volume: 1
        Visible Starting Vertex outside Detector Volume: 2
        Starting Vertex inside Detector Volume and Meets New Parameterization: 12
        Throughgoing Event: 3 (Enters Detector)
        Skimming Event: 43 (Throughgoing Event that Doesn't Enter the Detector)
        Starting Skimming: 42 (Throughoing, Meets Visible Starting Condition)

        """
        if np.abs(frame["PolyplopiaPrimary"].pdg_encoding) in [12, 14, 16]:
            starting_metrics = get_topology_metrics(
                frame,
                self._dom_list,
                self._outer_boundar,
            )
        else:
            # Corsika - No Starting Events
            starting_metrics = [0, 0, 0, 0, 0]

        starting_metrics = np.asarray(starting_metrics, dtype=bool)
        # Skimming Event
        is_skimming_event = compute_skimming_event(
            frame,
            starting_inside=starting_metrics[0],
        )

        dict_topology = {
            "true_starting": int(
                starting_metrics[0]
            ),  # True Starting Vertex Inside Detector Volume
            "topology_primary_high": self._label_topologies(
                starting_metrics[0], starting_metrics[1], is_skimming_event
            ),
            "topology_primary_low": self._label_topologies(
                starting_metrics[0], starting_metrics[2], is_skimming_event
            ),
            "topology_leading_high": self._label_topologies(
                starting_metrics[0], starting_metrics[2], is_skimming_event
            ),
            "topology_leading_low": self._label_topologies(
                starting_metrics[0], starting_metrics[3], is_skimming_event
            ),
            "length_in_detector": frame["TrackLength_Inside_Detector"].value,
            "length_in_detector_100": frame[
                "TrackLength_Near_Detector_100"
            ].value,
            "length_in_detector_200": frame[
                "TrackLength_Near_Detector_200"
            ].value,
            "closest_approach_x": frame["ClosestApproachPosition"].x,
            "closest_approach_y": frame["ClosestApproachPosition"].y,
            "closest_approach_z": frame["ClosestApproachPosition"].z,
        }

        return dict_topology

    def _label_topologies(
        self,
        starting_inside,
        starting_visible,
        is_skimming,
    ):

        if starting_inside & (starting_visible == False):
            topology = 1
        elif starting_inside & (starting_visible == True):
            topology = 12
        elif starting_visible & (starting_inside == False):
            topology = 2
        elif starting_visible & is_skimming:
            topology = 42
        elif (
            (starting_inside == False)
            & (starting_visible == False)
            & (is_skimming == False)
        ):
            topology = 3
        elif (
            (starting_inside == False)
            & (starting_visible == False)
            & (is_skimming == True)
        ):
            topology = 43

        return topology

    def _get_muon_multiplicity(
        self,
        frame,
        pdg,
    ):
        """Muon Multiplicity Information NuMu -> 1 NuTau -> 1 NuE -> 1 Corsika
        -> Count Muons."""

        if np.abs(pdg) in [12, 14, 16]:
            muon_multiplicity = 1
            muon_multiplicity_cylinder = 1
            deposited_muon_multiplicity = 1
            deposited_muon_multiplicity_residual_primary = 0
            deposited_muon_multiplicity_residual_energy = 0
            deposited_muon_multiplicity_residual_charge = 0
        else:
            muon_multiplicity = frame["MultiplicityInfo"][
                "muon_multiplicity_surface"
            ]
            muon_multiplicity_cylinder = frame["MultiplicityInfo"][
                "muon_multiplicity_cylinder"
            ]
            deposited_muon_multiplicity = frame["MultiplicityInfo"][
                "deposited_muon_multiplicity"
            ]
            deposited_muon_multiplicity_residual_primary = frame[
                "MultiplicityInfo"
            ]["primary_residual_multiplicity"]
            deposited_muon_multiplicity_residual_energy = frame[
                "MultiplicityInfo"
            ]["leading_residual_multiplicity"]
            deposited_muon_multiplicity_residual_charge = frame[
                "MultiplicityInfo"
            ]["charge_residual_multiplicity"]

        muon_mult = {
            "muon_multiplicity": muon_multiplicity,
            "muon_multiplicity_cylinder": muon_multiplicity_cylinder,
            "deposited_muon_multiplicity": deposited_muon_multiplicity,
            "deposited_muon_multiplicity_residual_primary": deposited_muon_multiplicity_residual_primary,
            "deposited_muon_multiplicity_residual_energy": deposited_muon_multiplicity_residual_energy,
            "deposited_muon_multiplicity_residual_charge": deposited_muon_multiplicity_residual_charge,
        }

        return muon_mult

    def _get_lateral_and_stochasticity_info(
        self,
        frame,
    ):

        make_shower_and_stochasticity_info(frame)

    def _generation_spectrum_correction(
        self,
        frame,
    ):

        pdg_encoding = frame["PolyplopiaPrimary"].pdg_encoding

        if np.abs(pdg_encoding) in [12, 14, 16]:
            # Neutrino
            power_law_index = np.abs(frame["I3MCWeightDict"]["PowerLawIndex"])
            primary_energy = frame["I3MCWeightDict"]["PrimaryNeutrinoEnergy"]
            min_energy = 10 ** frame["I3MCWeightDict"]["MinEnergyLog"]
        else:
            power_law_index = np.abs(
                frame["CorsikaWeightMap"]["PrimarySpectralIndex"]
            )
            primary_energy = frame["CorsikaWeightMap"]["PrimaryEnergy"]
            min_energy = frame["CorsikaWeightMap"]["EnergyPrimaryMin"]

        return 1 / ((primary_energy / min_energy) ** (1 - power_law_index))

    # Utility methods
    def _find_data_type(self, mc: bool, input_file: str) -> str:
        """Determine the data type.

        Args:
            mc: Whether `input_file` is Monte Carlo simulation.
            input_file: Path to I3-file.

        Returns:
            The simulation/data type.
        """
        print(len(input_file))
        # @TODO: Rewrite to automatically infer `mc` from `input_file`?
        if not mc:
            sim_type = "data"
        elif "muon" in input_file:
            sim_type = "muongun"
        elif "corsika" in input_file:
            sim_type = "corsika"
        # elif "genie" in input_file or "nu" in input_file.lower():
        #    sim_type = "genie"
        elif "noise" in input_file:
            sim_type = "noise"
        elif "L2" in input_file:  # not robust
            sim_type = "dbang"
        else:
            sim_type = "NuGen"
        return sim_type
