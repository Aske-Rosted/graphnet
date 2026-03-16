"""Filter classes for filtering I3-frames when converting I3-files."""
from abc import abstractmethod
from graphnet.utilities.logging import Logger
from typing import List

from graphnet.utilities.imports import has_icecube_package

import numpy as np
if has_icecube_package():
    from icecube import icetray
    from icecube import dataio
    from icecube import dataclasses
    from icecube import simclasses
from collections import defaultdict 
import pandas as pd

class I3Filter(Logger):
    """A generic filter for I3-frames."""

    @abstractmethod
    def _keep_frame(self, frame: "icetray.I3Frame") -> bool:
        """Return True if the frame is kept, False otherwise.

        Args:
            frame: I3-frame
                The I3-frame to check.

        Returns:
            bool: True if the frame is kept, False otherwise.
        """
        raise NotImplementedError

    def __call__(self, frame: "icetray.I3Frame") -> bool:
        """Return True if the frame passes the filter, False otherwise.

        Args:
            frame: I3-frame
                The I3-frame to check.

        Returns:
            bool: True if the frame passes the filter, False otherwise.
        """
        pass_flag = self._keep_frame(frame)
        try:
            assert isinstance(pass_flag, bool)
        except AssertionError:
            raise TypeError(
                f"Expected _pass_frame to return bool, got {type(pass_flag)}."
            )
        return pass_flag


class NullSplitI3Filter(I3Filter):
    """A filter that skips all null-split frames."""

    def _keep_frame(self, frame: "icetray.I3Frame") -> bool:
        """Check that frame is not a null-split frame.

        returns False if the frame is a null-split frame, True otherwise.

        Args:
            frame: I3-frame
                The I3-frame to check.
        """
        if frame.Has("I3EventHeader"):
            if frame["I3EventHeader"].sub_event_stream == "NullSplit":
                return False
        return True


class I3FilterMask(I3Filter):
    """checks list of filters from the FilterMask in I3 frames."""

    def __init__(self, filter_names: List[str], filter_any: bool = True):
        """Initialize I3FilterMask.

        Args:
        filter_names: List[str]
            A list of filter names to check for.
        filter_any: bool
            standard: True
            If True, the frame is kept if any of the filter names are present.
            If False, the frame is kept if all of the filter names are present.
        """
        self._filter_names = filter_names
        self._filter_any = filter_any

    def _keep_frame(self, frame: "icetray.I3Frame") -> bool:
        """Check if current frame should be kept.

        Args:
            frame: I3-frame
                The I3-frame to check.
        """
        if "FilterMask" in frame:
            if (
                self._filter_any is True
            ):  # Require any of the filters to pass to keep the frame
                bool_list = []
                for filter_name in self._filter_names:
                    if filter_name not in frame["FilterMask"]:
                        self.warning_once(
                            f"FilterMask {filter_name} not found in frame. skipping filter."
                        )
                        continue
                    elif frame["FilterMask"][filter].condition_passed is True:
                        bool_list.append(True)
                    else:
                        bool_list.append(False)
                if len(bool_list) == 0:
                    self.warning_once(
                        "None of the FilterMask filters found in frame, FilterMask filters will not be applied."
                    )
                return any(bool_list) or len(bool_list) == 0
            else:  # Require all filters to pass in order to keep the frame.
                for filter_name in self._filter_names:
                    if filter_name not in frame["FilterMask"]:
                        self.warning_once(
                            f"FilterMask {filter_name} not found in frame, skipping filter."
                        )
                        continue
                    elif frame["FilterMask"][filter].condition_passed is True:
                        continue  # current filter passed, continue to next filter
                    else:
                        return (
                            False  # current filter failed so frame is skipped.
                        )
                return True
        else:
            self.warning_once(
                "FilterMask not found in frame, FilterMask filters will not be applied."
            )
            return True

class ChargeFilter(I3Filter):
    """Passes if charge meets charge_cut threshold."""

    def __init__(self, min_charge: float = 1e4, pulsemap: str = 'SplitInIcePulses'):
        """Initialize ChargeFilter.
        
        min_charge: float
            The minimum charge threshold for the event to pass the filter.
        pulsemap: str
            The name of the pulsemap to use for calculating the charge.
        """
        self._charge_cut = min_charge
        self._pulsemap = pulsemap

        gcd_file = '/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_2020.Run134142.Pass2_V0.i3.gz'
        f = dataio.I3File(gcd_file)
        self._cal = f.pop_frame(icetray.I3Frame.Calibration)['I3Calibration']

    def _keep_frame(self, frame: "icetray.I3Frame") -> bool:
        """Check if current frame should be kept.

        Args:
            frame: I3-frame
                The I3-frame to check.
        """
        oms = defaultdict(list)

        try:
            data = frame['SplitInIcePulses'].apply(frame)
        except:
            # no splitinicepulses
            return False
        
        try:
            mctree = frame['I3MCTree']
        except:
            # mctree issue
            print('mctree issue')
            return False
        
        om_keys = data.keys()

        for om_key in om_keys:

            pulses = data[om_key]

            rde = self._cal.dom_cal[om_key].relative_dom_eff

            for _,pulse in enumerate(pulses):

                oms['time'].append(pulse.time)
                oms['charge'].append(pulse.charge)
                oms['string'].append(om_key.string)
                oms['dom'].append(om_key.om)
                oms['rde'].append(rde)

        reco_pulses = pd.DataFrame(
            {
            "string":oms['string'], 
            "dom":oms['dom'],
            't':oms["time"], 
            'charge':oms['charge'],
            'rde':oms['rde'],
            },
        )


        event_id = frame['I3EventHeader'].event_id

        remove_deepcore = reco_pulses[(reco_pulses['string'] < 79) & (reco_pulses['rde'] <= 1.1)]
        total_charge = remove_deepcore['charge'].sum()

        q_total = remove_deepcore.groupby(["string", "dom"], as_index=False)['charge'].sum()

        q_total_bubble_cut = q_total[q_total['charge'] < total_charge/2]

        try:
            frame['HQTOT'] = dataclasses.I3Double(q_total_bubble_cut.charge.sum())
        except:
            del frame['HQTOT']

            frame['HQTOT'] = dataclasses.I3Double(q_total_bubble_cut.charge.sum())


        if frame['HQTOT'].value < self._charge_cut:
            return False
        else:
            return True

class CutZenith(I3Filter):
    """Passes Events Below the Zenith Cut."""

    def __init__(self, max_zenith: float = np.radians(100)):
        """Initialize .

        Args:
        max_zenith: float
            The maximum zenith angle for the event to pass the filter.
        """
        self._max_zenith = max_zenith

    def _keep_frame(self, frame: "icetray.I3Frame") -> bool:
        """
        If the event's zenith angle is below the maximum threshold, keep it.
        """

        if frame['PolyplopiaPrimary']['zenith'] > self._max_zenith:
            return False
        
        return True
    

class CoincidenceDominant(I3Filter):
    """Filters Out Events where the C."""

    def __init__(self):
        """Initialize ChargeFilter.

        Args:
        filter_names: List[str]
            A list of filter names to check for.
        filter_any: bool
            standard: True
            If True, the frame is kept if any of the filter names are present.
            If False, the frame is kept if all of the filter names are present.
        """

    def _keep_frame(self, frame: "icetray.I3Frame") -> bool:
        """Check if current frame should be kept.

        Args:
            frame: I3-frame
                The I3-frame to check.
        """

        # Not Implemented Yet, But Might Be Useful
        
        return True
    

class ShortenFile(I3Filter):
    """For Debugging: Shortening Tests of Files 
    that Take Too Long to Process."""

    def __init__(self, max_event_id: int = 100):
        """Initialize ShortenFile.

        Args:
        max_event_id: int
            The maximum event ID to keep.
        """
        self._max_event_id = max_event_id

    def _keep_frame(self, frame: "icetray.I3Frame") -> bool:
        """Check if current frame should be kept.

        Args:
            frame: I3-frame
                The I3-frame to check.
        """

        if frame['I3EventHeader'].event_id > self._max_event_id:
            return False
        else:
            return True