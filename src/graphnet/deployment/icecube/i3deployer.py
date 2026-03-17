"""Contains an IceCube-specific implementation of Deployer."""

from typing import TYPE_CHECKING, List, Union, Sequence
import os
import numpy as np

from graphnet.utilities.imports import has_icecube_package
from graphnet.deployment.icecube import I3InferenceModule
from graphnet.data.dataclasses import Settings
from graphnet.deployment import Deployer
import logging

import time

if has_icecube_package() or TYPE_CHECKING:
    from icecube import (
        icetray,
        dataio,
        snowstorm,
        dataclasses,
        simclasses,
    )  # pyright: reportMissingImports=false
    from I3Tray import I3Tray


class I3Deployer(Deployer):
    """A generic baseclass for applying `DeploymentModules` to analysis files.

    Modules are applied in the order that they appear in `modules`.
    """

    def __init__(
        self,
        modules: Union[I3InferenceModule, Sequence[I3InferenceModule]],
        gcd_file: Union[str, list[str]],
        n_workers: int = 1,
        worker_per_file: bool = True,
    ) -> None:
        """Initialize `Deployer`.

        Will apply `DeploymentModules` to files in the order in which they
        appear in `modules`. Each module is run independently.

        Args:
            modules: List of `DeploymentModules`.
                              Order of appearence in the list determines order
                              of deployment.
            gcd_file: path to gcd file.
            n_workers: Number of workers. The deployer will divide the number
                       of input files across workers. Defaults to 1.
        """
        super().__init__(modules=modules, n_workers=n_workers)

        # Member variables
        self._gcd_file = gcd_file
        self._worker_per_file = worker_per_file

    def _process_files(
        self,
        settings: Settings,
    ) -> None:
        """Will start an IceTray read/write chain with graphnet modules.

        If n_workers > 1, this function is run in parallel n_worker times. Each
        worker will loop over an allocated set of i3 files. The new i3 files
        will appear as copies of the original i3 files but with reconstructions
        added. Original i3 files are left untouched.
        """
        if isinstance(settings.gcd_file, str):
            settings.gcd_file = [settings.gcd_file] * len(settings.i3_files)
        assert len(settings.i3_files) == len(
            settings.gcd_file
        ), "Number of i3 files must match number of gcd files."
        for i3_file, gcd_file in zip(settings.i3_files, settings.gcd_file):
            message_str = (
                f"Processing {i3_file}\n"
                + f"Using gcd file {gcd_file}\n"
                + f"Output folder {settings.output_folder}\n"
            )
            self.info(message_str)
            ts = time.time()
            tray = I3Tray()
            tray.context["I3FileStager"] = dataio.get_stagers()
            tray.AddModule(
                "I3Reader",
                "reader",
                FilenameList=[gcd_file, i3_file],
            )
            for i3_module in settings.modules:
                if isinstance(i3_module, I3InferenceModule):
                    # Set the GCD file for each module extractor
                    for extractor in i3_module._i3_extractors:
                        extractor.set_gcd(gcd_file)

                tray.AddModule(i3_module)
            tray.Add(
                "I3Writer",
                Streams=[
                    icetray.I3Frame.DAQ,
                    icetray.I3Frame.Physics,
                    icetray.I3Frame.Simulation,
                    icetray.I3Frame.Stream("M"),
                ],
                filename=settings.output_folder + "/" + i3_file.split("/")[-1],
                DropOrphanStreams=[icetray.I3Frame.DAQ],
            )

            tray.Execute()
            tray.Finish()
            te = time.time()
            self.info(
                f"Finished processing {i3_file} in {te - ts:.2f} seconds.\n"
            )
            self.info(
                f"Output file {settings.output_folder + '/' + i3_file.split('/')[-1]}\n"
            )
        return

    def _prepare_settings(
        self,
        input_files: List[str],
        output_folder: str,
        ignore_folder_exists: bool = False,
    ) -> List[Settings]:
        """Will prepare the settings for each worker."""
        if ignore_folder_exists is False:
            try:
                os.makedirs(output_folder)
            except FileExistsError as e:
                self.error(
                    f"{output_folder} already exists. To avoid overwriting "
                    "existing files, the process has been stopped."
                )
                raise e
        else:
            logging.info(f"ignoring existing folder {output_folder}")
            os.makedirs(output_folder, exist_ok=True)
        if isinstance(self._gcd_file, list):
            assert len(self._gcd_file) == len(
                input_files
            ), "Number of gcd files must match number of input files."

        if self._n_workers > len(input_files):
            self._n_workers = len(input_files)
        if self._n_workers > 1:
            if isinstance(self._gcd_file, list):
                file_batches = np.array_split(
                    np.vstack([input_files, self._gcd_file]),
                    self._n_workers,
                    axis=-1,
                )
                gcd_batches = [x[1] for x in file_batches]
                file_batches = [x[0] for x in file_batches]
            else:
                file_batches = np.array_split(input_files, self._n_workers)
                gcd_batches = [self._gcd_file] * self._n_workers

            settings: List[Settings] = []
            for i in range(self._n_workers):
                settings.append(
                    Settings(
                        file_batches[i],
                        gcd_batches[i],
                        output_folder,
                        self._modules,
                    )
                )
        else:
            settings = [
                Settings(
                    input_files,
                    self._gcd_file,
                    output_folder,
                    self._modules,
                )
            ]

        return settings
