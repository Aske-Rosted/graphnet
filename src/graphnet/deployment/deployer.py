"""Contains the graphnet deployment module."""

import random
from abc import abstractmethod, ABC
from typing import TYPE_CHECKING, List, Union, Sequence, Any
import time


from pathos.multiprocessing import ProcessingPool as Pool

import multiprocess.context as ctx

ctx._force_start_method("spawn")

from graphnet.utilities.imports import has_torch_package
from .deployment_module import DeploymentModule
from graphnet.utilities.logging import Logger


if has_torch_package or TYPE_CHECKING:
    import torch


class Deployer(ABC, Logger):
    """A generic baseclass for applying `DeploymentModules` to analysis files.

    Modules are applied in the order that they appear in `modules`.
    """

    @abstractmethod
    def _process_files(
        self,
        settings: Any,
    ) -> None:
        """Process a single file.

        If n_workers > 1, this function is run in parallel n_worker times. Each
        worker will loop over an allocated set of files.
        """
        raise NotImplementedError

    @abstractmethod
    def _prepare_settings(
        self, input_files: List[str], output_folder: str
    ) -> List[Any]:
        """Produce a list of inputs for each worker.

        This function must produce and return a list of arguments to each
        worker.
        """
        raise NotImplementedError

    def __init__(
        self,
        modules: Union[DeploymentModule, Sequence[DeploymentModule]],
        n_workers: int = 1,
    ) -> None:
        """Initialize `Deployer`.

        Will apply `DeploymentModules` to files in the order in which they
        appear in `modules`. Each module is run independently.

        Args:
            modules: List of `DeploymentModules`.
                              Order of appearence in the list determines order
                              of deployment.
            n_workers: Number of workers. The deployer will divide the number
                       of input files across workers. Defaults to 1.
        """
        super().__init__(name=__name__, class_name=self.__class__.__name__)
        # This makes sure that one worker cannot access more
        # than 1 core's worth of compute.

        if torch.get_num_interop_threads() > 1:
            torch.set_num_interop_threads(1)
        if torch.get_num_threads() > 1:
            torch.set_num_threads(1)

        # Check
        if isinstance(modules, list):
            self._modules = modules
        else:
            self._modules = [modules]

        # Member Variables
        self._n_workers = n_workers

    def _launch_jobs(self, settings: List[Any]) -> None:
        """Will launch jobs in parallel if n_workers > 1, else run on main."""
        if self._n_workers > 1:

            # processes = []
            # collapse settings into an iterable
            i_settings = {}
            with Pool(self._n_workers) as pool:
                pool.map(self._process_files, settings)

            # for i in range(self._n_workers):
            #     processes.append(
            #         multiprocessing.Process(
            #             target=self._process_files,
            #             args=[settings[i]],  # type: ignore
            #         )
            #     )

            # for process in processes:
            #     process.start()

            # for process in processes:
            #     process.join()
            #     process.close()
        else:
            self._process_files(settings[0])

    def run(
        self,
        input_files: Union[List[str], str],
        output_folder: str,
        ignore_folder_exists: bool = False,
    ) -> None:
        """Apply `modules` to input files.

        Args:
            input_files: Path(s) to i3 file(s) that you wish to
                         apply the graphnet modules to.
            output_folder: The output folder to which the i3 files are written.
        """
        start_time = time.time()
        if isinstance(self._gcd_file, list):
            assert len(self._gcd_file) == len(
                input_files
            ), "GCD file must be same length as input files"

        if isinstance(input_files, list) & isinstance(self._gcd_file, str):
            random.Random(42).shuffle(input_files)
        elif isinstance(input_files, list) & isinstance(self._gcd_file, list):
            # Shuffle the input files and gcd files together
            input_files = list(zip(input_files, self._gcd_file))
            random.Random(42).shuffle(input_files)
            # Unzip the input files and gcd files
            input_files, self._gcd_file = zip(*input_files)
            input_files = list(input_files)
            self._gcd_file = list(self._gcd_file)

        else:
            input_files = [input_files]

        settings = self._prepare_settings(
            input_files=input_files,
            output_folder=output_folder,
            ignore_folder_exists=ignore_folder_exists,
        )
        assert (
            len(settings) == self._n_workers
        ), f"Number of settings must match number of workers but got {len(settings)} and {self._n_workers}"

        self.info(
            f"""processing {len(input_files)} files \n
                using {self._n_workers} workers"""
        )
        self._launch_jobs(settings)
        self.info(
            f"""Processing {len(input_files)} files was completed in \n
         {time.time() - start_time} seconds using {self._n_workers} cores."""
        )
