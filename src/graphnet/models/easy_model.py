"""Suggested Model subclass that enables simple user syntax."""

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union, Type

import numpy as np
import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import Tensor
from torch.nn import ModuleList
from torch.optim import Adam
from torch.utils.data import DataLoader, SequentialSampler
from torch_geometric.data import Data, Batch
import pandas as pd
from pytorch_lightning.loggers import Logger as LightningLogger

from graphnet.training.callbacks import ProgressBar
from graphnet.models.model import Model
from graphnet.models.task import StandardLearnedTask
from graphnet.models.task.multitask_utils import LossWeightBalancing


class EasySyntax(Model):
    """A suggested Model class that comes with simple user syntax.

    This class delivers simple user syntax for training and prediction, while
    imposing minimal constraints on structure.
    """

    def __init__(
        self,
        *,
        tasks: Union[StandardLearnedTask, List[StandardLearnedTask]],
        optimizer_class: Type[torch.optim.Optimizer] = Adam,
        optimizer_kwargs: Optional[Dict] = None,
        scheduler_class: Optional[type] = None,
        scheduler_kwargs: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None,
        additional_attributes: Optional[List[str]] = None,
    ) -> None:
        """Construct `StandardModel`."""
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

        # Check(s)
        if not isinstance(tasks, (list, tuple)):
            tasks = [tasks]

        # Member variable(s)
        self._tasks = ModuleList(tasks)
        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs or dict()
        self._scheduler_class = scheduler_class
        self._scheduler_kwargs = scheduler_kwargs or dict()
        self._scheduler_config = scheduler_config or dict()
        self._additional_attributes = additional_attributes or []

        self.validate_tasks()

    def compute_loss(
        self, preds: Tensor, data: List[Data], verbose: bool = False
    ) -> Tensor:
        """Compute and sum losses across tasks."""
        raise NotImplementedError

    def forward(
        self, data: Union[Data, List[Data]]
    ) -> List[Union[Tensor, Data]]:
        """Forward pass, chaining model components."""
        raise NotImplementedError

    def shared_step(
        self, batch: List[Data], batch_idx: int
    ) -> tuple[Tensor, Tensor]:
        """Perform shared step.

        Applies the forward pass and the following loss calculation, shared
        between the training and validation step.
        """
        raise NotImplementedError

    def validate_tasks(self) -> None:
        """Verify that self._tasks contain compatible elements."""
        raise NotImplementedError

    @staticmethod
    def _construct_trainer(
        max_epochs: int = 10,
        gpus: Optional[Union[List[int], int]] = None,
        callbacks: Optional[List[Callback]] = None,
        logger: Optional[LightningLogger] = None,
        log_every_n_steps: int = 1,
        gradient_clip_val: Optional[float] = None,
        distribution_strategy: Optional[str] = "ddp",
        **trainer_kwargs: Any,
    ) -> Trainer:
        if gpus:
            accelerator = "gpu"
            devices = gpus
        else:
            accelerator = "cpu"
            devices = 1

        trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            callbacks=callbacks,
            log_every_n_steps=log_every_n_steps,
            logger=logger,
            gradient_clip_val=gradient_clip_val,
            strategy=distribution_strategy,
            **trainer_kwargs,
        )

        return trainer

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        *,
        max_epochs: int = 10,
        early_stopping_patience: int = 5,
        gpus: Optional[Union[List[int], int]] = None,
        callbacks: Optional[List[Callback]] = None,
        ckpt_path: Optional[str] = None,
        logger: Optional[LightningLogger] = None,
        log_every_n_steps: int = 1,
        gradient_clip_val: Optional[float] = None,
        distribution_strategy: Optional[str] = "ddp",
        **trainer_kwargs: Any,
    ) -> None:
        """Fit `StandardModel` using `pytorch_lightning.Trainer`."""
        # Checks
        if callbacks is None:
            # We create the bare-minimum callbacks for you.
            callbacks = self._create_default_callbacks(
                val_dataloader=val_dataloader,
                early_stopping_patience=early_stopping_patience,
            )
            self.debug("No Callbacks specified. Default callbacks added.")
        else:
            # You are on your own!
            self.debug("Initializing training with user-provided callbacks.")
            pass
        self._print_callbacks(callbacks)
        has_early_stopping = self._contains_callback(callbacks, EarlyStopping)
        has_model_checkpoint = self._contains_callback(
            callbacks, ModelCheckpoint
        )

        if (has_early_stopping) & (has_model_checkpoint is False):
            self.warning(
                "No ModelCheckpoint found in callbacks. Best-fit model will"
                " not automatically be loaded after training!"
                ""
            )

        self.train(mode=True)
        trainer = self._construct_trainer(
            max_epochs=max_epochs,
            gpus=gpus,
            callbacks=callbacks,
            logger=logger,
            log_every_n_steps=log_every_n_steps,
            gradient_clip_val=gradient_clip_val,
            distribution_strategy=distribution_strategy,
            **trainer_kwargs,
        )

        try:
            trainer.fit(
                self, train_dataloader, val_dataloader, ckpt_path=ckpt_path
            )
        except KeyboardInterrupt:
            self.warning("[ctrl+c] Exiting gracefully.")
            pass

        # Load weights from best-fit model after training if possible
        if has_early_stopping & has_model_checkpoint:
            for callback in callbacks:
                if isinstance(callback, ModelCheckpoint):
                    checkpoint_callback = callback
            self.load_state_dict(
                torch.load(checkpoint_callback.best_model_path)["state_dict"]
            )
            self.info("Best-fit weights from EarlyStopping loaded.")

    def _print_callbacks(self, callbacks: List[Callback]) -> None:
        callback_names = []
        for cbck in callbacks:
            callback_names.append(cbck.__class__.__name__)
        self.info(
            f"Training initiated with callbacks: {', '.join(callback_names)}"
        )

    def _contains_callback(
        self, callbacks: List[Callback], callback: Callback
    ) -> bool:
        """Check if `callback` is in `callbacks`."""
        for cbck in callbacks:
            if isinstance(cbck, callback):
                return True
        return False

    @property
    def target_labels(self) -> List[str]:
        """Return target label."""
        return [label for task in self._tasks for label in task._target_labels]

    @property
    def prediction_labels(self) -> List[str]:
        """Return prediction labels."""
        return [
            label for task in self._tasks for label in task._prediction_labels
        ]

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure the model's optimizer(s)."""
        if hasattr(self, "loss_weight_balancing"):
            self.info("Using Loss Weight Balancing for multi-task learning.")
            # seperate the parameters of the loss weight balancing from the rest
            loss_weight_params = list(self.loss_weight_balancing.noise_params)
            other_params = list(
                set(self.parameters()) - set(loss_weight_params)
            )
            param_groups = [
                {"params": other_params, **self._optimizer_kwargs},
                {"params": loss_weight_params, **self._optimizer_kwargs},
            ]
            # scale the lr down by two orders of magnitude for the loss weight balancing parameters
            param_groups[-1]["lr"] = param_groups[-1].get("lr", 0.001) * 1e-2
            optimizer = self._optimizer_class(param_groups)
        else:
            optimizer = self._optimizer_class(
                self.parameters(), **self._optimizer_kwargs
            )
        config = {
            "optimizer": optimizer,
        }
        if self._scheduler_class is not None:
            scheduler = self._scheduler_class(
                optimizer, **self._scheduler_kwargs
            )
            config.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        **self._scheduler_config,
                    },
                }
            )
        return config

    def training_step(
        self, train_batch: Union[Data, List[Data]], batch_idx: int
    ) -> Tensor:
        """Perform training step."""
        if isinstance(train_batch, Data):
            train_batch = [train_batch]
        loss, loss_stack = self.shared_step(train_batch, batch_idx)
        self.log(
            "train_loss",
            loss,
            batch_size=self._get_batch_size(train_batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, prog_bar=True, on_step=True)
        return loss

    def validation_step(
        self, val_batch: Union[Data, List[Data]], batch_idx: int
    ) -> Tensor:
        """Perform validation step."""
        with torch.no_grad():
            if isinstance(val_batch, Data):
                val_batch = [val_batch]
            loss, loss_stack = self.shared_step(val_batch, batch_idx)
            self.log(
                "val_loss",
                loss,
                batch_size=self._get_batch_size(val_batch),
                prog_bar=True,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )
        return loss

    def predict_step(
        self, batch, batch_idx, dataloader_idx=0
    ) -> Dict[str, Tensor]:
        return_dict = {}
        with torch.no_grad():  # redundant with self.inference()?
            if isinstance(batch, Data):
                batch = [batch]
            preds = self.forward(batch)
            for pred, task in zip(preds, self._tasks):
                names = task._prediction_labels
                for i, name in enumerate(names):
                    return_dict[name] = pred[:, i].detach().cpu()

        if len(self._additional_attributes) > 0:
            for attr in self._additional_attributes:
                attr_temp = []
                for b in batch:
                    attribute = b[attr]
                    if not isinstance(attribute, torch.Tensor):
                        attribute = torch.as_tensor(attribute)
                    attribute = attribute.detach().cpu()
                    if attribute.dim() == 0:
                        attribute = attribute.unsqueeze(0)
                    attr_temp.append(attribute)

                attribute = torch.cat(attr_temp, dim=0)
                return_dict[attr] = attribute

        return return_dict

    def inference(self) -> None:
        """Activate inference mode."""
        for task in self._tasks:
            task.inference()

    def train(self, mode: bool = True) -> "Model":
        """Deactivate inference mode."""
        super().train(mode)
        if mode:
            for task in self._tasks:
                task.train_eval()
        return self

    def predict(
        self,
        dataloader: DataLoader,
        gpus: Optional[Union[List[int], int]] = None,
        distribution_strategy: Optional[str] = "auto",
        **trainer_kwargs: Any,
    ) -> Dict[str, np.ndarray]:
        """Return predictions for `dataloader`."""
        self.inference()
        self.train(mode=False)

        callbacks = self._create_default_callbacks(
            val_dataloader=None,
        )

        inference_trainer = self._construct_trainer(
            gpus=gpus,
            distribution_strategy=distribution_strategy,
            callbacks=callbacks,
            **trainer_kwargs,
        )

        outputs = inference_trainer.predict(self, dataloader)
        if len(outputs) == 0:
            raise ValueError("Got no predictions")
        # stack the dictionaries

        outputs = {
            key: torch.cat([out[key] for out in outputs], dim=0).numpy()
            for key in outputs[0].keys()
        }

        return outputs

    def predict_as_dataframe(
        self,
        dataloader: DataLoader,
        prediction_columns: Optional[List[str]] = None,
        *,
        # additional_attributes: Optional[List[str]] = None,
        gpus: Optional[Union[List[int], int]] = None,
        distribution_strategy: Optional[str] = "auto",
        **trainer_kwargs: Any,
    ) -> pd.DataFrame:
        """Return predictions for `dataloader` as a DataFrame.

        Include `additional_attributes` as additional columns in the output
        DataFrame.
        """
        if prediction_columns is None:
            prediction_columns = self.prediction_labels

        self.info(f"Column names for predictions are: \n {prediction_columns}")
        self.warning_once(
            "DEPRECATION WARNING 2.0: predict_as_dataframe will be deprecated as it has just become a simple wrapper around predict. Please use predict instead and convert the output to a DataFrame yourself."
        )
        predictions_torch = self.predict(
            dataloader=dataloader,
            gpus=gpus,
            distribution_strategy=distribution_strategy,
            **trainer_kwargs,
        )
        results = pd.DataFrame.from_dict(predictions_torch)

        return results

    def _create_default_callbacks(
        self,
        val_dataloader: DataLoader,
        early_stopping_patience: Optional[int] = None,
    ) -> List:
        """Create default callbacks.

        Used in cases where no callbacks are specified by the user in .fit
        """
        callbacks = [ProgressBar()]
        if val_dataloader is not None:
            assert early_stopping_patience is not None
            # Add Early Stopping
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                )
            )
            # Add Model Check Point
            callbacks.append(
                ModelCheckpoint(
                    save_top_k=1,
                    monitor="val_loss",
                    mode="min",
                    filename=f"{self.backbone.__class__.__name__}"
                    + "-{epoch}-{val_loss:.2f}-{train_loss:.2f}",
                )
            )
            self.info(
                "EarlyStopping has been added"
                f" with a patience of {early_stopping_patience}."
            )
        return callbacks

    def _add_early_stopping(
        self, val_dataloader: DataLoader, callbacks: List
    ) -> List:
        if val_dataloader is None:
            return callbacks
        has_early_stopping = False
        assert isinstance(callbacks, list)
        for callback in callbacks:
            if isinstance(callback, EarlyStopping):
                has_early_stopping = True

        if not has_early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                )
            )
            self.warning_once(
                "Got validation dataloader but no EarlyStopping callback. An "
                "EarlyStopping callback has been added automatically with "
                "patience=5 and monitor = 'val_loss'."
            )
        return callbacks
