"""Suggested Model subclass that enables simple user syntax."""

from typing import Any, Dict, List, Optional, Tuple, Union, Type

import numpy as np
import torch
import torch.distributed as dist
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import Tensor
from torch.nn import ModuleList
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch_geometric.data import Data
import pandas as pd
from pytorch_lightning.loggers import Logger as LightningLogger

from graphnet.training.callbacks import ProgressBar
from graphnet.models.model import Model
from graphnet.models.task import StandardLearnedTask


class EasySyntax(Model):
    """A suggested Model class that comes with simple user syntax.

    This class delivers simple user syntax for training and prediction,
    while imposing minimal constraints on structure.
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
        log_on_epoch: bool = True,
        log_on_step: bool = False,
    ) -> None:
        """Construct `StandardModel`.

        Args:
            tasks: Task(s) appended as the head(s) of the model, defining
                the prediction target(s) and loss(es).
            optimizer_class: Optimizer class used during training.
            optimizer_kwargs: Keyword arguments passed to `optimizer_class`.
            scheduler_class: Learning-rate scheduler class. If `None`, no
                scheduler is used.
            scheduler_kwargs: Keyword arguments passed to `scheduler_class`.
            scheduler_config: Additional configuration for how the scheduler
                is invoked by PyTorch Lightning (e.g. `interval`, `frequency`).
            log_on_epoch: If `True`, logs the training loss on epoch end.
            log_on_step: If `True`, logs the training loss on step end.
                per-batch training loss under `train_loss_step`.
        """
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
        self._predict_attributes: List[str] = []
        self._log_on_step = log_on_step
        self._log_on_epoch = log_on_epoch

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

    def shared_step(self, batch: List[Data], batch_idx: int) -> Tensor:
        """Perform shared step.

        Applies the forward pass and the following loss calculation,
        shared between the training and validation step.
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
                torch.load(
                    checkpoint_callback.best_model_path, weights_only=False
                )["state_dict"]
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
        loss = self.shared_step(train_batch, batch_idx)
        self.log(
            "train_loss",
            loss,
            batch_size=self._get_batch_size(train_batch),
            prog_bar=True,
            on_epoch=self._log_on_epoch,
            on_step=self._log_on_step,
            sync_dist=True,
        )

        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, prog_bar=True, on_step=True)
        return loss

    def validation_step(
        self, val_batch: Union[Data, List[Data]], batch_idx: int
    ) -> Tensor:
        """Perform validation step."""
        if isinstance(val_batch, Data):
            val_batch = [val_batch]
        loss = self.shared_step(val_batch, batch_idx)
        self.log(
            "val_loss",
            loss,
            batch_size=self._get_batch_size(val_batch),
            prog_bar=True,
            on_epoch=self._log_on_epoch,
            on_step=self._log_on_step,
            sync_dist=True,
        )
        return loss

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

    def predict_step(
        self,
        batch: Union[Data, List[Data]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Tuple[List[Tensor], Dict[str, Tensor]]:
        """Return predictions and requested attributes for `batch`.

        The attributes listed in `_predict_attributes` are read from the same
        batch as the predictions, so they stay aligned with them regardless
        of sampling, shuffling or events dropped by the `collate_fn`.
        Event-level attributes are repeated per node for node-level
        predictions.
        """
        batches = [batch] if isinstance(batch, Data) else list(batch)
        predictions = self(batches)

        n_rows = int(predictions[0].shape[0]) if len(predictions) else 0
        n_nodes = sum(int(b.num_nodes) for b in batches)
        n_events = sum(int(b.num_graphs) for b in batches)
        node_level = n_rows == n_nodes and n_nodes != n_events

        attributes: Dict[str, Tensor] = {}
        for attr in self._predict_attributes:
            values = []
            for b in batches:
                value = torch.as_tensor(b[attr]).detach().cpu().reshape(-1)
                if node_level and len(value) == b.num_graphs:
                    counts = torch.bincount(b.batch, minlength=b.num_graphs)
                    value = value.repeat_interleave(counts.cpu())
                values.append(value)
            attributes[attr] = torch.cat(values)
        return predictions, attributes

    def _predict(
        self,
        dataloader: DataLoader,
        additional_attributes: List[str],
        gpus: Optional[Union[List[int], int]],
        distribution_strategy: Optional[str],
        **trainer_kwargs: Any,
    ) -> Tuple[List[np.ndarray], Dict[str, np.ndarray]]:
        """Run inference and gather predictions and attributes."""
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

        self._predict_attributes = list(additional_attributes)
        try:
            outputs = inference_trainer.predict(self, dataloader) or []
        finally:
            self._predict_attributes = []

        predictions: List[np.ndarray] = []
        attributes: Dict[str, np.ndarray] = {}
        if len(outputs) > 0:
            nb_outputs = len(outputs[0][0])
            predictions = [
                torch.cat([out[0][ix] for out in outputs], dim=0)
                .detach()
                .cpu()
                .numpy()
                for ix in range(nb_outputs)
            ]
            attributes = {
                attr: torch.cat([out[1][attr] for out in outputs]).numpy()
                for attr in additional_attributes
            }

        # In distributed inference each rank only holds its own shard.
        if dist.is_available() and dist.is_initialized():
            shards: List[Any] = [None] * dist.get_world_size()
            dist.all_gather_object(shards, (predictions, attributes))
            shards = [shard for shard in shards if len(shard[0]) > 0]
            if len(shards) > 0:
                predictions = [
                    np.concatenate([shard[0][ix] for shard in shards])
                    for ix in range(len(shards[0][0]))
                ]
                attributes = {
                    attr: np.concatenate([shard[1][attr] for shard in shards])
                    for attr in additional_attributes
                }

        assert len(predictions), "Got no predictions"
        return predictions, attributes

    def predict(
        self,
        dataloader: DataLoader,
        gpus: Optional[Union[List[int], int]] = None,
        distribution_strategy: Optional[str] = "auto",
        **trainer_kwargs: Any,
    ) -> List[Tensor]:
        """Return predictions for `dataloader`, one tensor per task."""
        predictions, _ = self._predict(
            dataloader,
            additional_attributes=[],
            gpus=gpus,
            distribution_strategy=distribution_strategy,
            **trainer_kwargs,
        )
        return [torch.from_numpy(pred) for pred in predictions]

    def predict_as_dataframe(
        self,
        dataloader: DataLoader,
        prediction_columns: Optional[List[str]] = None,
        *,
        additional_attributes: Optional[List[str]] = None,
        gpus: Optional[Union[List[int], int]] = None,
        distribution_strategy: Optional[str] = "auto",
        **trainer_kwargs: Any,
    ) -> pd.DataFrame:
        """Return predictions for `dataloader` as a DataFrame.

        Include `additional_attributes` as additional columns in the output
        DataFrame. The attributes are collected together with the
        predictions, so any sampler, `collate_fn` or distributed strategy
        may be used.
        """
        if prediction_columns is None:
            prediction_columns = self.prediction_labels
        additional_attributes = list(additional_attributes or [])

        self.info(f"Column names for predictions are: \n {prediction_columns}")
        predictions_list, attributes = self._predict(
            dataloader,
            additional_attributes=additional_attributes,
            gpus=gpus,
            distribution_strategy=distribution_strategy,
            **trainer_kwargs,
        )
        predictions = np.concatenate(
            [pred.reshape(len(pred), -1) for pred in predictions_list], axis=1
        )
        assert len(prediction_columns) == predictions.shape[1], (
            f"Number of provided column names ({len(prediction_columns)}) and "
            f"number of output columns ({predictions.shape[1]}) don't match."
        )

        results = pd.DataFrame(predictions, columns=prediction_columns)
        for attr, values in attributes.items():
            if len(values) != len(results):
                self.warning_once(
                    f"Length of additional attribute '{attr}' ({len(values)})"
                    f" does not match the predictions ({len(results)}), e.g."
                    " because pulse-level attributes were requested for"
                    " event-level predictions. Attribute skipped."
                )
                continue
            results[attr] = values
        return results

    def _create_default_callbacks(
        self,
        val_dataloader: DataLoader,
        early_stopping_patience: Optional[int] = None,
    ) -> List:
        """Create default callbacks.

        Used in cases where no callbacks are specified by the user in
        .fit
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
