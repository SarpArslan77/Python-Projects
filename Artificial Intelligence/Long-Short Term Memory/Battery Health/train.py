# data_preprocess.py

#! Custom TODO notes:
#TODO AD: Add docstring.
#TODO ATH: Add type hint.
#TODO CTH: Check type hint.
#TODO FTH: Fix the hardcoding.
#TODO HPE: Handle possible error.
#TODO AC: Add comment.
#TODO AIC: Add input control.

#! PW: Possibly wrong.

from dataclasses import dataclass
from datetime import datetime
from typing import Any
import os

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
)
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.nn.utils import clip_grad_norm_
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader

from lstm import LSTM

@dataclass
class ConfigTrainer:
    dataloaders: tuple[DataLoader, DataLoader, DataLoader]
    learning_rate: float
    num_epochs: int
    max_norm: float
    print_freq: int
    val_freq: int
    save_checkpoint_freq: int
    model_save_path: str
    saved_checkpoint: dict[str, Any] | None = None
    show_graph: bool = True

    def __post_init__(self) -> None:
        # - learning_rate
        if not isinstance(self.learning_rate, float):
            raise TypeError(f"learning_rate must be an float, got {type(self.learning_rate).__name__}.")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}.")

        # - num_epochs
        if not isinstance(self.num_epochs, int):
            raise TypeError(f"num_epochs must be an integer, got {type(self.num_epochs).__name__}.")
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}.")

        # - max_norm
        if not isinstance(self.max_norm, float):
            raise TypeError(f"max_norm must be an float, got {type(self.max_norm).__name__}.")
        if self.max_norm <= 0:
            raise ValueError(f"max_norm must be positive, got {self.max_norm}.")

        # - print_freq
        if not isinstance(self.print_freq, int):
            raise TypeError(f"print_freq must be an integer, got {type(self.print_freq).__name__}.")
        if self.print_freq < 0:
            raise ValueError(f"print_freq must be non-negative, got {self.print_freq}.")

        # - val_freq
        if not isinstance(self.val_freq, int):
            raise TypeError(f"val_freq must be an integer, got {type(self.val_freq).__name__}.")
        if self.val_freq < 0:
            raise ValueError(f"val_freq must be non-negative, got {self.val_freq}.")

        # - save_checkpoint_freq
        if not isinstance(self.save_checkpoint_freq, int):
            raise TypeError(f"save_checkpoint_freq must be an integer, got {type(self.save_checkpoint_freq).__name__}.")
        if self.save_checkpoint_freq < 0:
            raise ValueError(f"save_checkpoint_freq must be non-negative, got {self.save_checkpoint_freq}.")

        # - model_save_path
        if not isinstance(self.model_save_path, str):
            raise TypeError(f"model_save_path must be a string, got {type(self.model_save_path).__name__}.")        
        if not os.path.exists(self.model_save_path):
            raise FileNotFoundError(f"model_save_path does not exist: {self.model_save_path}")
        if not os.path.isdir(self.model_save_path):
            raise NotADirectoryError(f"model_save_path exists but is not a directory: {self.model_save_path}")
        
        # - saved_checkpoint
        if self.saved_checkpoint:
            if not isinstance(self.saved_checkpoint, dict):
                raise TypeError(f"saved_checkpoint must be an dict[str, Any], got {type(self.saved_checkpoint).__name__}.")

        # - show_graph
        if not isinstance(self.show_graph, bool):
            raise TypeError(f"show_graph must be a boolean, got {type(self.show_graph).__name__}.")   


class Trainer():
    def __init__(
            self,
            config_trainer: ConfigTrainer,
            model: LSTM
    ) -> None:
        # Define the input parameters.
        self.cfg: ConfigTrainer = config_trainer
        self.train_loader, self.val_loader, self.test_loader = self.cfg.dataloaders

        # Define the setup objects.
        self.model = model

        self.loss_func = MSELoss()

        self.optimizer = Adam(
            params = self.model.parameters(),
            lr = self.cfg.learning_rate
        )

        self.start_epoch: int = 0

        # Create trackers for the training and validation:
        self.train_avg_loss_history: list[float] = []
        self.train_rmse_loss_history: list[float] = [] # Root Mean Squared Error (RMSE).
        self.train_r2_loss_history: list[float] = [] # Coefficient of Determination (R^2 Score).
        self.train_mae_loss_history: list[float] = [] # Mean Absolute Error (MAE).

        self.val_avg_loss_history: list[float] = []
        self.val_rmse_loss_history: list[float] = [] 
        self.val_r2_loss_history: list[float] = [] 
        self.val_mae_loss_history: list[float] = [] 

        # If so, load a model.
        if self.cfg.saved_checkpoint:
            # Unpack the parameters from the checkpoint.
            self.start_epoch = self.cfg.saved_checkpoint["start_epoch"]

            self.model.load_state_dict(self.cfg.saved_checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(self.cfg.saved_checkpoint["optimizer_state_dict"])

            self.train_rmse_loss_history, self.train_r2_loss_history, self.train_mae_loss_history = self.cfg.saved_checkpoint["train_histories"]
            self.loss_history, self.val_rmse_loss_history, self.val_r2_loss_history, self.val_mae_loss_history = self.cfg.saved_checkpoint["val_histories"]

    def _create_folder(
            self,
            train_start_time: datetime
    ) -> str | None:
        # 1. Extract the folder name and path.
        trial_folder_name: str = train_start_time.strftime(format="%Y-%m-%d_%H-%M")
        trial_folder_path: str = os.path.join(self.cfg.model_save_path, trial_folder_name)

        # 2. Try to create a folder.
        try: # If succesfull, return the path of the folder.
            os.makedirs(trial_folder_path)
            print(f"\nSuccesfully created the folder at {trial_folder_path}")

            return trial_folder_path
        except Exception as e: # If failed, return None.
            print(f"\nWarning: Could not create the folder at {trial_folder_path}")
            print(f"Error: {e}")

            return None

    def _calculate_losses(
            self,
            losses: Tensor,
            targets: Tensor,
            logits: Tensor
    ) -> tuple[float, float, float, float]:
        # Calculate the average loss.
        avg_loss: float = sum(losses) / len(losses)
        
        # Calculate the root mean squared error.
        rmse_loss: float = np.sqrt(avg_loss)

        # Calculate the coefficients of determination.
        r2_loss: float = r2_score(targets, logits)

        # Calculate the mean absolute error.
        mae_loss: float = mean_absolute_error(targets, logits)

        return (avg_loss, rmse_loss, r2_loss, mae_loss)

    def _train_one_epoch(self) -> tuple[float, float, float, float]:
        # 1. Open the training mode.
        self.model.train()

        # 2. Create trackers for the training predictions, targets and losses.
        train_logits: list[Tensor] = []
        train_targets: list[Tensor] = []
        train_losses: list[float] = []

        # 3. Iterate over the training set.
        for train_input, train_target in self.train_loader:
            # Forward pass.
            train_logit: Tensor = self.model(train_input)
            train_loss: Tensor = self.loss_func(train_logit, train_target)

            # Backward pass and optimize.
            self.optimizer.zero_grad() # Deletes the old gradients so that the current update is based only on the current batch of data.
            train_loss.backward()

            # Prevents gradients from getting too large (Gradient Clipping).
            clip_grad_norm_(
                parameters = self.model.parameters(),
                max_norm = self.cfg.max_norm
            )

            # Update the optimizer.
            self.optimizer.step()

            # Track the variables for this epoch.
            train_logits.append(train_logit.detach())
            train_targets.append(train_target)
            train_losses.append(train_loss.item())

        # 4. Calculate the loss values.
        train_targets_tensor: Tensor = torch.cat(train_targets)
        train_logits_tensor: Tensor = torch.cat(train_logits)

        train_avg_loss, train_rmse_loss, train_r2_loss, train_mae_loss = self._calculate_losses(
            losses = train_losses,
            targets = train_targets_tensor,
            logits = train_logits_tensor
        )

        return (train_avg_loss, train_rmse_loss, train_r2_loss, train_mae_loss)

    def _validate_one_epoch(self) -> tuple[float, float, float, float]:
        with torch.no_grad(): #TODO What does it do?
            # 1. Open the evaluation mode for the validation.
            self.model.eval()

            # Create trackers for the validation predictions, targets and losses.
            val_logits: list[Tensor] = []
            val_targets: list[Tensor] = []
            val_losses: list[float] = []

            # 2. Iterate over the validation set.
            for val_input, val_target in self.val_loader:
                # Forward pass.
                val_logit: Tensor = self.model(val_input)
                val_loss: Tensor = self.loss_func(val_logit, val_target)

                # Track the variables for this validation run.
                val_logits.append(val_logit.detach())
                val_targets.append(val_target)
                val_losses.append(val_loss.item())

            # 3. Calculate the loss values.
            val_targets_tensor: Tensor = torch.cat(val_targets)
            val_logits_tensor: Tensor = torch.cat(val_logits)

            val_avg_loss, val_rmse_loss, val_r2_loss, val_mae_loss = self._calculate_losses(
                losses = val_losses,
                targets = val_targets_tensor,
                logits = val_logits_tensor
            )

            return (val_avg_loss, val_rmse_loss, val_r2_loss, val_mae_loss)

    def _save_checkpoint(
            self,
            trial_folder_path: str,
            current_epoch: int
    ) -> tuple[dict[str, Any], str]:
        # 1. Create the checkpoint path.
        checkpoint_path: str = os.path.join(trial_folder_path, f"model_epoch_{current_epoch+1}.pth")

        # 2. Pack all the training and validation tracker lists into history tuples.
        train_history: tuple[list[float], list[float], list[float], list[float]] = self.train_avg_loss_history, self.train_rmse_loss_history, self.train_r2_loss_history, self.train_mae_loss_history
        val_history: tuple[list[float], list[float], list[float], list[float]] = self.val_avg_loss_history, self.val_rmse_loss_history, self.val_r2_loss_history, self.val_mae_loss_history

        # 3. Create the dictionary to hold all the states.
        checkpoint: dict[str, Any] = {
            "start_epoch": current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_history": train_history,
            "val_history": val_history
        }

        return (checkpoint, checkpoint_path)


    def fit(self) -> tuple[tuple[list[float], list[float], list[float], list[float]], tuple[list[float], list[float], list[float], list[float]], tuple[float, float, float]]:
        # 1. Start the training loop.
        print("\nStarting with Training!")

        # Start tracking the training time.
        train_start_time = datetime.now()

        # 2. Create a timestamped folder.
        trial_folder_path: str | None = self._create_folder(train_start_time=train_start_time)

        # 3. Start with the loop.
        for epoch in range(self.start_epoch, self.cfg.num_epochs):
            # Check for the frequencies.
            print_this_epoch: bool = ((epoch + 1) % self.cfg.print_freq == 0)
            validate_this_epoch: bool = ((epoch + 1) % self.cfg.val_freq == 0)
            save_checkpoint_this_epoch: bool = ((epoch + 1) % self.cfg.save_checkpoint_freq == 0)

            # Train the model for one epoch.
            train_avg_loss, train_rmse_loss, train_r2_loss, train_mae_loss = self._train_one_epoch()

            # If so, print the progress.
            if print_this_epoch:
                print(f"\nEpoch: {epoch+1}/{self.cfg.num_epochs}")
                print(f" Training Losses:")
                print(f" -> Average: {train_avg_loss:.4f}")
                print(f" -> RMSE: {train_rmse_loss:.4f}")
                print(f" -> R^2: {train_r2_loss:.4f}")
                print(f" -> MAE: {train_mae_loss:.4f}")

            # Append the losses to the histories.
            self.train_avg_loss_history.append(train_avg_loss)
            self.train_rmse_loss_history.append(train_rmse_loss)
            self.train_r2_loss_history.append(train_r2_loss)
            self.train_mae_loss_history.append(train_mae_loss)

            # If so, validate.
            if validate_this_epoch:
                val_avg_loss, val_rmse_loss, val_r2_loss, val_mae_loss = self._validate_one_epoch()

                print(f"\nEpoch: {epoch+1}/{self.cfg.num_epochs}")
                print(f" Validation Losses:")
                print(f" -> Average: {val_avg_loss:.4f}")
                print(f" -> RMSE: {val_rmse_loss:.4f}")
                print(f" -> R^2: {val_r2_loss:.4f}")
                print(f" -> MAE: {val_mae_loss:.4f}")

                # Append the losses to the histories.
                self.val_avg_loss_history.append(val_avg_loss)
                self.val_rmse_loss_history.append(val_rmse_loss)
                self.val_r2_loss_history.append(val_r2_loss)
                self.val_mae_loss_history.append(val_mae_loss)

            # If so, save the checkpoint.
            if save_checkpoint_this_epoch:
                if not(trial_folder_path):
                    print("\n!!! The folder creation had failed, the model can not get saved during this fitting run !!!")
                else:
                    checkpoint, checkpoint_path = self._save_checkpoint(
                        trial_folder_path = trial_folder_path,
                        current_epoch = epoch
                    )

                    torch.save(
                        obj = checkpoint,
                        f = checkpoint_path
                    )

        # 4. Calculate the training time.
        train_end_time = datetime.now()
        train_elapsed_secs: float = (train_end_time - train_start_time).total_seconds()
        train_elapsed_mins, train_sec = divmod(train_elapsed_secs, 60)
        train_hour, train_min = divmod(train_elapsed_mins, 60)

        # Pack all the times in a tuple for the title of the graph.
        train_time: tuple[float, float, float] = (train_hour, train_min, train_sec)
        print(f"\nFinished Training!")
        print(f" Training Time: {int(train_hour):02d}:{int(train_min):02d}:{int(train_sec):02d}") # TODO What are those 02 and d mean?

        #TODO Continue here...
            

