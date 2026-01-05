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
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from lstm import LSTM


@dataclass
class ConfigTrainer:
    """
    Configuration dataclass for the training process.

    Attributes:
        dataloaders (tuple): A tuple containing (train_loader, val_loader, test_loader).
        learning_rate (float): The learning rate for the Adam optimizer.
        mode (str): The metric for the scheduler.
        factor (float): Decreasing factor for learning rate for the scheduler.
        patience (int): Number of epochs without improvement, after which the learning rate will be reduced.
        min_lr (float): Minimum learning rate, that the optimizer can drop to.
        num_epochs (int): Total number of training epochs.
        max_norm (float): Maximum norm for gradient clipping to prevent exploding gradients.
        print_freq (int): Frequency (in epochs) to print training statistics to console.
        val_freq (int): Frequency (in epochs) to run validation.
        save_checkpoint_freq (int): Frequency (in epochs) to save model checkpoints.
        model_save_path (str): Directory path where model checkpoints will be stored.
        saved_checkpoint (dict | None): Optional dictionary containing a loaded checkpoint state to resume training.
        show_graph (bool): Whether to display loss graphs after training (default: True).
    """

    dataloaders: tuple[DataLoader, DataLoader, DataLoader]
    learning_rate: float
    mode: str
    factor: float
    patience: int
    min_lr: float
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
        
        # - mode
        if self.mode not in ['min', 'max']:
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode}")
        
        # - factor
        if not isinstance(self.factor, float):
            raise TypeError(f"factor must be a float.")
        if not (0 < self.factor < 1):
            raise ValueError(f"factor must be between 0 and 1.")
        
        # - patience
        if not isinstance(self.patience, int):
            raise TypeError(f"patience must be an integer.")
        if self.patience < 0:
            raise ValueError(f"patience must be non-negative.")

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

# Define custom type hints.
History = tuple[NDArray, NDArray]
Metrics = tuple[float, float, float, float]

class Trainer():
    """
    Manages the full training lifecycle of the LSTM model.

    This class handles:
    1. Training and Validation loops.
    2. Metric calculation (MSE, RMSE, R2, MAE).
    3. Checkpoint saving and loading.
    4. Gradient optimization and clipping.
    5. History tracking for visualization.
    """

    def __init__(
            self,
            config_trainer: ConfigTrainer,
            model: LSTM
    ) -> None:
        """
        Initializes the Trainer.

        Sets up the optimizer (Adam), Loss Function (MSE), Schuler (ReduceLROnPlateau), 
        and history trackers. If a checkpoint is provided in the config, it restores the model state,
        optimizer state, and loss histories to resume training.

        Args:
            config_trainer (ConfigTrainer): Configuration object with hyperparameters and paths.
            model (LSTM): The PyTorch model instance to train.
        """

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

        self.scheduler = ReduceLROnPlateau(
            optimizer = self.optimizer,
            mode = self.cfg.mode,
            factor = self.cfg.factor,
            patience = self.cfg.patience,
            min_lr = self.cfg.min_lr
        )

        self.start_epoch: int = 0

        # Create trackers for the training and validation:
        self.train_mse_history: list[float] = [] # Mean Squared Error (MSE).
        self.train_rmse_history: list[float] = [] # Root Mean Squared Error (RMSE).
        self.train_r2_history: list[float] = [] # Coefficient of Determination (R^2 Score).
        self.train_mae_history: list[float] = [] # Mean Absolute Error (MAE).

        self.val_mse_history: list[float] = []
        self.val_rmse_history: list[float] = [] 
        self.val_r2_history: list[float] = [] 
        self.val_mae_history: list[float] = [] 

        # If so, load a model.
        if self.cfg.saved_checkpoint:
            # Unpack the parameters from the checkpoint.
            self.start_epoch = self.cfg.saved_checkpoint["start_epoch"]

            self.model.load_state_dict(self.cfg.saved_checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(self.cfg.saved_checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(self.cfg.saved_checkpoint["scheduler_state_dict"])

            self.train_rmse_history, self.train_r2_history, self.train_mae_history = self.cfg.saved_checkpoint["train_histories"]
            self.loss_history, self.val_rmse_history, self.val_r2_history, self.val_mae_history = self.cfg.saved_checkpoint["val_histories"]

    def _create_folder(
            self,
            train_start_time: datetime
    ) -> str | None:
        """
        Creates a timestamped subdirectory for saving model checkpoints.

        Format: 'YYYY-MM-DD_HH-MM'.

        Args:
            train_start_time (datetime): The timestamp of when training started.

        Returns:
            str | None: The path to the created folder if successful, else None.
        """

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

    def _calculate_metrics(
            self,
            targets: Tensor,
            logits: Tensor
    ) -> Metrics:
        """
        Calculates regression performance metrics.

        Computes:
        1. MSE (Mean Squared Error).
        2. RMSE (Root Mean Squared Error).
        3. R^2 Score (Coefficient of Determination).
        4. MAE (Mean Absolute Error).

        Args:
            targets (Tensor): Ground truth values (concatenated from all batches).
            logits (Tensor): Model predictions (concatenated from all batches).

        Returns:
            Metric: (Avg Loss, RMSE, R2, MAE).
        """

        # Calculate the mean squared error.
        mse: float = mse_loss(
            input = logits,
            target = targets
        )
        
        # Calculate the root mean squared error.
        rmse: float = np.sqrt(mse)

        # Calculate the coefficients of determination.
        r2: float = r2_score(targets, logits)

        # Calculate the mean absolute error.
        mae: float = mean_absolute_error(targets, logits)

        return (mse, rmse, r2, mae)

    def _train_one_epoch(self) -> Metrics:
        """
        Runs one complete training epoch.

        Performs forward pass, loss calculation, backpropagation, gradient clipping,
        and optimizer updates. Tracks metrics for the entire epoch.

        Returns:
            Metric: The aggregated training metrics 
            (MSE, RMSE, R2, MAE) for this epoch.
        """

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

        train_metrics = self._calculate_metrics(
            targets = train_targets_tensor,
            logits = train_logits_tensor
        )

        return train_metrics

    def _validate_one_epoch(self) -> Metrics:
        """
        Runs one complete validation epoch.

        Disables gradient calculation (torch.no_grad) and model dropout (model.eval).
        Used to monitor model performance on unseen data without updating weights.

        Returns:
            Metric: The aggregated validation metrics 
            (MSE, RMSE, R2, MAE) for this epoch.
        """

        with torch.no_grad(): # Stops the tracking the history of operations.
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

            # 3. Calculate the metrics.
            val_targets_tensor: Tensor = torch.cat(val_targets)
            val_logits_tensor: Tensor = torch.cat(val_logits)

            val_metrics = self._calculate_metrics(
                targets = val_targets_tensor,
                logits = val_logits_tensor
            )

            return val_metrics

    def _save_checkpoint(
            self,
            trial_folder_path: str,
            current_epoch: int
    ) -> tuple[dict[str, Any], str]:
        """
        Bundles the current model state and training history into a checkpoint dictionary.

        Args:
            trial_folder_path (str): The directory where the checkpoint file will be saved.
            current_epoch (int): The current epoch index (0-based).

        Returns:
            tuple[dict[str, Any], str]: A tuple containing:
                - checkpoint (dict): The dictionary containing model/optimizer states and history.
                - checkpoint_path (str): The full file path where the .pth file will be written.
        """

        # 1. Create the checkpoint path.
        checkpoint_path: str = os.path.join(trial_folder_path, f"model_epoch_{current_epoch+1}.pth")

        # 2. Pack all the training and validation tracker lists into history tuples.
        train_history: tuple[list[float], list[float], list[float], list[float]] = self.train_mse_history, self.train_rmse_history, self.train_r2_history, self.train_mae_history
        val_history: tuple[list[float], list[float], list[float], list[float]] = self.val_mse_history, self.val_rmse_history, self.val_r2_history, self.val_mae_history

        # 3. Create the dictionary to hold all the states.
        checkpoint: dict[str, Any] = {
            "start_epoch": current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_history": train_history,
            "val_history": val_history
        }

        return (checkpoint, checkpoint_path)


    def fit(self) -> tuple[History, History, History, History, tuple[float, float, float], str]:
        """
        Executes the main training loop.

        Orchestrates the process of:
        1. Training for `num_epochs`.
        2. Logging metrics to console.
        3. Running validation at specified frequencies.
        4. Saving checkpoints to disk.

        Returns:
            tuple: A tuple containing:
                - mse_history (tuple): (train_mse, val_mse) arrays.
                - rmse_history (tuple): (train_rmse, val_rmse) arrays.
                - r2_history (tuple): (train_r2, val_r2) arrays.
                - mae_history (tuple): (train_mae, val_mae) arrays.
                - train_time (tuple): Elapsed training time (Hours, Minutes, Seconds).
                - trial_folder_path (str): The path where models/graphs are saved.
        """

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
            train_mse, train_rmse, train_r2, train_mae = self._train_one_epoch()

            # If so, print the progress.
            if print_this_epoch:
                print(f"\nEpoch: {epoch+1}/{self.cfg.num_epochs}")
                print(f" Training Metrics:")
                print(f" -> MSE: {train_mse:.4f}")
                print(f" -> RMSE: {train_rmse:.4f}")
                print(f" -> R^2: {train_r2:.4f}")
                print(f" -> MAE: {train_mae:.4f}")

            # Append the losses to the histories.
            self.train_mse_history.append(train_mse)
            self.train_rmse_history.append(train_rmse)
            self.train_r2_history.append(train_r2)
            self.train_mae_history.append(train_mae)

            # If so, validate.
            if validate_this_epoch:
                val_mse, val_rmse, val_r2, val_mae = self._validate_one_epoch()

                # Update the scheduler.
                self.scheduler.step(metrics=val_mse)

                print(f"\n Validation Metrics:")
                print(f" -> MSE: {val_mse:.4f}")
                print(f" -> RMSE: {val_rmse:.4f}")
                print(f" -> R^2: {val_r2:.4f}")
                print(f" -> MAE: {val_mae:.4f}")

                # Append the losses to the histories.
                self.val_mse_history.append(val_mse)
                self.val_rmse_history.append(val_rmse)
                self.val_r2_history.append(val_r2)
                self.val_mae_history.append(val_mae)

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
        print(f" Training Time: {int(train_hour):02d}:{int(train_min):02d}:{int(train_sec):02d}") # 0: Pad with zeros. & 2: Min width of charachters. & d: Input is a int. => (5 -> 05) and 12 stays.

        # 5. Convert all the lists into tuples.
        train_mse_history_np: NDArray = np.array(self.train_mse_history)
        train_rmse_history_np: NDArray = np.array(self.train_rmse_history)
        train_r2_history_np: NDArray = np.array(self.train_r2_history)
        train_mae_history_np: NDArray = np.array(self.train_mae_history)

        val_mse_history_np: NDArray = np.array(self.val_mse_history)
        val_rmse_history_np: NDArray = np.array(self.val_rmse_history)
        val_r2_history_np: NDArray = np.array(self.val_r2_history)
        val_mae_history_np: NDArray = np.array(self.val_mae_history)

        # Pack all the loss histories into tuples.
        mse_history: tuple[NDArray, NDArray] = (train_mse_history_np, val_mse_history_np)
        rmse_history: tuple[NDArray, NDArray] = (train_rmse_history_np, val_rmse_history_np)
        r2_history: tuple[NDArray, NDArray] = (train_r2_history_np, val_r2_history_np)
        mae_history: tuple[NDArray, NDArray] = (train_mae_history_np, val_mae_history_np)

        return (mse_history, rmse_history, r2_history, mae_history, train_time, trial_folder_path)

    def test(self) -> tuple[NDArray, NDArray]:
        """
        Evaluates the model on the Test Set.

        Performs a forward pass on the test data, calculates global metrics,
        and returns the raw predictions and targets for visualization.

        Returns:
            tuple[NDArray, NDArray]: A tuple containing:
                - predictions: NumPy array of model predictions.
                - targets: NumPy array of ground truth values.
        """

        print("\n Starting with testing!")

        with torch.no_grad():
            # 1. Open the evaluation mode for the testing.
            self.model.eval()

            # 2. Create trackers for the test predictions and targets.
            test_logits: list[Tensor] = []
            test_targets: list[Tensor] = []

            # 3. Iterate over the test dataset.
            for test_input, test_target in self.test_loader:
                # Forward pass.
                test_logit: Tensor = self.model(test_input)

                # Track the variables.
                test_logits.append(test_logit.detach())
                test_targets.append(test_target)

            # 4. Calculate the metrics.
            test_targets_tensor: Tensor = torch.cat(test_targets)
            test_logits_tensor: Tensor = torch.cat(test_logits)

            (test_mse, test_rmse, test_r2, test_mae) = self._calculate_metrics(
                targets = test_targets_tensor,
                logits = test_logits_tensor
            )

            print(f"\n Test Metrics:")
            print(f" -> MSE: {test_mse:.4f}")
            print(f" -> RMSE: {test_rmse:.4f}")
            print(f" -> R^2: {test_r2:.4f}")
            print(f" -> MAE: {test_mae:.4f}")

            # 5. Return the data and predictions as numpy arrays, to plot them.
            concat_test_logits: NDArray = np.concatenate(test_logits)
            concat_test_targets: NDArray = np.concatenate(test_targets)

            return (concat_test_logits, concat_test_targets)
