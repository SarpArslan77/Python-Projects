
# graph.py

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
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import mplcursors
import numpy as np
from numpy.typing import NDArray

@dataclass
class ConfigHistoryPlotter:
    """
    Configuration dataclass for the HistoryPlotter visualization tool.

    Attributes:
        train_time (tuple[float, float, float]): The duration of the training session
            formatted as (Hours, Minutes, Seconds).
    """
    
    train_time: tuple[float, float, float]

    def __post_init__(self) -> None:
        # Unpack the times.
        train_hour, train_min, train_sec = self.train_time

        # - train_hour
        if not isinstance(train_hour, float):
            raise TypeError(f"train_hour must be an float, got {type(train_hour).__name__}.")
        if train_hour < 0:
            raise ValueError(f"train_hour must be non-negative, got {train_hour}.")
        # - train_min
        if not isinstance(train_min, float):
            raise TypeError(f"train_min must be an float, got {type(train_min).__name__}.")
        if train_min < 0:
            raise ValueError(f"train_min must be non-negative, got {train_min}.")
        # - train_sec
        if not isinstance(train_sec, float):
            raise TypeError(f"train_sec must be an float, got {type(train_sec).__name__}.")
        if train_sec < 0:
            raise ValueError(f"train_sec must be non-negative, got {train_sec}.")

# Define the custom type hints.
History = tuple[NDArray, NDArray]

class HistoryPlotter():
    """
    Visualization engine for comparing training and validation metrics.

    Generates a 2x2 grid of subplots for MSE, RMSE, R2, and MAE.
    Handles the alignment of validation data (which often has fewer data points)
    to match the training epochs x-axis.
    """
    
    def __init__(
            self,
            config_history_plotter: ConfigHistoryPlotter
    ) -> None:
        # Define the input parameters.
        self.cfg: ConfigHistoryPlotter = config_history_plotter

        # Unpack the training time for the title.
        train_hour, train_min, train_sec = self.cfg.train_time
        
        # Define the figure.
        self.fig: Figure = plt.figure(figsize=(10, 10))
        self.fig.suptitle(f"Training Time: {int(train_hour):02d}:{int(train_min):02d}:{int(train_sec):02d}")

        # Define the gridspace.
        self.gs = gridspec.GridSpec(
            nrows = 2,
            ncols = 2,
            figure = self.fig
        )
        # Adjust the spacing between the subplots.
        self.gs.update(wspace=0.3)

    def _upsample_data(
            self,
            train_data: NDArray,
            val_data: NDArray
    ) -> NDArray:
        """
        Upsamples the validation data to match the length of the training data.

        Since validation is performed less frequently (e.g., every 5 epochs),
        this method repeats the validation values to fill the gaps, creating a
        "step" pattern that aligns perfectly with the training epoch axis.

        Args:
            train_data (NDArray): The array of training metric values (reference length).
            val_data (NDArray): The array of validation metric values (to be expanded).

        Returns:
            NDArray: The expanded validation array with length equal to `len(train_data)`.
        """

        # 1. Calculate the number of data.
        n_train_data: int = len(train_data)
        n_val_data: int = len(val_data)

        # If so, return zeros.
        if n_val_data == 0:
            return np.zeros(n_train_data)

        # 2. Calculate the repeating frequency and remainder.
        freq: int = int(n_train_data // n_val_data )
        remain: int = n_train_data % n_val_data

        # 3. Initialize every element of the array with the frequency.
        freq_val_data: NDArray = np.full(
            shape = n_val_data,
            fill_value = freq,
            dtype = np.int64
        )

        # 4. Add the remainder to the last element.
        freq_val_data[n_val_data-remain:] += 1

        # 5. Perform the repetition.
        expanded_val_data: NDArray = np.repeat(
            a = val_data,
            repeats = freq_val_data
        )

        return expanded_val_data

    def _plot_subplot(
            self,
            ax: Axes,
            data: tuple[NDArray, NDArray],
            colors: tuple[str, str],
            linestyles: tuple[str, str],
            label: str
    ) -> None:
        """
        Plots a single metric comparison (Train vs Val) on a specific Matplotlib axis.

        Args:
            ax (Axes): The subplot axis object to draw on.
            data (tuple): A pair of arrays (train_metric, val_metric).
            colors (tuple): A pair of color codes (train_color, val_color).
            linestyles (tuple): A pair of line styles (train_style, val_style).
            label (str): The title/label for this specific metric (e.g., "RMSE").
        """

        # 1. Split the parameters into training and validation.
        train_data, val_data = data
        train_color, val_color = colors
        train_linestyle, val_linestyle = linestyles

        # 2. Extend the validation data, so it's size matches the training data.
        expanded_val_data: NDArray = self._upsample_data(
            train_data = train_data,
            val_data = val_data
        )

        # 3. Create the x-axis vector.
        num_epoch: int = len(train_data)

        epoch_range: NDArray = np.arange(
            start = 1,
            stop = num_epoch + 1
        )

        # 4. Plot the training and validation data.
        train_line, = ax.plot(
            epoch_range,
            train_data,
            color = train_color,
            label = f"Training {label}",
            linestyle = train_linestyle
        )

        val_line, = ax.plot(
            epoch_range,
            expanded_val_data,
            color = val_color,
            label = f"Validation {label}",
            linestyle = val_linestyle
        )

        # 5. Define the settings for the axis.
        ax.set_title(f"{label} History")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.grid(visible=True)
        ax.legend(handles=[train_line, val_line])

        # 6. Add a cursor to the graph.
        mplcursors.cursor([train_line, val_line])

    def _save_plots(
            self,
            trial_folder_path: str
    ) -> None:
        """
        Saves the generated figure to the specified directory.

        The file is saved as 'loss_histories.png'.

        Args:
            trial_folder_path (str): The directory path where the image should be saved.
        """

        try: # Try to create a graph for the losses.
            graph_path: str = os.path.join(trial_folder_path, "loss_histories.png")
            plt.savefig(graph_path)
            print(f"\nThe Graph {graph_path} has been created succesfully!")
        except Exception as e:
            print(f"\n!!! The saving of ratio graph has failed !!!")
            print(f"Error: {e}")

    def plot_history(
            self,
            histories: tuple[History, History, History, History],
            plot_colors: tuple[str, str],
            plot_linestyles: tuple[str, str],
            trial_folder_path: str,
            show_graph: bool = True
    ) -> None:
        """
        Main interface to generate, display, and save the training history plots.

        Orchestrates the creation of 4 subplots (Avg Loss, RMSE, R2, MAE),
        populates them with data, saves the figure, and optionally shows the window.

        Args:
            histories (tuple): A tuple containing the 4 history pairs (Train/Val) for
                               Avg Loss, RMSE, R2, and MAE.
            plot_colors (tuple): Colors for training and validation lines.
            plot_linestyles (tuple): Line styles for training and validation lines.
            trial_folder_path (str): The folder path to save the output image.
            show_graph (bool): Whether to open the interactive Matplotlib window.
        """

        # 1. Unpack the histories.
        avg_loss_history, rmse_loss_history, r2_loss_history, mae_loss_history = histories

        # 2. Define the axes.
        avg_loss_ax: Axes = self.fig.add_subplot(self.gs[0, 0])
        rmse_loss_ax: Axes = self.fig.add_subplot(self.gs[0, 1])
        r2_loss_ax: Axes = self.fig.add_subplot(self.gs[1, 0])
        mae_loss_ax: Axes = self.fig.add_subplot(self.gs[1, 1])

        # 3. Plot the losses.
        self._plot_subplot(
            ax = avg_loss_ax,
            data = avg_loss_history,
            colors = plot_colors,
            linestyles = plot_linestyles,
            label = "Average Loss"
        )

        self._plot_subplot(
            ax = rmse_loss_ax,
            data = rmse_loss_history,
            colors = plot_colors,
            linestyles = plot_linestyles,
            label = "Root Mean Squared Error Loss"
        )

        self._plot_subplot(
            ax = r2_loss_ax,
            data = r2_loss_history,
            colors = plot_colors,
            linestyles = plot_linestyles,
            label = "Coefficient of Determination Loss"
        )

        self._plot_subplot(
            ax = mae_loss_ax,
            data = mae_loss_history,
            colors = plot_colors,
            linestyles = plot_linestyles,
            label = "Mean Absolute Error Loss"
        )

        # 4. Save the graphs.
        self._save_plots(trial_folder_path=trial_folder_path)

        # 5. If so, show the graph.
        if show_graph:
            plt.show()