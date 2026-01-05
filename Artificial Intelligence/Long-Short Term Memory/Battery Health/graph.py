
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

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import mplcursors
import numpy as np
from numpy.typing import NDArray

# Define the custom type hints.
History = tuple[NDArray, NDArray]

class Plotter():
    """
    Visualization engine for comparing training and validation metrics.

    Generates a 2x2 grid of subplots for MSE, RMSE, R2, and MAE.
    Handles the alignment of validation data (which often has fewer data points)
    to match the training epochs x-axis.
    """

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

    def _generate_x_axis_range(
            self,
            data: NDArray
    ) -> NDArray:
        """
        Creates a sequential integer array to represent the X-axis.

        Args:
            data (NDArray): The data array whose length determines the range.

        Returns:
            NDArray: An array [1, 2, ..., N] where N is the length of data.
        """
        # Get the length of data.
        num_epoch: int = len(data)

        # Create the range.
        epoch_range: NDArray = np.arange(
            start = 0,
            stop = num_epoch
        )

        return epoch_range

    def _plot_history_subplot(
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
        epoch_range: NDArray = self._generate_x_axis_range(data=train_data)

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

    def _save_graph(
            self,
            trial_folder_path: str,
            image_name: str,
    ) -> None:
        """
        Saves the generated figure to the specified directory with a specified name.

        Args:
            trial_folder_path (str): The directory path where the image should be saved.
            image_name  (str): The name, under which the image will be saved as .png file.
        """

        try: # Try to create a graph for the losses.
            graph_path: str = os.path.join(trial_folder_path, f"{image_name}.png")
            plt.savefig(graph_path)
            print(f"\nThe Graph {graph_path} has been created succesfully!")
        except Exception as e:
            print(f"\n!!! The saving of ratio graph has failed !!!")
            print(f"Error: {e}")

    def plot_training_results(
            self,
            train_time: tuple[float, float, float],
            histories: tuple[History, History, History, History],
            colors: tuple[str, str],
            linestyles: tuple[str, str],
            trial_folder_path: str,
            show_graph: bool = True
    ) -> None:
        """
        Main interface to generate, display, and save the training history plots.

        Args:
            train_time (tuple): Duration (H, M, S) to display in the figure title.
            histories (tuple): A tuple containing the 4 history pairs (Train/Val).
            colors (tuple): Colors for training and validation lines.
            linestyles (tuple): Line styles for training and validation lines.
            trial_folder_path (str): The folder path to save the output image.
            show_graph (bool): Whether to open the interactive Matplotlib window.
        """

        # 1. Unpack the training time.
        train_hour, train_min, train_sec = train_time

        # 2. Define the figure and gridspace.
        fig: Figure = plt.figure(figsize=(10, 10))
        fig.suptitle(f"Training Time: {int(train_hour):02d}:{int(train_min):02d}:{int(train_sec):02d}")

        gs = GridSpec(
            nrows = 2,
            ncols = 2,
            figure = fig
        )
        # Adjust the spacing between the subplots.
        gs.update(wspace=0.3)

        # 3. Define the axes.
        mse_ax: Axes = fig.add_subplot(gs[0, 0])
        rmse_loss_ax: Axes = fig.add_subplot(gs[0, 1])
        r2_loss_ax: Axes = fig.add_subplot(gs[1, 0])
        mae_loss_ax: Axes = fig.add_subplot(gs[1, 1])

        # 4. Unpack the histories.
        mse_history, rmse_loss_history, r2_loss_history, mae_loss_history = histories

        # 5. Plot the metrics.
        self._plot_history_subplot(
            ax = mse_ax,
            data = mse_history,
            colors = colors,
            linestyles = linestyles,
            label = "Average Loss"
        )

        self._plot_history_subplot(
            ax = rmse_loss_ax,
            data = rmse_loss_history,
            colors = colors,
            linestyles = linestyles,
            label = "Root Mean Squared Error Loss"
        )

        self._plot_history_subplot(
            ax = r2_loss_ax,
            data = r2_loss_history,
            colors = colors,
            linestyles = linestyles,
            label = "Coefficient of Determination Loss"
        )

        self._plot_history_subplot(
            ax = mae_loss_ax,
            data = mae_loss_history,
            colors = colors,
            linestyles = linestyles,
            label = "Mean Absolute Error Loss"
        )

        # 6. Save the graphs.
        self._save_graph(
            trial_folder_path = trial_folder_path,
            image_name = "loss_histories"
        )

        # 7. If so, show the graph.
        if show_graph:
            plt.show()

    def plot_test_results(
            self,
            scaling_factors: tuple[NDArray, NDArray],
            data: tuple[NDArray, NDArray],
            colors: tuple[str, str],
            linestyles: tuple[str, str],
            trial_folder_path: str,
            show_graph: bool = True
    ) -> None:
        """
        Plots the model predictions against the ground truth targets for the test set.

        Denormalizes the data using the provided scaling factors (min/max) to show
        real-world units (Capacity in Ah).

        Args:
            scaling_factors (tuple): (min_val, max_val) used during training normalization.
            data (tuple): (predictions, targets) normalized arrays.
            colors (tuple): Colors for prediction and target lines.
            linestyles (tuple): Line styles for prediction and target lines.
            trial_folder_path (str): Directory to save the output graph.
            show_graph (bool): Whether to display the plot window.
        """

        # 1. Unpack the parameters.
        min_val, max_val = scaling_factors
        pred_data, target_data = data
        pred_color, target_color = colors
        pred_linestyle, target_linestyle = linestyles

        # 2. Define a figure, gridspace and axis.
        fig: Figure = plt.figure(figsize=(5, 5))
        gs = GridSpec(
            nrows = 1,
            ncols = 1,
            figure = fig
        )
        ax: Axes = fig.add_subplot(gs[0, 0])

        # 3. Create the x-axis vector.
        sample_index: NDArray = self._generate_x_axis_range(data=pred_data)

        # 4. Scale the test data back to it's original form.
        scaled_pred_data: NDArray = pred_data * (max_val - min_val) + min_val 
        scaled_target_data: NDArray = target_data * (max_val - min_val) + min_val

        # 5. Plot predictions against the targets.
        pred_line, = ax.plot(
            sample_index,
            scaled_pred_data,
            color = pred_color,
            label = "Predictions",
            linestyle = pred_linestyle
        )

        target_line, = ax.plot(
            sample_index,
            scaled_target_data,
            color = target_color,
            label = "Targets",
            linestyle = target_linestyle
        )

        # 6. Define the settings for the axis.
        ax.set_title("Prediction against Target Capacities")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Capacity (Ah)")
        ax.grid(visible=True)
        ax.legend(handles=[pred_line, target_line])

        # 7. Add a cursor to the graph.
        mplcursors.cursor([pred_line, target_line])

        # 8. Save the graph.
        self._save_graph(
            trial_folder_path = trial_folder_path,
            image_name = "pred_vs_target_capacities"
        )

        # 9. If so, save the graph.
        if show_graph:
            plt.show()
