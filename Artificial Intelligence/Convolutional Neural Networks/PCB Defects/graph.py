
#! Custom TODO notes:
#TODO AD: Add docstring.
#TODO ATH: Add type hint.
#TODO CTH: Check type hint.
#TODO FTH: Fix the hardcoding.
#TODO HPE: Handle possible error.
#TODO AC: Add comment.

import matplotlib.axes
import matplotlib.pyplot as plt
import mplcursors
import numpy as np
from numpy.typing import NDArray

def plot_graph(
        num_epochs: int,
        training_time: tuple[float, float, float],
        train_loss_history: NDArray,
        val_loss_history: NDArray,
        lr_history: NDArray
) -> None:
    # Create the figure and axes object for the graphs.
    fig, (loss_ax, lr_ax) = plt.subplots(
        nrows = 1,
        ncols = 2,
        figsize = (10, 5)
    )
    loss_ax: matplotlib.axes.Axes
    lr_ax: matplotlib.axes.Axes
    # Unpackt the training time for the title of whole figure.
    training_hours, training_minutes, training_seconds = training_time
    fig.suptitle(f"Training Time: {int(training_hours):02d}:{int(training_minutes):02d}:{int(training_seconds):02d}")

    # Plot the training and validation loss on the same graph.
    train_loss_line, = loss_ax.plot(
        np.arange(1, num_epochs+1),
        train_loss_history,
        color = "red",
        label = "Training"
    )
    val_loss_line, = loss_ax.plot(
        np.arange(1, num_epochs+1),
        val_loss_history,
        color = "blue",
        label = "Validation"
    )
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Training and Validation Loss Progress")
    loss_ax.legend(loc="lower right")
    loss_ax.grid(True)
    # Add a cursor to the graph.
    mplcursors.cursor([train_loss_line, val_loss_line])

    # Plot the learning rate in a seperate graph.
    lr_line, = lr_ax.plot(
        np.arange(1, num_epochs+1),
        lr_history,
        color = "orange"
    )
    lr_ax.set_xlabel("Epoch")
    lr_ax.set_ylabel("Learning Rate Progress")
    lr_ax.grid(True)
    mplcursors.cursor(lr_line)