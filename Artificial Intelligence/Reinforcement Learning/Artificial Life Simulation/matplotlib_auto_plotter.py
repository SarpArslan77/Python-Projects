from math import (
    ceil,
    sqrt
)
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import mplcursors
import numpy as np
from numpy.typing import NDArray

class MatplotlibAutoPlotter():
    def __init__(self) -> None:
        pass

    def generate_plot( #TODO Add a parameter to specify the max number of plots in a single figure.
            self,
            history: dict[str, list],
            screen_ratio: tuple[int, int] = (16, 9) 
    ) -> None:
        if not history:
            return
        
        # 1. Calculates the figure size.
        width, height = screen_ratio

        ratio: float = width / height
        ntracker: int = len(history.keys()) #TODO Add a setting to make this also user specifable.

        nfig_cols: int = ceil(sqrt(ntracker * ratio))
        nfig_rows: int = ceil(ntracker / nfig_cols)

        # Validation: Ensures that there is not more rows than needed.
        if (nfig_rows - 1) * nfig_cols >= ntracker:
            nfig_rows -= 1

        # 2. Defines the figure and axes (flattened).
        fig, axes_arr = plt.subplots(
            nrows = nfig_rows,
            ncols = nfig_cols,
            squeeze = False, # Forces matplotlib to always return a 2D array regardless of number of plots.
            #   Else, 'flatten' breaks, if the array is 1D and not 2D.    
            layout = "constrained" # Automatically manages the space between the subplots.
        )
        axes_arr: NDArray

        flat_axes: NDArray = axes_arr.flatten()

        # 3. Creates the x-axis vector for all of the subplots.
        ndata: int = len(next(iter(history.values())))

        nframes_vector: NDArray = np.arange(
            start = 1,
            stop = ndata + 1
        )

        # 4. Plots the data.
        for i, (col, val) in enumerate(history.items()):
            ax: Axes = flat_axes[i]

            ax.plot(
                nframes_vector,
                val
            )

            ax.set_xlabel("Frames")
            ax.set_ylabel(f"Mean {col}")
            ax.grid(True)

        # Hides the unused axes.
        nused_axes: int = len(flat_axes)

        for j in range(ntracker, nused_axes):
            unused_ax: Axes = flat_axes[j]

            unused_ax.set_visible(b=False) # Sets the axis (here unused) unvisible.

        plt.show()