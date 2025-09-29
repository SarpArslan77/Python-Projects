
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.patches
from matplotlib.animation import FuncAnimation
import mplcursors

def plot_intervall(
        start: int,
        end: int,
        sums: np.ndarray
) -> None:
    fig, ax = plt.subplots()
    ax: matplotlib.axes.Axes
    ax.set_xlabel("Number")
    ax.set_ylabel("Sum of Proper Divisors") # Proper Divisor: Divisors of the number exluding the number itself.
    ax.grid(
        axis = "y",
        linestyle = "--",
        alpha = 0.7
    )
    nums: np.ndarray = np.arange(start, end+1)
    # We have to find out whether the sum of the proper divisors of a number
    #   is under or over the perfect value.
    under_values: np.ndarray = np.where(
        sums < nums,
        sums,
        0
    )
    over_values: np.ndarray = np.where(
        sums > nums,
        sums,
        0
    )
    perfect_values: np.ndarray = np.where(
        sums == nums,
        sums,
        0
    )

    # Plot the data in bar-graph.
    under_bars = ax.bar(
        nums,
        under_values,
        color = "red"
    )
    over_bars = ax.bar(
        nums,
        over_values,
        color = "green"
    )
    bar_original = ax.bar(
        nums,
        nums,
        color = "black",
        alpha = 0.5
    )
    # In order to emphasize the perfect bars, do not show the should-value as black.
    perfect_bars = ax.bar(
        nums,
        perfect_values,
        color = "gold"
    )
    
    # Annote the perfect numbers with an arrow.
    # First found out, which index-values are non zero.
    perfect_nums: np.ndarray = np.nonzero(perfect_values)[0] # It returns a tuple so we have to unpack it.
    for i in perfect_nums:
        # Get the specific bar object.
        target_bar: matplotlib.patches.Rectangle = perfect_bars[i]
        # Calculate the position.
        x_pos: int = target_bar.get_x() + target_bar.get_width() / 2
        y_pos: int = target_bar.get_height()
        # The text to display is the bar's height, since it is it's value.
        text_label: str = f"{y_pos}"
        
        ax.annotate(
            text = text_label, # Text to display
            xy = (x_pos, y_pos), # Point to point
            xytext = (x_pos, y_pos + 20), # Position for the text
            fontsize = 12, # Text font size
            fontweight = "bold", # Make the text bold.
            ha = "center", # Horizontal alignment
            arrowprops = dict( # Create an arrow.
                facecolor = "gold", 
                shrink = 0.05, # Shrink the arrow to not touch the point.
                width = 1.5,
                headwidth = 8
            )
        )

    plt.show()




    