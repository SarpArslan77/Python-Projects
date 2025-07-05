
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mplcursors
from collections import Counter

# Option 1: Animation setup and execution
def animate_each_value(
        start_value: int,
        hailstone_numbers: np.ndarray,
        total_stoppage_time: np.ndarray
) -> None:
    """
    Animates each value with from the start till finish with its steps
    Args:
        start_value: Starting value for the conjecture
        hailstone_numbers: Hailstone number sequences
        total_stoppage_time: Amount of steps per sequence
    """
    for i in range(len(hailstone_numbers)):
        current_start = start_value + i  # Current starting value (3, 4, ..., 100)
        
        # Create figure and axis for this animation
        fig, ax = plt.subplots()
        ax.set_xlim(0, max(total_stoppage_time[i]) + 1)
        ax.set_ylim(0, max(hailstone_numbers[i]) + 5)  # Y-axis padding
        ax.set_xlabel("Step Count")
        ax.set_ylabel("Value")
        ax.set_title(f"Collatz Conjecture for N = {current_start}")
        ax.grid(True)

        # Initialize line plot for current sequence
        line, = ax.plot([], [], marker='o', linestyle='-', color='b')

        # Plot previous sequences with low opacity (if any)
        if i > 0:
            for j in range(i):
                ax.plot(
                    total_stoppage_time[j], 
                    hailstone_numbers[j], 
                    alpha=0.2, 
                    color="b"
                )

        # Animation Update Function
        def update(frame: int):
            # Update current sequence data
            line.set_data(
                total_stoppage_time[i][:frame + 1], 
                hailstone_numbers[i][:frame + 1]
            )
            line.set_color((0, 0, 1, 1))  # Solid blue
            
            # Update title with current step and value
            ax.set_title(
                f"Collatz Conjecture for N = {current_start}\n"
            )
            return line,

        # Create animation
        ani = FuncAnimation(
            fig,
            update,
            frames=len(total_stoppage_time[i]),  # Number of frames = sequence length
            interval=50,  # 50ms between frames
            blit=True,    # Optimize rendering
            repeat=False  # Play once
        )

        # Display and auto-close
        plt.show(block=False)
        animation_duration = len(total_stoppage_time[i]) * 0.05  # Convert ms to seconds
        plt.pause(animation_duration + 0.25)  # Extra 0.25s buffer
        plt.close(fig)

# Option 2: Direct plotting of the results
def plot_all_value(
       start_value: int,
       end_value: int,
       hailstone_numbers: np.ndarray,
       total_stoppage_time: np.ndarray
) -> None:
    """
    Plots all the sequences for all values from start till end at once
    Args:
        start_value: Starting value for the conjecture
        end_value: Ending value for the conjecture
        hailstone_numbers: Hailstone number sequences
        total_stoppage_time: Amount of steps per sequence
    """
    # Set the settings for the plot
    fig, ax = plt.subplots()
    ax.set_xlabel("Step Count")
    ax.set_ylabel("Value")
    ax.set_title(f"Collatz Conjecture for the values between {start_value} and {end_value}")
    ax.grid(True)

    # Start the loop
    for i in range(len(hailstone_numbers)):

        line, = ax.plot(
            total_stoppage_time[i],
            hailstone_numbers[i],
            alpha = 0.2,
            color = "b"
        )
        # Add interactive cursor
        mplcursors.cursor(line)

    plt.show()

# Option 3: Plot Logaritmic value of one number
def plot_log_value_of_one_number(
        num: int,
        hailstone_numbers: np.ndarray,
        total_stoppage_time: np.ndarray,
        detrended: bool
) -> None:
    """
    Shows the logarithmic value of a number
    Args:
        num: Number to calculate the logaritmic value of
        hailstone_numbers: Hailstone number sequences
        total_stoppage_time: Amount of steps per sequence
        detrended: With or without linear trend for the logaritmic scale
    """
    fig, ax = plt.subplots()
    ax.set_xlabel("Step Count")
    ax.set_ylabel("Logaritmic Value")
    ax.set_title(f"Logaritmic Collatz Conjecture for the number {num}")
    ax.grid(True)

    log_hailstone_numbers = np.log(hailstone_numbers[0]) # Log of first and only value of the sequence
    steps = total_stoppage_time[0]

    if detrended:
        # Degrade the linear trend
        # Find the best fit line (polynomial of degree 1)
        slope, intercept = np.polyfit(steps, log_hailstone_numbers, 1)
        # Calculate the y-values according to the trend line
        trend_line = slope * steps + intercept
        # Subtract the trend from the original data
        detrended_log_hailstone_numbers = log_hailstone_numbers - trend_line
        log_hailstone_numbers = detrended_log_hailstone_numbers

    line, = ax.plot(
        steps, 
        log_hailstone_numbers, 
        color = "b"
    )
    mplcursors.cursor(line)

    plt.show()

# Option 4: Animate leading digit of each element of a hailstone sequences
def animate_leading_digit(
    start_value: int,
    end_value: int,
    hailstone_numbers: list[list[int]] 
) -> None:
    """
    Processes hailstone sequences one by one and animates the cumulative count of their leading digits.
    Args:
        start_value: Starting value for the Hailstone Sequences.
        end_value: Ending value for the sequences.
        hailstone_numbers: Hailstone Sequences itself
    """
    leading_digits_per_sequence: list[list[int]] = [
        [int(str(num)[0]) for num in sequence] for sequence in hailstone_numbers
    ]
    
    num_sequences = len(leading_digits_per_sequence)
    categories: list[str] = [str(i) for i in range(1, 10)]
    
    # This will store the counts
    cumulative_counts: np.ndarray = np.zeros(9, dtype=int)

    fig, ax = plt.subplots()
    bars = ax.bar(categories, cumulative_counts, color="skyblue")
    ax.set_ylabel("Leading Digit Count")
    ax.set_xlabel("Digit")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Add text labels on top of each bar, initialized to '0'
    text_labels = [ax.text(bar.get_x() + bar.get_width()/2.0, 0, '0', ha='center', va='bottom') for bar in bars]

    def update(frame: int):
        # Get the leading digits for the current sequence
        current_sequence_digits = leading_digits_per_sequence[frame]
        
        counts = Counter(current_sequence_digits)
        
        # Add these new counts to our cumulative total
        for i in range(9):
            cumulative_counts[i] += counts.get(i + 1, 0) # .get(key, 0) is safer

        # Update the height of each bar
        max_height = 0
        for i, bar in enumerate(bars):
            height = cumulative_counts[i]
            bar.set_height(height)
            text_labels[i].set_text(f'{height}')
            text_labels[i].set_y(height)
            if height > max_height:
                max_height = height

        # Dynamically adjust the y-axis limit to fit the growing bars
        ax.set_ylim(0, max_height * 1.15)
        
        ax.set_title(f"Leading Digit Distribution Animation between {start_value} and {end_value} (Processing Sequence {start_value + frame})")

        # Return the artists that have been modified
        return list(bars) + text_labels

    # The number of frames is the number of sequences we need to process.
    ani = FuncAnimation(
        fig,
        update,
        frames=num_sequences,
        blit=False, # blit=False is often easier to work with when text/axes are changing
        repeat=False,
        interval=50 # Speed in milliseconds between frames
    )

    plt.tight_layout()
    plt.show()

# Option 5: Plot leading digit of each element of a hailstone sequences at once
def plot_all_leading_digit(
        start_value: int,
        end_value: int,
        hailstone_numbers: list[list[int]]
) -> None:
    """
    Process hailstone sequences at once and plot the count of their leading digits together
    Args:
        start_value: Starting_value for the hailstone sequences
        end_value: Ending value for the sequences
        hailstone_numbers: Hailstone Sequences itself
    """
    leading_digits_per_sequence: list[list[int]] = [
        [int(str(num)[0]) for num in sequence] for sequence in hailstone_numbers
    ]
    num_sequences = len(leading_digits_per_sequence)
    categories: list[str] = [str(i) for i in range(1, 10)]
    cumulative_counts: np.ndarray = np.zeros(9, dtype=int)
    for i in range(len(leading_digits_per_sequence[:])):
        current_sequence_digits = leading_digits_per_sequence[i]
        counts = Counter(current_sequence_digits)
        for j in range(9):
            cumulative_counts[j] += counts.get(j+1, 0)

    fig, ax = plt.subplots()
    bars = ax.bar(categories, cumulative_counts, color="skyblue")
    ax.set_ylabel("Leading Digit Count")
    ax.set_xlabel("Digit")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_title(f"Leading Digit Distribution between {start_value} and {end_value} at once")
    ax.set_ylim(0, max(cumulative_counts)*1.1)
    text_labels = [ax.text(bar.get_x() + bar.get_width()/2.0, 0, "0", ha="center", va="bottom") for bar in bars]
    for i, bar in enumerate(bars):
        height = cumulative_counts[i]
        bar.set_height(height)
        text_labels[i].set_text(f"{height}")
        text_labels[i].set_y(height)

    plt.tight_layout()
    plt.show()

# Option 6: Plot each sequences highest reached number
def plot_highest_reached_number(
        start_value: int,
        end_value: int,
        hailstone_numbers: list[list[int]]
) -> None:
    """
    Plot the highst reached number for each sequence on a scatter plot
    Args:
        start_value: Starting_value for the hailstone sequences
        end_value: Ending value for the sequences
        hailstone_numbers: Hailstone Sequences itself
    """
    max_reached_numbers: list[int] = [
        max(single_sequence) for single_sequence in hailstone_numbers
        ]
    
    fig, ax = plt.subplots()
    ax.set_xlabel("Number")
    ax.set_ylabel("Maximum Reached Number")
    ax.set_title(f" Maximum Reached Number for each Sequence between {start_value} and {end_value}")
    ax.grid(True)

    x_numbers = list(range(start_value, end_value+1))

    scatter = ax.scatter(
        x_numbers,
        max_reached_numbers,
        color = "b",
        s=10
    )
    mplcursors.cursor(scatter)

    plt.show()
