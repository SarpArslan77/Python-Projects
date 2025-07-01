
#TODO: make the graph interactable, such as zoom in / zoom out

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collatz_conjecture_functions import generate_hailstone_sequences


start_value: int = 3
end_value: int = 1000

# Call of the main function
hailstone_numbers, total_stoppage_time = generate_hailstone_sequences(start_value, end_value)

"""
# Option 1: Animation setup and execution
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
"""

# Option 2: Direct plotting of the results
# Set the settings for the plot
fig, ax = plt.subplots()
ax.set_xlabel("Step Count")
ax.set_ylabel("Value")
ax.set_title(f"Collatz Conjecture for the values between {start_value} and {end_value}")
ax.grid(True)

# Start the loop
for i in range(len(hailstone_numbers)):
    current_start = start_value + i

    ax.plot(
        total_stoppage_time[i],
        hailstone_numbers[i],
        alpha = 0.2,
        color = "b"
    )

plt.show()