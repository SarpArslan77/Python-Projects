
#TODO: rn every iteration is seperate window, combine them and the old
# ones should be with less opaque, so the current one is distinct

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def run_collatz_conjecture(step_value: int) -> int:
    """Performs a single step of the Collatz conjecture."""
    # if its 1, the process has ended for this step, but we let the main loop handle the stop
    if step_value == 1:
        return 1
    if step_value % 2 != 0:  # if it's odd -> 3n+1
        return 3 * step_value + 1
    else:  # if it's even -> n/2
        return int(step_value / 2)


# data generation
starting_value: int = 3
ending_value: int = 5  
curve: list[list[int]] = []
steps: list[list[int]] = []

current_value = starting_value
step_count = 1
i = 0

for x in range(starting_value, ending_value+1):

    current_value = x
    step_count = 1
    
    curve.append([current_value])
    steps.append([step_count])

    while current_value != 1:
        current_value = run_collatz_conjecture(current_value)
        step_count += 1
        curve[i].append(current_value)
        steps[i].append(step_count)

    i += 1

print(curve)
print(steps)


i = 0

# animation setup
for x in range(starting_value, ending_value+1):

    starting_value = x

    fig, ax = plt.subplots()
    ax.set_xlim(0, max(steps[i]) + 1)
    ax.set_ylim(0, max(curve[i]) + 5) # Add some padding to the y-axis
    ax.set_xlabel("Step Count")
    ax.set_ylabel("Value")
    ax.set_title(f"Collatz Conjecture for N = {starting_value}")
    ax.grid(True)

    line, = ax.plot([], [], marker='o', linestyle='-', color='b')

    def old_ones(): # just draw the old ones with less opaquenes to show the progress
        pass

    def update(frame):

        # set the data for the current step
        line.set_data(steps[i][:frame+1], curve[i][:frame+1])
        line.set_color((0, 0, 1, 0.2))
        # dynamically update the title
        current_step = steps[i][frame]
        current_value = curve[i][frame]
        ax.set_title(f"Collatz Conjecture for N = {starting_value}\nStep: {current_step}, Value: {current_value}")
        return line,

    ani = FuncAnimation(fig, update, frames=len(steps[i]), interval=50, blit=True, repeat=False)

    plt.show()

    i += 1

