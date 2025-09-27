import numpy as np
import matplotlib.pyplot as plt
import mplcursors

def plot_prime_counting_function(n: int, step_points: np.ndarray[int]) -> None:
    """
    Plot the prime counting function π(x) as a step function.
    Parameters:
    n (int): Maximum number to display
    step_points (np.ndarray[int]): Array of prime numbers (x-values where steps occur)
    """
    # Create the step function data
    x_values = [0]  # Start at (0, 0)
    y_values = [0]
    
    # Create steps at each prime number
    for i, prime in enumerate(step_points, start=1):
        x_values.extend([prime, prime])
        y_values.extend([i-1, i])  # Step up by 1 at each prime
    
    # Extend to n if needed
    if x_values[-1] < n:
        x_values.append(n)
        y_values.append(y_values[-1])
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    line = ax.step(x_values, y_values, where='post', linewidth=2, color='blue')[0]
    
    # Formatting
    ax.set_xlim(0, n)
    ax.set_ylim(0, len(step_points)+1)
    ax.set_xlabel("Numbers", fontsize=12)
    ax.set_ylabel("Total number of primes", fontsize=12)
    ax.set_title(f"Prime Counting Function π(x) up to {n}", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add interactive cursor
    cursor = mplcursors.cursor(line, hover=True)
    
    # Customize what the tooltip shows
    @cursor.connect("add")
    def on_add(sel):
        x, y = sel.target
        if sel.artist == line:
            # For the line, show the current count
            sel.annotation.set(text=f"π({x:.0f}) = {y:.0f}",
                             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9))
        sel.annotation.get_bbox_patch().set_alpha(0.8)
    
    plt.tight_layout()
    plt.show()