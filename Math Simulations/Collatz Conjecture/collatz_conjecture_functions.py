
import numpy as np

# Collatz Conjecture Calculation
def run_collatz_conjecture(step_value: int) -> int:
    """
    Performs a single step of the Collatz conjecture.
    Args:
        step_value (int): Current value in the sequence.
    Returns:
        int: Next value in the sequence.
    """
    if step_value == 1:  # Sequence ends at 1
        return 1
    if step_value % 2 != 0:  # Odd: 3n + 1
        return 3 * step_value + 1
    else:  # Even: n/2
        return int(step_value / 2)

# hailstone_numbers stores sequences for each starting value
# total_stoppage_time stores step counts for each sequence 

# Generate Hailstone Sequences (Collatz Paths)
def generate_hailstone_sequences(
        start_value: int, 
        end_value: int
        ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Creates Hailstone Sequences between a start and end value
    Args:
        start_value (int): Starting value for the HS generation
        end_value (int): Ending value for the HS generation 
    Returns:
        tuple: Numpy array of generated hailstone numbers,
            Numpy array of total steps for each generated hailstone number
    """
    hailstone_numbers: list[np.ndarray] = []
    total_stoppage_time: list[np.ndarray] = []

    for x in range(start_value, end_value + 1):
        current_value = x
        step_count = 1
        current_sequence = [current_value]  # Track current sequence
        current_steps = [step_count]       # Track steps
        
        while current_value != 1:
            current_value = run_collatz_conjecture(current_value)
            step_count += 1
            current_sequence.append(current_value)
            current_steps.append(step_count)
        
        hailstone_numbers.append(np.array(current_sequence, dtype=np.int64))
        total_stoppage_time.append(np.array(current_steps, dtype=np.int64))
    
    return (hailstone_numbers, total_stoppage_time)
    
