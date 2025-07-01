


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
        ) -> tuple[list[list[int]], list[list[int]]]:
    """
    Creates Hailstone Sequences between a start and end value
    Args:
        start_value (int): Starting value for the HS generation
        end_value (int): Ending value for the HS generation
        hailstone_numbers (list[list[int]]) : 
        total_stoppage_time (list[list[int]]) : 
    Returns:
        (list[list[int]], list[list[int]]): List of generated hailstone numbers
            and list of total steps for each generated hailstone number
    """
    hailstone_numbers: list[list[int]] = []
    total_stoppage_time: list[list[int]] = []

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
        
        hailstone_numbers.append(current_sequence)
        total_stoppage_time.append(current_steps)
    
    return (hailstone_numbers, total_stoppage_time)
    