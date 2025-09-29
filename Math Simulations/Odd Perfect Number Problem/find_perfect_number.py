import numpy as np
import math

def brute_force(
        start: int,
        end: int
) -> np.ndarray[int]:
    """
    Check every number smaller then the given number to see if it's a proper divisor.
        If it is add it to a running sum.
    """
    all_sums: list[int] = []
    # Calculate the sum for all the numbers in the intervall.
    for num in range(start, end+1):
        proper_divisors: list[int] = []
        max_possible_divisor = int(np.sqrt(num)) # Round up to positiv infinity.
        # Check whether it is a possible divisor.
        for div in range(1, max_possible_divisor+1):
            # If a divisor, then add both of them.
            if num % div == 0:
                proper_divisors.append(div)
                proper_divisors.append(num//div)
        # Edge Case: If the number is a perfect square, the square root is added 2 times.
        root = math.isqrt(num)
        if (root*root) == num:
            proper_divisors.remove(root)
        # Remove the number itself from the proper divisors.
        proper_divisors.remove(num)
        # Calculate the sum of proper divisors.
        sum: int = np.sum(proper_divisors)
        all_sums.append(sum)
    
    return np.array(all_sums)

if __name__ == "__main__":
    test_intervall = brute_force(
        1, 30
    )
    print(test_intervall)