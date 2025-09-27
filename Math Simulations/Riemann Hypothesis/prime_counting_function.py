
import numpy as np

def prime_counting_function(
        n: int
    ) -> np.ndarray:
    """
    Compute all prime numbers up to n using the Sieve of Eratosthenes algorithm.
    Args:
        n: Upper limit for prime number generation
    Returns:
        np.ndarray: Array of prime numbers ≤ n
    """
    # Handle edge cases
    if n < 2:
        return np.array([], dtype=np.int64)
    
    # Initialize sieve (boolean array where index represents the number)
    #   True means prime, False means composite
    sieve = np.ones(n + 1, dtype=bool)  # Creates array of True values
    
    # 0 and 1 are not primes
    sieve[0:2] = False  # Set first two elements to False
    
    # Main sieve algorithm
    for i in range(2, int(np.sqrt(n)) + 1):
        if sieve[i]:  # If i is still marked as prime
            #   Mark multiples of i starting from i^2
            sieve[i*i::i] = False
    
    # Extract primes - indices where sieve is True
    primes = np.nonzero(sieve)[0]
    
    return primes
