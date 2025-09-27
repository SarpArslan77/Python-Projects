
import numpy as np

from animate_riemann_hypothesis_datas import plot_prime_counting_function
from prime_counting_function import prime_counting_function

n: int = 100000
primes: np.ndarray[int] = prime_counting_function(n)

plot_prime_counting_function(n, primes)
