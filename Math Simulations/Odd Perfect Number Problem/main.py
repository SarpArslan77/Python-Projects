
import numpy as np

from find_perfect_number import (
    brute_force
)
from plot import (
    plot_intervall
)

if __name__ == "__main__":
    start, end = 1, 100
    sums: np.ndarray = brute_force(
        start,
        end
    )
    plot_intervall(
        start,
        end,
        sums
    )