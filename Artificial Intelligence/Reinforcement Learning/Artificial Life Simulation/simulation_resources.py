
# simulation_resources.py

#! Custom TODO notes:
#TODO AD: Add docstring.
#TODO ATH: Add type hint.
#TODO CTH: Check type hint.
#TODO FTH: Fix the hardcoding.
#TODO HPE: Handle possible error.
#TODO AC: Add comment.
#TODO AIC: Add input control.

#! PW: Possibly wrong.

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

@dataclass
class ConfigResourceManager:
    simulation_size: int
    WASTE_DECAY_RATE: float = 0.005
    CORPSE_DECAY_RATE: float = 0.005

    TERRAIN_THRESHOLDS: tuple[int] = (20, 70, 95)
    BLUR_STRENGTH: int = 6 # Standard Deviation of the Gaussian bell curve.

# Validates the inputs.
def __post_init__(self) -> None:
    pass

class ResourceManager: #TODO AD

    def __init__(
            self,
            config_resource_manager: ConfigResourceManager
    ) -> None:
        # 1. Defines input parameters.
        self.config = config_resource_manager

        # 2. Creates different layers.
        self.terrain_matrix: NDArray = np.zeros(
            shape = (self.config.simulation_size, self.config.simulation_size),
            dtype = np.uint8
        ) # (0: Water, 1: Grass, 2: Rock, 3: Mountain).

        self.corpse_matrix: NDArray = np.zeros(
            shape = (self.config.simulation_size, self.config.simulation_size),
            dtype = np.float16
        )

        self.food_matrix: NDArray = np.zeros(
            shape = (self.config.simulation_size, self.config.simulation_size),
            dtype = np.float16
        )

        self.waste_matrix: NDArray = np.zeros(
            shape = (self.config.simulation_size, self.config.simulation_size),
            dtype = np.float16
        )

        self.root_matrix: NDArray = np.zeros(
            shape = (self.config.simulation_size, self.config.simulation_size),
            dtype = np.float16
        )

        # 8. Creates the terrain.
        self.create_terrain()

    def create_terrain(self) -> NDArray:
        # 1. Creates the generator.
        rng = np.random.default_rng()

        # 2. Generates a matrix full of random floats.
        random_matrix: NDArray = rng.random(
            size = (self.config.simulation_size, self.config.simulation_size),
            dtype = np.float32
        )

        # 3. Applys Gaussian Filter.
        blurred_matrix: NDArray = gaussian_filter(
           input = random_matrix,
           sigma = self.config.BLUR_STRENGTH # Controls the amount of blur.
        )

        # 4. Filters the levels according to thresholds.
        # Since the Gaussian Filter creates a bell curve, the values below which a specific percentage of the data falls, need to be determined.
        thresholds: NDArray = np.percentile( # q: Quantile.
            a = blurred_matrix,
            q = self.config.TERRAIN_THRESHOLDS
        )

        # 5. Digitizes the matrix using the calculated thresholds.
        self.terrain_matrix: NDArray = np.digitize( # Bins continuous data into discrete categories.
            x = blurred_matrix,
            bins = thresholds
        ).astype(np.uint8) # Creates a copy of the array with the new type. ('digitize' returns by default 'int64').

        return self.terrain_matrix

    def _decay_resource(self) -> None:
        # 1. Decays the wastes.
        self.waste_matrix = np.maximum(self.waste_matrix - self.config.WASTE_DECAY_RATE, 0.0) # 'maximum' does the comparison elementwise, whereas 'max' does it for the whole matrix together.

        # 2. Decays the corpses.
        self.corpse_matrix = np.maximum(self.corpse_matrix - self.config.CORPSE_DECAY_RATE, 0.0)

    def get_resource_state(self) -> tuple[NDArray, NDArray, NDArray]:
        return (self.waste_matrix, self.food_matrix, self.corpse_matrix)

    def resource_loop(self) -> None:
        # 1. Decays the resource.
        self._decay_resource()
        
"""
    def debug_randomize_resources(self) -> None:
            # 1. Setup Generator
            rng = np.random.default_rng()
            rows, cols = self.waste_matrix.shape
            
            # 2. Create a "Dice Roll" for every cell (0.0 to 1.0)
            # Generating one float array is faster than generating three boolean arrays.
            chance = rng.random((rows, cols), dtype=np.float32)

            # 3. Assign Resources based on probability buckets
            # Clear previous data
            self.waste_matrix.fill(0)
            self.food_matrix.fill(0)
            self.corpse_matrix.fill(0)

            # [0.00 - 0.10]: 10% Chance for Waste
            self.waste_matrix[chance < 0.10] = 1

            # [0.10 - 0.15]: 5% Chance for Food (No overlap with Waste)
            self.food_matrix[(chance >= 0.10) & (chance < 0.15)] = 1

            # [0.98 - 1.00]: 2% Chance for Corpse
            self.corpse_matrix[chance > 0.98] = 1
"""
