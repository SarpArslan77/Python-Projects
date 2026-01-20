
# resource.py

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

@dataclass
class ConfigResourceManager:
    simulation_size: int
    waste_decay_rate: float
    corpse_decay_rate: float

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
        self.corpse_matrix: NDArray = np.zeros(
            shape = (self.config.simulation_size, self.config.simulation_size),
            dtype = np.float16
        )

        self.food_matrix: NDArray = np.zeros(
            shape = (self.config.simulation_size, self.config.simulation_size),
            dtype = np.uint8
        )

        self.waste_matrix: NDArray = np.zeros(
            shape = (self.config.simulation_size, self.config.simulation_size),
            dtype = np.float16
        )

        self.root_matrix: NDArray = np.zeros(
            shape = (self.config.simulation_size, self.config.simulation_size),
            dtype = np.uint8
        )

    def _decay_resource(self) -> None:
        # 1. Decays the wastes.
        self.waste_matrix -= self.config.waste_decay_rate

        # 2. Decays the corpses.
        self.corpse_matrix -= self.config.corpse_decay_rate

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
