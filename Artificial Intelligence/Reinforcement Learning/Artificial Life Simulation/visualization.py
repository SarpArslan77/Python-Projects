
# visualization.py

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
import pygame
from scipy.ndimage import gaussian_filter

@dataclass(frozen=True)
class ConfigVisualizer:
    simulation_size: int
    window_size: int
    blur_strength: int # Standard Deviation of the Gaussian bell curve.
    terrain_thresholds: list[int, int, int]

    # Validate the inputs.
    def __post_init__(self) -> None:
        pass

class Visualizer: #TODO AD

    grid_colors: dict[str, tuple[int, int, int]] = {
        "white" : (255, 255, 255),
        "black" : (0, 0, 0)
    }

    terrain_colors: NDArray = np.array(
        [
            [0, 0, 255], # Water (Blue): ID 0.
            [0, 255, 0], # Grass (Green): ID 1 ...
            [92, 64, 51], # Rock (Dark Brown).
            [169, 169, 169] # Mountain (Gray).

        ],
        dtype = np.uint8
    )

    resource_colors: NDArray = np.array(
        [
            [255, 255, 255], # Empty (White): ID 0.
            [150, 75, 0], # Food (Brown): ID 1 ...
            [255, 165, 0], # Corpse (Orange).
            [0, 100, 0] # Waste (Dark Green).
        ],
        dtype = np.uint8
    )

    def __init__(
            self,
            config_visualizer: ConfigVisualizer
    ) -> None: 
        
        # 1. Parameter definitions.
        self.cfg = config_visualizer

        # 2. Pygame initialization.
        pygame.init()

        # 3. Screen and Clock setup.
        self.screen = pygame.display.set_mode((self.cfg.window_size, self.cfg.window_size))
        self.clock = pygame.time.Clock()

        # 4. Create the simulation surface..
        self.simulation_surface = pygame.Surface((self.cfg.simulation_size, self.cfg.simulation_size))

        # 5. Create the grid surface.
        self.grid_surface = pygame.Surface((self.cfg.window_size, self.cfg.window_size), pygame.SRCALPHA)

        # Calculate the cell size.
        cell_size: int = self.cfg.window_size // self.cfg.simulation_size

        # Draw the vertical grid lines.
        for x in range(0, self.cfg.window_size, cell_size):
            pygame.draw.line(
                self.grid_surface,
                self.grid_colors["black"],
                (x, 0),
                (x, self.cfg.window_size)
            )
        # Draw the horizontal grid lines.
        for y in range(0, self.cfg.window_size, cell_size):
            pygame.draw.line(
                self.grid_surface,
                self.grid_colors["black"],
                (0, y),
                (self.cfg.window_size, y)
            )

        # 6. Active choosen map.
        self.draw_terrain: bool = True
        self.draw_resources: bool = False

        # 7. Creates matrixes to represent the maps.
        self.terrain_matrix: NDArray = np.zeros(
            shape = (self.cfg.simulation_size, self.cfg.simulation_size),
            dtype = np.uint8
        )

        self.resource_matrix: NDArray = np.zeros(
            shape = (self.cfg.simulation_size, self.cfg.simulation_size),
            dtype = np.uint8
        )

        # 8. Create the terrain.
        self._create_terrain()

    def _create_terrain(self) -> None:
        # 1. Create the generator.
        rng = np.random.default_rng()

        # 2. Generates a matrix full of random floats.
        random_matrix: NDArray = rng.random(
            size = (self.cfg.simulation_size, self.cfg.simulation_size),
            dtype = np.float32
        )

        # 3. Apply Gaussian Filter.
        blurred_matrix: NDArray = gaussian_filter(
           input = random_matrix,
           sigma = self.cfg.blur_strength # Controls the amount of blur.
        )

        # 4. Filters the levels according to thresholds.
        # Since the Gaussian Filter creates a bell curve, the values below which a specific percentage of the data falls, need to be determined.
        adapted_water_threshold, adapted_grass_threshold, adapted_rock_threshold = np.percentile(
            a = blurred_matrix,
            q = self.cfg.terrain_thresholds
        )

        # Create the threshold bin.
        bins: NDArray = np.array([adapted_water_threshold, adapted_grass_threshold, adapted_rock_threshold])

        self.terrain_matrix: NDArray = np.digitize( # Bins continuous data into discrete categories.
            x = blurred_matrix,
            bins = bins
        )

    def _draw_matrix(
            self,
            colors: NDArray,
            matrix: NDArray
    ) -> None: #TODO AD
        # 1. 'Advanced Indexing' replaces IDs with RGB values.
        rgb_matrix: NDArray = colors[matrix] # (Width, Height, 3).

        # 2. Swaps the first two axes (rows and columns), while keeping the color channel (depth).
        transposed_rgb_matrix: NDArray = np.swapaxes(
            a = rgb_matrix,
            axis1 = 0,
            axis2 = 1
        ) # (Height, Width, 3).

        # 3. Blits the RGB data onto the tiny surface.
        pygame.surfarray.blit_array(
            self.simulation_surface,
            transposed_rgb_matrix
        )

        # 4. Scales the tiny surface up to the choosen size.
        pygame.transform.scale(
            self.simulation_surface,
            (self.cfg.window_size, self.cfg.window_size),
            self.screen
        )

    def game_loop(self) -> bool: #TODO AD
        # 1. Detect whether a key is pressed.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.K_0:
                self.draw_terrain = not self.draw_terrain
            if event.type == pygame.K_1:
                self.draw_resources = not self.draw_resources
        
        # 2. Resets the screen to white every frame.
        self.screen.fill(self.grid_colors["white"])

        # 3. If so, draw the maps.
        if self.draw_terrain:
            self._draw_matrix(
                colors = self.terrain_colors,
                matrix = self.terrain_matrix
            )
        if self.draw_resources:
            self._draw_matrix(
                colors = self.resource_colors,
                matrix = self.resource_matrix
            ) # Scales a solid image onto screen, it paints over everything.

        # 4. Draw the grid.
        self.screen.blit(
            self.grid_surface,
            (0, 0), #TODO AC
        )

        # 5. Update the screen.
        pygame.display.update()

        return True

