
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
import polars as pl
from polars import DataFrame
import pygame
from scipy.ndimage import gaussian_filter
from types import MappingProxyType

from simulation_resources import ResourceManager
from agent import AgentManager

@dataclass(frozen=True)
class ConfigVisualizer: #TODO AD
    simulation_size: int
    window_size: int
    BLUR_STRENGTH: int = 6 # Standard Deviation of the Gaussian bell curve.
    TERRAIN_THRESHOLDS: tuple[int] = (20, 70, 95)
    FPS: int = 60

    # Validates the inputs.
    def __post_init__(self) -> None:
        pass

class Visualizer: #TODO AD

    # Define class constants.
    ID_EMPTY: int = 1
    ID_WASTE: int = 1
    ID_FOOD: int = 2
    ID_CORPSE: int = 3

    ID_AGENT: int = 1

    _grid_colors_data: dict[str, tuple[int, int, int]] = {
        "white" : (255, 255, 255),
        "black" : (0, 0, 0)
    }
    # Creates read-only view of a dictionary.
    grid_colors = MappingProxyType(mapping=_grid_colors_data) # Attempting to set the items raises a TypeError.

    terrain_colors: NDArray = np.array(
        [
            [0, 0, 255], # Water (Blue): ID 0.
            [0, 255, 0], # Grass (Green): ID 1 ...
            [92, 64, 51], # Rock (Dark Brown).
            [169, 169, 169] # Mountain (Gray).

        ],
        dtype = np.uint8
    )
    terrain_colors.setflags(write=False) # Locks the write permission.

    resource_colors: NDArray = np.array(
        [
            [255, 255, 255], # Empty (White).
            [0, 100, 0], # Waste (Dark Green).
            [150, 75, 0], # Food (Brown).
            [255, 165, 0], # Corpse (Orange).
        ],
        dtype = np.uint8
    )
    resource_colors.setflags(write=False) 

    def __init__(
            self,
            config_visualizer: ConfigVisualizer,
            resource_manager: ResourceManager,
            agent_manager: AgentManager
    ) -> None: 
        
        # 1. Parameter definitions.
        self.config: ConfigVisualizer = config_visualizer
        self.resource_manager: ResourceManager = resource_manager
        self.agent_manager: AgentManager = agent_manager

        # 2. Screen and Clock setup.
        self.screen = pygame.display.set_mode((self.config.window_size, self.config.window_size))
        self.clock = pygame.time.Clock()

        # 3. Creates the simulation surface..
        self.simulation_surface = pygame.Surface((self.config.simulation_size, self.config.simulation_size))

        # 4. Creates the grid surface.
        self.grid_surface = pygame.Surface((self.config.window_size, self.config.window_size), pygame.SRCALPHA)

        # Calculates the cell size.
        cell_size: int = self.config.window_size // self.config.simulation_size

        # Draws the vertical grid lines.
        for x in range(0, self.config.window_size, cell_size):
            pygame.draw.line(
                self.grid_surface,
                self.grid_colors["black"],
                (x, 0),
                (x, self.config.window_size)
            )
        # Draws the horizontal grid lines.
        for y in range(0, self.config.window_size, cell_size):
            pygame.draw.line(
                self.grid_surface,
                self.grid_colors["black"],
                (0, y),
                (self.config.window_size, y)
            )

        # 5. Active chosen map.
        self.draw_terrain: bool = True
        self.draw_resources: bool = False
        self.draw_agents: bool = True

        # 6. Creates matrices to represent the maps.
        self.terrain_matrix: NDArray = np.zeros(
            shape = (self.config.simulation_size, self.config.simulation_size),
            dtype = np.uint8
        ) # (0: Water, 1: Grass, 2: Rock, 3: Mountain).

        self.resource_matrix: NDArray = np.zeros(
            shape = (self.config.simulation_size, self.config.simulation_size),
            dtype = np.uint8
        ) # (0: Empty, 1: Waste, 2: Food, 3:Corpse)

        self.agent_matrix: NDArray = np.zeros(
            shape = (self.config.simulation_size, self.config.simulation_size),
            dtype = np.uint8
        ) # (0: Empty, 1: Agent).

        self.agent_color_matrix: NDArray = np.zeros(
            shape = (self.config.simulation_size, self.config.simulation_size, 3),
            dtype = np.uint8
        ) # (0: Empty, e.g. [31, 69, 0]: Agent).

        # 7. Creates a render buffer.
        self.render_buffer: NDArray = np.zeros(
            shape = (self.config.simulation_size, self.config.simulation_size, 3),
            dtype = np.uint8
        )

        # 8. Creates the terrain.
        self._create_terrain()

    def _create_terrain(self) -> None:
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

    def _render_frame(self) -> None:
        # 1. If so, draw the terrain.
        if self.draw_terrain:
            # Fills the existing buffer.
            self.render_buffer[:] = self.terrain_colors[self.terrain_matrix]
        else:
            # If not, falls back to white background.
            self.render_buffer[:] = self.grid_colors["white"]
        
        # 2. Overlays resources.
        if self.draw_resources:
            # Creates a resource mask, where the resources are not empty.
            resource_mask: NDArray = self.resource_matrix > 0

            # Applies the resource overlay.
            self.render_buffer[resource_mask] = self.resource_colors[self.resource_matrix][resource_mask]

        # 3. Overlays agents.
        if self.draw_agents:
            # Creates a agent mask, where agents exist.
            agent_mask: NDArray = self.agent_matrix > 0

            # Applies the agent overlay.
            self.render_buffer[agent_mask] = self.agent_color_matrix[agent_mask]

        # 4. Transpose for pygame.
        transposed_rgb: NDArray = np.swapaxes(
            a = self.render_buffer,
            axis1 = 0,
            axis2 = 1
        )

        # 5. Blit the tiny surface.
        pygame.surfarray.blit_array(
            self.simulation_surface,
            transposed_rgb
        )

        # 6. Scale to window size.
        pygame.transform.scale(
            self.simulation_surface,
            (self.config.window_size, self.config.window_size),
            self.screen
        )

    def _update_resource_matrix(
            self,
            waste_matrix: NDArray,
            food_matrix: NDArray,
            corpse_matrix: NDArray
    ) -> None:
        # 1. Resets the matrix to ID 0 (Empty).
        self.resource_matrix.fill(0)

        # 2. Assigns IDs based on priority (Corpse > Food > Waste).
        self.resource_matrix[waste_matrix > 0] = self.ID_WASTE
        self.resource_matrix[food_matrix > 0] = self.ID_FOOD
        self.resource_matrix[corpse_matrix > 0] = self.ID_CORPSE

    def _update_agent_matrix(
            self,
            agents_df: DataFrame
    ) -> None:
        # 1. Resets the matrix to ID 0 (Empty).
        self.agent_matrix.fill(0)
        self.agent_color_matrix.fill(0)

        # 2. Extracts the positions of agents as array from the dataframe and converts them into necessary data types.
        x_pos: NDArray = agents_df["x_pos"].to_numpy().astype(dtype=np.int16) # Polar columns has to be first extracted and then converted.
        y_pos: NDArray = agents_df["y_pos"].to_numpy().astype(dtype=np.int16)

        # 3. Stacks the colors into array (N, 3).
        colors: NDArray = np.column_stack(
            (
                agents_df["color_r"].to_numpy().astype(dtype=np.uint8),
                agents_df["color_g"].to_numpy().astype(dtype=np.uint8),
                agents_df["color_b"].to_numpy().astype(dtype=np.uint8)
            )
        )

        # 4. Updates the matrices.
        self.agent_matrix[y_pos, x_pos] = self.ID_AGENT
        self.agent_color_matrix[y_pos, x_pos] = colors

    def simulation_loop(self) -> bool: #TODO AD
        # 1. Detects whether a key is pressed.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    self.draw_terrain = not self.draw_terrain
                if event.key == pygame.K_2:
                    self.draw_resources = not self.draw_resources
                if event.key == pygame.K_3:
                    self.draw_agents = not self.draw_agents
        
        # 2. Updates the agents.
        agents_df: DataFrame = self.agent_manager.get_agent_state()

        self._update_agent_matrix(agents_df=agents_df)

        # 3. Update the resources.
        waste_matrix, food_matrix, corpse_matrix = self.resource_manager.get_resource_state()
        
        self._update_resource_matrix(
            waste_matrix = waste_matrix,
            food_matrix = food_matrix,
            corpse_matrix = corpse_matrix
        )

        # 4. Resets the screen to white every frame.
        self.screen.fill(self.grid_colors["white"])

        # 5. Renders one frame.
        self._render_frame()

        # 6. Draws the grid.
        self.screen.blit(
            self.grid_surface,
            (0, 0), #TODO AC
        )

        # 7. Updates the screen.
        self.clock.tick(self.config.FPS)
        pygame.display.update()

        return True
