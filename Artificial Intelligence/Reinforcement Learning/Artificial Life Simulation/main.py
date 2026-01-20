
# main.py

#! Custom TODO notes:
#TODO AD: Add docstring.
#TODO ATH: Add type hint.
#TODO CTH: Check type hint.
#TODO FTH: Fix the hardcoding.
#TODO HPE: Handle possible error.
#TODO AC: Add comment.
#TODO AIC: Add input control.

#! PW: Possibly wrong.

import pygame

if __name__ == "__main__":

    from visualization import (
        ConfigVisualizer,
        Visualizer
    )

    from simulation_resources import (
        ConfigResourceManager,
        ResourceManager
    )

    # 2. Pygame initialization.
    pygame.init()

    config_resource_manager = ConfigResourceManager(
        simulation_size = 100,
        waste_decay_rate = 0.0005,
        corpse_decay_rate = 0.0005
    )

    resource_manager = ResourceManager(
        config_resource_manager = config_resource_manager
    )

    #? resource_manager.debug_randomize_resources()

    config_visualizer = ConfigVisualizer(
        simulation_size = 100,
        window_size = 800,
        blur_strength = 6,
        terrain_thresholds = [20, 70, 95]
    )

    visualizer = Visualizer(
        config_visualizer=config_visualizer,
        resource_manager = resource_manager
    )

    running: bool = True
    while running:
        running = visualizer.simulation_loop()
        resource_manager.resource_loop()
