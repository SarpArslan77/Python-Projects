
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

    # 1. Imports classes.
    from visualization import (
        ConfigVisualizer,
        Visualizer
    )

    from simulation_resources import (
        ConfigResourceManager,
        ResourceManager
    )

    from agent import(
        ConfigAgentManager,
        AgentManager
    )

    # 2. Defines constants.
    SIMULATION_SIZE: int = 100
    WINDOW_SIZE: int = 800

    # 3. Initializes pygame.
    pygame.init()

    # 4. Declares the classes.
    config_resource_manager = ConfigResourceManager(
        simulation_size = SIMULATION_SIZE
    )
    resource_manager = ResourceManager(
        config_resource_manager = config_resource_manager
    )

    #! resource_manager.debug_randomize_resources()

    config_agent_manager = ConfigAgentManager()
    agent_manager = AgentManager(
        config_agent_manager = config_agent_manager,
        resource_manager = resource_manager,
        simulation_size = SIMULATION_SIZE
    )

    config_visualizer = ConfigVisualizer(
        simulation_size = SIMULATION_SIZE,
        window_size = WINDOW_SIZE
    )
    visualizer = Visualizer(
        config_visualizer = config_visualizer,
        resource_manager = resource_manager,
        agent_manager = agent_manager
    )

    # 5. Creates the starting generation for the simulation.
    agent_manager.agents_df = agent_manager.create_agents(
        old_df = agent_manager.agents_df,
        count = 100,
        x_positions = None,
        y_positions = None
    )

    running: bool = True
    while running:
        running = visualizer.simulation_loop()

        resource_manager.resource_loop()

        agent_manager.agent_loop(map_temperature=25)
