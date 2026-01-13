
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

if __name__ == "__main__":

    from visualization import (
        ConfigVisualizer,
        Visualizer
    )

    config_visualizer = ConfigVisualizer(
        simulation_size = 100,
        window_size = 800,
        blur_strength = 6,
        terrain_thresholds = [20, 70, 95]
    )

    visualizer = Visualizer(config_visualizer=config_visualizer)

    running: bool = True
    while running:
        running = visualizer.game_loop()