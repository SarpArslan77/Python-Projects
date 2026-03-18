
# data_visualization.py

from dataclasses import (
    dataclass,
    field
)

import numpy as np
from numpy.typing import NDArray
import polars as pl
from polars import DataFrame

from agent import AgentManager
from matplotlib_auto_plotter import MatplotlibAutoPlotter


@dataclass(frozen=True)
class ConfigDataVisualizer:

    # Defines columns shown in the live updating data table.
    TARGET_COLUMNS: NDArray = field(
        default_factory = lambda: np.array(
            [
            #"sex",
            "size", "max_age", "max_health", "max_energy", "optimal_temperature", "resistance",
            #"base_energy_burn_rate", "chronotype", "plant_digestion_efficiency", "meat_digestion_efficiency",
            #"has_venom", "has_mimicry",
            #"kinship_loyalty", "parental_instinct", "hostility", "fight_flight_bias",
            #"age", "health", "energy", "waste_accumulated",
            #"total_energy_burn_rate", "effective_resistance", "metabolism_modifier", "health_cost",
            #"is_infected", "is_bleeding", "is_sleeping", "is_pregnant", "is_hibernating",
            #"color_r", "color_g", "color_b"
        ],
        dtype = "U" # Tells numpy to find the length of the longest string.
        )
    )

class DataVisualizer:

    def __init__(
            self,
            config_data_visualizer: ConfigDataVisualizer,
            matplotlib_auto_plotter: MatplotlibAutoPlotter,
            agent_manager: AgentManager,
    ) -> None:
        self.cfg: ConfigDataVisualizer = config_data_visualizer
        self.matplotlib_auto_plotter: MatplotlibAutoPlotter = matplotlib_auto_plotter
        self.agent_manager: AgentManager = agent_manager

        self.stats_history: dict[str, list] = {col: [] for col in self.cfg.TARGET_COLUMNS}

    def update_stats_history(
            self
    ) -> None:
        # 1. Gets the current dataframe from the agent manager class.
        df: DataFrame = self.agent_manager.get_agent_dataframe()

        # 2. Calculates the mean values for all columns and converts them into a dictionary.
        current_stats: dict[str, list] = df.select( # Selects columns.
            (pl.col(self.cfg.TARGET_COLUMNS).mean()) # Calculates mean.
        ).row( # Retrieves the data at a specific horizontal position.
            0, # Gives the first (and here only) row of these results.
            named=True # Returns a dictionary by mappin gthe names(keys) to their calculated means(values).
            #   Otherwise only the values are returned.
        )

        # 3. Updates the history dictionary.
        for col, val in current_stats.items():
            self.stats_history[col].append(val)

    def plot_data(
            self
    ) -> None:
        self.matplotlib_auto_plotter.generate_plot(history=self.stats_history)
