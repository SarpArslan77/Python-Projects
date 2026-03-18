# agent.py

#! Custom TODO notes:
#TODO AD: Add docstring.
#TODO ATH: Add type hint.
#TODO CTH: Check type hint.
#TODO FTH: Fix the hardcoding.
#TODO HPE: Handle possible error.
#TODO AC: Add comment.
#TODO AIC: Add input control.
#TODO ACV: Add constant variable.
#TODO WOO: Work on organising.

#! PW: Possibly wrong.

from dataclasses import (
    dataclass,
    field
)

import numpy as np
from numpy.typing import NDArray
import polars as pl
from polars import DataFrame

from simulation_resources import ResourceManager

@dataclass(frozen=True)
class ConfigAgentManager:
    """
    Configuration for the Agent Manager.
    Variables are grouped logically by function.
    """

    # 1. SIMULATION & MATRIX CONSTANTS
    EMPTY_MATRIX_ID: int = 0
    AGENT_MATRIX_ID: int = 1

    # Order: Up, Right, Down, Left
    MOVEMENT_DIRECTIONS: NDArray = field(
        default_factory=lambda: np.array(
            ((0, -1), (1, 0), (0, 1), (-1, 0)),
            dtype=np.int8
        )
    )

    VISION_TEMPLATE: NDArray = field(
        default_factory=lambda: np.column_stack(
            (
                np.indices(((4 * 2) + 1, (4 * 2) + 1))[1].flatten() - 4,  # X Values
                np.indices(((4 * 2) + 1, (4 * 2) + 1))[0].flatten() - 4,  # Y Values
            )
        ).astype(np.int8)
    )

    # Derived fields (calculated in __post_init__)
    VISION_TEMPLATE_X: NDArray = field(init=False)
    VISION_TEMPLATE_Y: NDArray = field(init=False)
    NEIGHBOUR_INDICES: NDArray = field(init=False)

    # 2. PHYSICAL GENETICS
    MIN_SIZE: int = 1
    MAX_SIZE: int = 3
    MIN_AGE: int = 50
    MAX_AGE: int = 250
    MIN_HEALTH: int = 50
    MAX_HEALTH: int = 250
    MIN_ENERGY: int = 50
    MAX_ENERGY: int = 250
    MIN_TEMP: int = 0
    MAX_TEMP: int = 50
    
    MIN_ENERGY_BURN: float = 0.1
    MAX_ENERGY_BURN: float = 1.0
    MIN_RESISTANCE: float = 0.1
    MAX_RESISTANCE: float = 1.0

    # 3. SURVIVAL GENETICS (Ranges & Chances)
    MIN_CHRONOTYPE: float = 0.1
    MAX_CHRONOTYPE: float = 1.0
    MIN_PLANT_DIGEST: float = 0.1
    MAX_PLANT_DIGEST: float = 1.0
    MIN_MEAT_DIGEST: float = 0.1
    MAX_MEAT_DIGEST: float = 1.0
    
    VENOM_CHANCE: float = 0.1
    MIMICRY_CHANCE: float = 0.1

    MIN_PREGNANCY_ENERGY: float = 0.1
    MIN_PREGNANCY_HEALTH: float = 0.1

    # 4. BEHAVIORAL GENETICS (Ranges)
    MIN_KINSHIP: float = 0.1
    MAX_KINSHIP: float = 1.0
    MIN_PARENTAL: float = 0.1
    MAX_PARENTAL: float = 1.0
    MIN_HOSTILITY: float = 0.1
    MAX_HOSTILITY: float = 1.0
    MIN_FIGHT_FLIGHT: float = 0.1
    MAX_FIGHT_FLIGHT: float = 1.0

    # 5. METABOLISM & MULTIPLIERS
    BIRTH_ENERGY_FACTOR: float = 0.5
    METABOLISM_INFECTED_MULT: float = 1.25
    METABOLISM_BLEEDING_MULT: float = 1.25
    METABOLISM_SLEEPING_MULT: float = 0.2
    METABOLISM_PREGNANT_MULT: float = 1.5
    METABOLISM_HIBERNATING_MULT: float = 0.05
    METABOLISM_TEMP_HARSHNESS: float = 0.1

    # 6. COSTS & PENALTIES
    MOVE_ENERGY_COST: int = 5
    DISEASE_HEALTH_COST: int = 5
    BLEEDING_HEALTH_COST: int = 10

    # 7. TIMERS & CAPACITIES
    GESTATION_TIME: int = 250
    MAX_WASTE_CAPACITY: int = 10
    WASTE_PORTION: float = 0.25

    def __post_init__(self) -> None:
        """
        Validates all configuration parameters and populates derived fields.
        """
        
        # --- 1. VALIDATION ---

        # A. SIMULATION & MATRIX CONSTANTS Validation 
        if not isinstance(self.EMPTY_MATRIX_ID, int) or not isinstance(self.AGENT_MATRIX_ID, int):
            raise TypeError("Matrix IDs must be integers.")
        if self.EMPTY_MATRIX_ID == self.AGENT_MATRIX_ID:
            raise ValueError(f"EMPTY_MATRIX_ID and AGENT_MATRIX_ID cannot be the same ({self.EMPTY_MATRIX_ID}).")

        if not isinstance(self.MOVEMENT_DIRECTIONS, np.ndarray):
            raise TypeError("MOVEMENT_DIRECTIONS must be a NumPy ndarray.")
        if self.MOVEMENT_DIRECTIONS.shape != (4, 2):
            raise ValueError(f"MOVEMENT_DIRECTIONS must have shape (4, 2), got {self.MOVEMENT_DIRECTIONS.shape}.")

        # B. PHYSICAL GENETICS Validation 
        int_phys_ranges = [
            ("SIZE", self.MIN_SIZE, self.MAX_SIZE),
            ("AGE", self.MIN_AGE, self.MAX_AGE),
            ("HEALTH", self.MIN_HEALTH, self.MAX_HEALTH),
            ("ENERGY", self.MIN_ENERGY, self.MAX_ENERGY),
            ("TEMP", self.MIN_TEMP, self.MAX_TEMP),
        ]
        for name, min_val, max_val in int_phys_ranges:
            if not isinstance(min_val, int) or not isinstance(max_val, int):
                raise TypeError(f"MIN_{name} and MAX_{name} must be integers.")
            if min_val < 0:
                raise ValueError(f"MIN_{name} must be non-negative.")
            if max_val < min_val:
                raise ValueError(f"MAX_{name} ({max_val}) cannot be smaller than MIN_{name} ({min_val}).")

        float_phys_ranges = [
            ("ENERGY_BURN", self.MIN_ENERGY_BURN, self.MAX_ENERGY_BURN),
            ("RESISTANCE", self.MIN_RESISTANCE, self.MAX_RESISTANCE),
        ]
        for name, min_val, max_val in float_phys_ranges:
            if not isinstance(min_val, (float, int)) or not isinstance(max_val, (float, int)):
                raise TypeError(f"MIN_{name} and MAX_{name} must be numbers.")
            if max_val < min_val:
                raise ValueError(f"MAX_{name} ({max_val}) cannot be smaller than MIN_{name} ({min_val}).")

        # C. SURVIVAL GENETICS Validation 
        float_surv_ranges = [
            ("CHRONOTYPE", self.MIN_CHRONOTYPE, self.MAX_CHRONOTYPE),
            ("PLANT_DIGEST", self.MIN_PLANT_DIGEST, self.MAX_PLANT_DIGEST),
            ("MEAT_DIGEST", self.MIN_MEAT_DIGEST, self.MAX_MEAT_DIGEST),
        ]
        for name, min_val, max_val in float_surv_ranges:
            if not isinstance(min_val, (float, int)) or not isinstance(max_val, (float, int)):
                raise TypeError(f"MIN_{name} and MAX_{name} must be numbers.")
            if max_val < min_val:
                raise ValueError(f"MAX_{name} ({max_val}) cannot be smaller than MIN_{name} ({min_val}).")

        probs = [("VENOM_CHANCE", self.VENOM_CHANCE), ("MIMICRY_CHANCE", self.MIMICRY_CHANCE)]
        for name, val in probs:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be between 0.0 and 1.0.")

        # D. BEHAVIORAL GENETICS Validation 
        float_behav_ranges = [
            ("KINSHIP", self.MIN_KINSHIP, self.MAX_KINSHIP),
            ("PARENTAL", self.MIN_PARENTAL, self.MAX_PARENTAL),
            ("HOSTILITY", self.MIN_HOSTILITY, self.MAX_HOSTILITY),
            ("FIGHT_FLIGHT", self.MIN_FIGHT_FLIGHT, self.MAX_FIGHT_FLIGHT),
        ]
        for name, min_val, max_val in float_behav_ranges:
            if not isinstance(min_val, (float, int)) or not isinstance(max_val, (float, int)):
                raise TypeError(f"MIN_{name} and MAX_{name} must be numbers.")
            if max_val < min_val:
                raise ValueError(f"MAX_{name} ({max_val}) cannot be smaller than MIN_{name} ({min_val}).")

        # E. METABOLISM & MULTIPLIERS Validation 
        multipliers = [
            ("BIRTH_ENERGY_FACTOR", self.BIRTH_ENERGY_FACTOR),
            ("METABOLISM_INFECTED_MULT", self.METABOLISM_INFECTED_MULT),
            ("METABOLISM_BLEEDING_MULT", self.METABOLISM_BLEEDING_MULT),
            ("METABOLISM_SLEEPING_MULT", self.METABOLISM_SLEEPING_MULT),
            ("METABOLISM_PREGNANT_MULT", self.METABOLISM_PREGNANT_MULT),
            ("METABOLISM_HIBERNATING_MULT", self.METABOLISM_HIBERNATING_MULT),
            ("METABOLISM_TEMP_HARSHNESS", self.METABOLISM_TEMP_HARSHNESS),
        ]
        for name, val in multipliers:
            if val < 0:
                raise ValueError(f"{name} must be non-negative.")

        # F. COSTS & PENALTIES Validation 
        costs = [
            ("MOVE_ENERGY_COST", self.MOVE_ENERGY_COST),
            ("DISEASE_HEALTH_COST", self.DISEASE_HEALTH_COST),
            ("BLEEDING_HEALTH_COST", self.BLEEDING_HEALTH_COST),
        ]
        for name, val in costs:
            if not isinstance(val, int) or val < 0:
                raise ValueError(f"{name} must be a non-negative integer.")

        # G. TIMERS & CAPACITIES Validation 
        timers = [
            ("GESTATION_TIME", self.GESTATION_TIME),
            ("MAX_WASTE_CAPACITY", self.MAX_WASTE_CAPACITY),
        ]
        for name, val in timers:
            if not isinstance(val, int) or val < 0:
                raise ValueError(f"{name} must be a non-negative integer.")

        # --- 2. POPULATE DERIVED FIELDS ---
        # Using object.__setattr__ because the dataclass is frozen.
        
        # Split Vision Template into X and Y components
        object.__setattr__(self, "VISION_TEMPLATE_X", self.VISION_TEMPLATE[:, 0])
        object.__setattr__(self, "VISION_TEMPLATE_Y", self.VISION_TEMPLATE[:, 1])

        # Identify indices that constitute the immediate neighborhood (3x3 area excluding center).
        neighbour_mask = (
            (np.abs(self.VISION_TEMPLATE_X) <= 1) &
            (np.abs(self.VISION_TEMPLATE_Y) <= 1) &
            ~((self.VISION_TEMPLATE_X == 0) & (self.VISION_TEMPLATE_Y == 0))
        )

        object.__setattr__(
            self,
            "NEIGHBOUR_INDICES",
            np.where(neighbour_mask)[0]
        )

class AgentManager:

    def __init__(
            self,
            config_agent_manager: ConfigAgentManager,
            resource_manager: ResourceManager,
            simulation_size: int
    ) -> None:
        # 1. Defines input parameters.
        self.config_agent_manager: ConfigAgentManager = config_agent_manager
        self.resource_manager: ResourceManager = resource_manager
        self.simulation_size: int = simulation_size

        # 2. Creates agent matrix.
        self.agent_matrix: NDArray = np.zeros(
            shape = (self.simulation_size, self.simulation_size),
            dtype = np.float16
        )

        # 3. Creates a agent DataFrame.
        self.agents_df: DataFrame = self._create_empty_agent_df()

    def _create_empty_agent_df(self) -> DataFrame:   
            agent_df_schema: dict = {
                # Identity.
                "clan_id": pl.UInt8,
                "sex": pl.UInt8,
                
                # Physical Genetics.
                "size": pl.UInt8,
                "max_age": pl.UInt8,
                "max_health": pl.UInt8,
                "max_energy": pl.UInt8,
                "optimal_temperature": pl.UInt8,
                "resistance": pl.Float32, # Base genetic resistance.
                
                # Metabolic Genetics.
                "base_energy_burn_rate": pl.Float32, # Use instead of Float16.
                "chronotype": pl.Float32,
                "plant_digestion_efficiency": pl.Float32,
                "meat_digestion_efficiency": pl.Float32,

                # Special Traits.
                "has_venom": pl.Boolean, 
                "has_mimicry": pl.Boolean,

                # Behavioral Genetics.
                "kinship_loyalty": pl.Float32,
                "parental_instinct": pl.Float32,
                "hostility": pl.Float32,
                "fight_flight_bias": pl.Float32,

                # Current Vitals.
                "age": pl.UInt8,
                "health": pl.Float32,
                "energy": pl.Float32,
                "waste_accumulated": pl.Float32,

                # Calculated Rates & Modifiers. #TODO Delete these columns from the dataframe.
                "total_energy_burn_rate": pl.Float32,
                "effective_resistance": pl.Float32, # Base resistance +/- modifiers.
                "metabolism_modifier": pl.Float32,
                "health_cost": pl.Float32,

                # Boolean Status Flags.
                "is_infected": pl.Boolean,
                "is_bleeding": pl.Boolean,
                "is_sleeping": pl.Boolean,
                "is_pregnant": pl.Boolean,
                "is_hibernating": pl.Boolean,

                # Timers.
                "hibernation_timer": pl.UInt8,
                "gestation_timer": pl.UInt8,

                # Positional States.
                "x_pos": pl.Int16,
                "y_pos": pl.Int16,
                "new_x_pos": pl.Int16, #TODO Delete these columns from the dataframe
                "new_y_pos": pl.Int16,
                "valid_move": pl.Boolean,

                # Visual Genetics/State.
                "color_r": pl.UInt8,
                "color_g": pl.UInt8,
                "color_b": pl.UInt8,
            }

            return pl.DataFrame(schema=agent_df_schema) # Creates a dataframe from the schema dictionary.

    def create_agents(
            self,
            old_df: DataFrame,
            count: int,
            x_positions: NDArray | None = None,
            y_positions: NDArray | None = None
    ) -> DataFrame:
        #TODO WOO: The order of the attributes in the dictionary.
        # 1. Creates a random number generator instance.
        rng = np.random.default_rng()

        # 2. Generates all random data in a single dictionary definition.
        agent_data: dict[str, NDArray] = {
            # Identity.
            "clan_id": rng.integers(low=0, high=0 + 1, size=count, dtype=np.uint8), #TODO Add Clan ID mechanism.
            "sex": rng.integers(low=0, high=2 + 1, size=count, dtype=np.uint8), #TODO ACV

            # Physical Genetics.
            "size": rng.integers(low=self.config_agent_manager.MIN_SIZE, high=self.config_agent_manager.MAX_SIZE + 1, size=count, dtype=np.uint8),
            "max_age": rng.integers(low=self.config_agent_manager.MIN_AGE, high=self.config_agent_manager.MAX_AGE + 1, size=count, dtype=np.uint8),
            "max_health": rng.integers(low=self.config_agent_manager.MIN_HEALTH, high=self.config_agent_manager.MAX_HEALTH + 1, size=count, dtype=np.uint8),
            "max_energy": rng.integers(low=self.config_agent_manager.MIN_ENERGY, high=self.config_agent_manager.MAX_ENERGY + 1, size=count, dtype=np.uint8),
            "optimal_temperature": rng.integers(low=self.config_agent_manager.MIN_TEMP, high=self.config_agent_manager.MAX_TEMP + 1, size=count, dtype=np.uint8),
            "resistance": rng.uniform(low=self.config_agent_manager.MIN_RESISTANCE, high=self.config_agent_manager.MAX_RESISTANCE, size=count).astype(np.float32),

            # Metabolic Genetics.
            "base_energy_burn_rate": rng.uniform(low=self.config_agent_manager.MIN_ENERGY_BURN, high=self.config_agent_manager.MAX_ENERGY_BURN + 1, size=count).astype(np.float32),
            "chronotype": rng.uniform(low=self.config_agent_manager.MIN_CHRONOTYPE, high=self.config_agent_manager.MAX_CHRONOTYPE, size=count).astype(np.float32),
            "plant_digestion_efficiency": rng.uniform(low=self.config_agent_manager.MIN_PLANT_DIGEST, high=self.config_agent_manager.MAX_PLANT_DIGEST, size=count).astype(np.float32),
            "meat_digestion_efficiency": rng.uniform(low=self.config_agent_manager.MIN_MEAT_DIGEST, high=self.config_agent_manager.MAX_MEAT_DIGEST, size=count).astype(np.float32),

            # Special Traits.
            "has_venom": rng.uniform(low=0, high=1, size=count).astype(np.float32) < self.config_agent_manager.VENOM_CHANCE, # Creates a boolean array according to the condition. #TODO Check whether the dtype matches.
            "has_mimicry": rng.uniform(low=0, high=1, size=count).astype(np.float32) < self.config_agent_manager.MIMICRY_CHANCE,

            # Behavioral Genetics.
            "kinship_loyalty": rng.uniform(low=self.config_agent_manager.MIN_KINSHIP, high=self.config_agent_manager.MAX_KINSHIP, size=count).astype(np.float32),
            "parental_instinct": rng.uniform(low=self.config_agent_manager.MIN_PARENTAL, high=self.config_agent_manager.MAX_PARENTAL, size=count).astype(np.float32),
            "hostility": rng.uniform(low=self.config_agent_manager.MIN_HOSTILITY, high=self.config_agent_manager.MAX_HOSTILITY, size=count).astype(np.float32),
            "fight_flight_bias": rng.uniform(low=self.config_agent_manager.MIN_FIGHT_FLIGHT, high=self.config_agent_manager.MAX_FIGHT_FLIGHT, size=count).astype(np.float32),

            # Current Vitals.
            "age": np.zeros(shape=count, dtype=np.uint8),
            "waste_accumulated": np.zeros(shape=count, dtype=np.float32),

            # Calculated Rates & Modifiers.
            "metabolism_modifier": np.ones(shape=count, dtype=np.float32),
            "health_cost": np.ones(shape=count, dtype=np.float32),

            # Boolean Status Flags.
            "is_infected": np.full(shape=count, fill_value=False, dtype=np.bool_),
            "is_bleeding": np.full(shape=count, fill_value=False, dtype=np.bool_),
            "is_sleeping": np.full(shape=count, fill_value=False, dtype=np.bool_),
            "is_pregnant": np.full(shape=count, fill_value=False, dtype=np.bool_),
            "is_hibernating": np.full(shape=count, fill_value=False, dtype=np.bool_),

            # Timers.
            "hibernation_timer": np.zeros(shape=count, dtype=np.uint8),
            "gestation_timer": np.zeros(shape=count, dtype=np.uint8),

            # Positional States.
            "valid_move": np.full(shape=count, fill_value=False, dtype=np.bool_)
        }

        # 3. Creates the DataFrame.
        new_agents_df = DataFrame(data=agent_data)

        # 4. Adds new columns.
        new_agents_df = new_agents_df.with_columns(
            # Creates columns for the attributes, that have starting values.
            (pl.col("max_health").cast(pl.Float32).alias("health")), # Gets the reference to the column.
            (pl.col("max_energy") * self.config_agent_manager.BIRTH_ENERGY_FACTOR).cast(pl.Float32).alias("energy"),  # Name of the added attribute.
            (pl.col("resistance").cast(pl.Float32).alias("effective_resistance")),
            (pl.col("base_energy_burn_rate").cast(pl.Float32).alias("total_energy_burn_rate")),
            # Adds derived color columns
            (pl.col("meat_digestion_efficiency") * 255).cast(pl.UInt8).alias("color_r"),
            (pl.col("plant_digestion_efficiency") * 255).cast(pl.UInt8).alias("color_g"),
            (pl.col("size") * 85).cast(pl.UInt8).alias("color_b") # (Size 1, 2, 3 -> 85, 170, 255).
        )

        # 6. Handles the placement in the agent matrix.
        if (x_positions is None) or (y_positions is None): # If the positions are not passed as argument to the method.
            # Finds the flat indices of zero elements.
            zero_indices_flat: NDArray = np.flatnonzero(a=(self.agent_matrix == 0))

            # Checks whether there are enough available spots.
            if count > len(zero_indices_flat):
                raise ValueError(
                    f"Can't create {count} agents. Only {len(zero_indices_flat)} empty spots are available!"
                )

            # Randomly chooses from these indices.
            random_indices_flat: NDArray = np.random.choice(
                a = zero_indices_flat,
                size = count,
                replace = False
            )

            # Assigns the choosen indices and x- and y positions.
            choosen_indices_flat: NDArray = random_indices_flat

            x_positions: NDArray = choosen_indices_flat % self.simulation_size
            y_positions: NDArray = choosen_indices_flat // self.simulation_size

        else: 
            # Calculates the choosen indices from the to the method passed positions.
            choosen_indices_flat: NDArray = y_positions * self.simulation_size + x_positions

        # Updates the agent matrix.
        np.put(
            a = self.agent_matrix,
            ind = choosen_indices_flat,
            v = self.config_agent_manager.AGENT_MATRIX_ID
        )

        # Adds position columns to the DataFrame.
        new_agents_df = new_agents_df.with_columns(
            pl.lit(x_positions).cast(pl.Int16).alias("x_pos"), # 'lit' converts an array into polars column.
            pl.lit(y_positions).cast(pl.Int16).alias("y_pos"),
            pl.lit(x_positions).cast(pl.Int16).alias("new_x_pos"),
            pl.lit(y_positions).cast(pl.Int16).alias("new_y_pos")
        )

        # 7. Updates the agent DataFrame.
        # Forces the new data frame to have the same column order as the schema.
        new_agents_df = new_agents_df.select(self.agents_df.columns)

        # Concates them together.
        new_agents_df = pl.concat([old_df, new_agents_df])

        return new_agents_df

    def _extract_vision_grid(
            self,
            old_df: DataFrame
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        # 1. Creates all the grids, that cells see.
        # Extracts the all cell positions.
        cells_x: NDArray = old_df["x_pos"].to_numpy().astype(dtype=np.int16)
        cells_y: NDArray = old_df["y_pos"].to_numpy().astype(dtype=np.int16)

        # Adds all the vision positions to the agent positions.
        vision_grids_x: NDArray = cells_x[:, None] + self.config_agent_manager.VISION_TEMPLATE_X
        vision_grids_y: NDArray = cells_y[:, None] + self.config_agent_manager.VISION_TEMPLATE_Y

        # 2. Creates a boolean mask to filter the valid positions.
        in_bounds_mask: NDArray = (
            (vision_grids_x >= 0) &
            (vision_grids_x < self.simulation_size) &
            (vision_grids_y >= 0) &
            (vision_grids_y < self.simulation_size)
        )

        # Filters inbound positions.
        inbound_x: NDArray = np.where(in_bounds_mask, vision_grids_x, 0) 
        inbound_y: NDArray = np.where(in_bounds_mask, vision_grids_y, 0)

        # 3. Finds out, whether there are any cells or resources on the vision grids.
        vision_cells: NDArray = self.agent_matrix[inbound_y, inbound_x]
        
        vision_corpses: NDArray = self.resource_manager.corpse_matrix[inbound_y, inbound_x]
        vision_foods: NDArray = self.resource_manager.food_matrix[inbound_y, inbound_x]
        vision_wastes: NDArray = self.resource_manager.waste_matrix[inbound_y, inbound_x]
        vision_roots: NDArray = self.resource_manager.root_matrix[inbound_y, inbound_x]

        # 4. Whereever the mask is false, a.k.a. a border is met, turn the value into '-1'.
        vision_cells[~in_bounds_mask] = -1

        vision_corpses[~in_bounds_mask] = -1
        vision_foods[~in_bounds_mask] = -1
        vision_wastes[~in_bounds_mask] = -1
        vision_roots[~in_bounds_mask] = -1

        return vision_cells, vision_corpses, vision_foods, vision_wastes, vision_roots

    def _process_death(
            self,
            old_df: DataFrame
    ) -> DataFrame:
        cfg: ConfigAgentManager = self.config_agent_manager

        # 1. Extracts the agents, that needs to die.
        death_mask: DataFrame = (pl.col("energy") <= 0) | (pl.col("health") <= 0)

        dead_agents: DataFrame = old_df.filter(death_mask)

        # Extracts the positions.
        new_corpse_x_positions: NDArray = dead_agents["x_pos"].to_numpy().astype(dtype=np.int16)
        new_corpse_y_positions: NDArray = dead_agents["y_pos"].to_numpy().astype(dtype=np.int16)

        # 2. Updates the corpse matrix with the new corpses.
        self.resource_manager.corpse_matrix[new_corpse_y_positions, new_corpse_x_positions] += 1.0

        # 3. Deletes the agents that died from the dataframe and agent matrix.
        new_df: DataFrame = old_df.filter(~death_mask) # '~' inverts the variable.

        self.agent_matrix[new_corpse_y_positions, new_corpse_x_positions] = cfg.EMPTY_MATRIX_ID

        return new_df

    def _process_metabolism(
            self,
            old_df: DataFrame,
            map_temperature: int
    ) -> DataFrame:
        cfg: ConfigAgentManager = self.config_agent_manager

        # 1. Calculates the metabolism modifier.
        new_df = old_df.with_columns(
            (
                pl.lit(1.0) # Starts with a base metabolism multiplier of 1.0.
                * pl.when(pl.col("is_infected"))
                .then(cfg.METABOLISM_INFECTED_MULT)
                .otherwise(1.0)
                * pl.when(pl.col("is_bleeding"))
                .then(cfg.METABOLISM_BLEEDING_MULT)
                .otherwise(1.0)
                * pl.when(pl.col("is_sleeping"))
                .then(cfg.METABOLISM_SLEEPING_MULT)
                .otherwise(1.0)
                * pl.when(pl.col("is_pregnant"))
                .then(cfg.METABOLISM_PREGNANT_MULT)
                .otherwise(1.0)
                * pl.when(pl.col("is_hibernating"))
                .then(cfg.METABOLISM_HIBERNATING_MULT)
                .otherwise(1.0)
            )
            .cast(pl.Float32)
            .alias("metabolism_modifier")
        )

        # 2. Calculates the total burn rate depending on base burn rate, metabolism modifier, size and temperature and deducts it from the current energy.
        new_df = new_df.with_columns(
            (pl.col("base_energy_burn_rate") * pl.col("metabolism_modifier") * pl.col("size")) + ((map_temperature - pl.col("optimal_temperature").abs()) * cfg.METABOLISM_TEMP_HARSHNESS * (pl.lit(1.0) - pl.col("effective_resistance")))
            .cast(pl.Float32)
            .alias("total_energy_burn_rate")
        )

        new_df = new_df.with_columns(
            (pl.col("energy") - pl.col("total_energy_burn_rate"))
            .cast(pl.Float32)
            .alias("energy")
        )

        # 3. Adds a portion of total energy burn rate to the waste accumulation.
        new_df = new_df.with_columns(
            (pl.col("waste_accumulated") + pl.col("total_energy_burn_rate") * cfg.WASTE_PORTION)
            .cast(pl.Float32)
            .alias("waste_accumulated")
        )

        return new_df

    def _process_vital_states(
            self,
            old_df: DataFrame
    ) -> DataFrame:
        cfg: ConfigAgentManager = self.config_agent_manager

        # 1. Calculates the health cost.
        new_df = old_df.with_columns(
            (
                pl.lit(0) # Starts with base health cost of 0.
                + pl.when(pl.col("is_infected"))
                .then(cfg.DISEASE_HEALTH_COST)
                .otherwise(0)
                + pl.when(pl.col("is_bleeding"))
                .then(cfg.BLEEDING_HEALTH_COST)
                .otherwise(0)
            )
            .cast(pl.Float32)
            .alias("health_cost")
        )

        # 2. Deducts the health cost from the current health.
        new_df = new_df.with_columns(
            (pl.col("health") - pl.col("health_cost"))
            .cast(pl.Float32)
            .alias("health")
        )

        return new_df

    def _find_neighbours(
            self,
            mother_visions: NDArray,
            mother_x: NDArray,
            mother_y: NDArray
    ) -> tuple[NDArray, NDArray, NDArray]:
        
        # Abbreviations:
        # number of mothers = M.
        # count of succesful births = S.
    
        cfg = self.config_agent_manager
        rng = np.random.default_rng()

        # 1. Looks only at the 8 neighbor columns within the vision slice.
        neighbour_cells: NDArray = mother_visions[:, cfg.NEIGHBOUR_INDICES] # (M, 8).

        # 2. Creates a mask of empty spots.
        empty_mask: NDArray = (neighbour_cells == cfg.EMPTY_MATRIX_ID) # (M, 8).

        # 3. Checks which mothers have at least one empty neighbour.
        has_spot_mask: NDArray = np.any(empty_mask, axis=1) # (M,)

        if not np.any(has_spot_mask):
            return (
                np.array([], dtype=np.int16),
                np.array([], dtype=np.int16),
                has_spot_mask
            )
        
        # 4. Random cell selection.
        # Assigns a random value to every empty spot and 0 to occupied spots.
        random_scores: NDArray = rng.random(empty_mask.shape) * empty_mask # (M, 8).
        # Take the argmax to find a random empty spot.
        chosen_indices: NDArray = np.argmax(random_scores, axis=1) # Finds the index of the highest value in each row (axis=1) and (M,). 

        # 5. Maps indices (0-7) indices back to VISION_TEMPLATE indices.
        chosen_template_indices: NDArray = cfg.NEIGHBOUR_INDICES[chosen_indices] # (M,).

        # 6. Calculates coordinates for the babies.
        baby_x: NDArray = mother_x[has_spot_mask] + cfg.VISION_TEMPLATE_X[chosen_template_indices[has_spot_mask]] # (S,).
        baby_y: NDArray = mother_y[has_spot_mask] + cfg.VISION_TEMPLATE_Y[chosen_template_indices[has_spot_mask]] # (S,).

        return (baby_x.astype(np.int16), baby_y.astype(np.int16), has_spot_mask)

    def _attempt_reproduction(
                self,
                old_df: DataFrame,
                cell_visions: NDArray
        ) -> DataFrame:
            cfg: ConfigAgentManager = self.config_agent_manager

            # 1. Filters the cells, that has the correct attributes to be mothers.
            is_possible_mother_mask: NDArray = (
                    (old_df["sex"] == 1) & # If female.
                    (old_df["is_pregnant"] == False) & # If not already pregnant.
                    (old_df["age"] > (old_df["max_age"] // 2)) & # If reached maturity.
                    (old_df["energy"] > (old_df["max_energy"] * cfg.MIN_PREGNANCY_ENERGY)) & # If has enough energy.
                    (old_df["health"] > (old_df["max_health"] * cfg.MIN_PREGNANCY_HEALTH)) # If has enough health.
            )

            # 2. Filters the cells, which have neighbours.
            neighbour_counts: NDArray = cell_visions.sum(
                axis = 1, # Horizontal summation.
                dtype = np.uint8
            )

            has_neighbour_mask: NDArray = pl.Series(neighbour_counts > 1) # Converts the array mask into dataframe mask.

            # 3. Final lists of agents, which can succesfully reproduce.
            final_mask: NDArray = is_possible_mother_mask & has_neighbour_mask

            # 4. Updates the old data frame with the new pregnant cells.
            new_df: DataFrame = old_df.with_columns(
                pl.when(final_mask)
                .then(pl.lit(True))
                .otherwise(pl.col("is_pregnant"))
                .cast(pl.Boolean)
                .alias("is_pregnant")
            )

            return new_df

    def _update_pregnancy_status(
            self,
            old_df: DataFrame
    ) -> DataFrame:

        # 1. Checks for miscarriage.
        new_df = old_df.with_columns(
            pl.when(
                (pl.col("is_pregnant")) & 
                (
                    (pl.col("energy") < (pl.col("max_energy") * self.config_agent_manager.MIN_PREGNANCY_ENERGY)) |
                    (pl.col("health") < (pl.col("max_health") * self.config_agent_manager.MIN_PREGNANCY_HEALTH))
                )
            )
            .then(pl.lit(False))
            .otherwise(pl.col("is_pregnant"))
            .cast(pl.Boolean)
            .alias("is_pregnant")
        )

        # 2. Updates the gestation timer.
        new_df: DataFrame = new_df.with_columns(
            (
                pl.when(pl.col("is_pregnant"))
                .then(pl.col("gestation_timer") + 1)
                .otherwise(pl.col("gestation_timer"))
            )
            .cast(pl.UInt8)
            .alias("gestation_timer")
        )

        return new_df

    def _give_birth(
            self,
            old_df: DataFrame,
            cell_visions: NDArray
    ) -> DataFrame:
        cfg = self.config_agent_manager

        # 1. Identifies mothers at the end of gestation.
        is_ready_mask: DataFrame = (old_df["gestation_timer"] >= cfg.GESTATION_TIME)
        ready_indices: NDArray = np.where(is_ready_mask.to_numpy())[0] #TODO AC

        if ready_indices.size == 0:
            return old_df
        
        # 2. Extracts their vision and positions.
        mother_visions: NDArray = cell_visions[ready_indices]
        mother_x: NDArray = old_df.filter(is_ready_mask)["x_pos"].to_numpy()
        mother_y: NDArray = old_df.filter(is_ready_mask)["y_pos"].to_numpy()

        # 3. Finds birth spots.
        baby_x, baby_y, success_mask = self._find_neighbours(
            mother_visions = mother_visions,
            mother_x = mother_x,
            mother_y = mother_y
        )

        # 4. Resets pregnancy for all mothers who reached term.
        new_df: DataFrame = old_df.with_columns(
            pl.when(is_ready_mask)
            .then(pl.lit(0))
            .otherwise(pl.col("gestation_timer"))
            .alias("gestation_timer"),

            pl.when(is_ready_mask)
            .then(pl.lit(False))
            .otherwise(pl.col("is_pregnant"))
            .alias("is_pregnant")
        )

        # 5. Creates the new babies.
        if baby_x.size > 0:
            new_df = self.create_agents(
                old_df = new_df,
                count = baby_x.size,
                x_positions = baby_x,
                y_positions = baby_y
            )

        return new_df

    def _process_gestation(
                self,
                old_df: DataFrame,
                cell_visions: NDArray
    ) -> DataFrame:
        cfg: ConfigAgentManager = self.config_agent_manager

        # 1. Attempts to reproduce.
        new_df: DataFrame = self._attempt_reproduction(
            old_df = old_df,
            cell_visions = cell_visions
        )

        # 2. Updates the pregnancy status.
        new_df = self._update_pregnancy_status(old_df=new_df)

        # 3. Gives birth.
        new_df = self._give_birth(
            old_df = new_df,
            cell_visions = cell_visions
        )

        return new_df

    def _process_waste_accumulation(
            self,
            old_df: DataFrame,
    ) -> DataFrame:
        cfg: ConfigAgentManager = self.config_agent_manager

        # 1. Extracts the agents, that need to remove their wastes.
        waste_full_agents: DataFrame = old_df.filter(pl.col("waste_accumulated") >= cfg.MAX_WASTE_CAPACITY)
 
        # Extracts the positions.
        new_waste_x_positions: NDArray = waste_full_agents["x_pos"].to_numpy().astype(dtype=np.int16)
        new_waste_y_positions: NDArray = waste_full_agents["y_pos"].to_numpy().astype(dtype=np.int16)

        # 2. Updates the waste matrix with the new wastes.
        self.resource_manager.waste_matrix[new_waste_y_positions, new_waste_x_positions] += 1.0

        # 3. Updates the dataframe column waste accumulated.
        new_df: DataFrame = old_df.with_columns(
            (
                pl.when(pl.col("waste_accumulated") >= cfg.MAX_WASTE_CAPACITY)
                .then(0.0)
                .otherwise(pl.col("waste_accumulated"))
            )
            .cast(pl.Float32)
            .alias("waste_accumulated")
        )

        return new_df

    def _move_agents(
            self,
            old_df: DataFrame,
            direction_indices: int # (0: Up, 1: Right, 2: Down, 3: Left).
    ) -> tuple[NDArray, DataFrame]:
        # 1. Calculates the target based on the direction.
        directions = self.config_agent_manager.MOVEMENT_DIRECTIONS[direction_indices]

        # Seperates the dx and dy vectors.
        dx: NDArray = directions[:, 0]
        dy: NDArray = directions[:, 1]

        new_df: DataFrame = old_df.with_columns(
            (pl.col("x_pos") + dx).alias("new_x_pos"),
            (pl.col("y_pos") + dy).alias("new_y_pos")
        )

        # 2. Checks for out of bound and updates the columns.
        new_df = new_df.with_columns(
            (
                (pl.col("new_x_pos") >= 0) &
                (pl.col("new_x_pos") < self.simulation_size) &
                (pl.col("new_y_pos") >= 0) & 
                (pl.col("new_y_pos") < self.simulation_size)
            ).alias("valid_move")
        )

        # 3. Checks for colision.
        # Extract the polars columns as numpy arrays.
        potential_y: NDArray = new_df["new_y_pos"].to_numpy().astype(dtype=np.int16) # For safe map size.
        potential_x: NDArray = new_df["new_x_pos"].to_numpy().astype(dtype=np.int16)

        # Clips the coordinates to safe range (0 to size-1).
        safe_y: NDArray = np.clip(potential_y, 0, self.simulation_size - 1)
        safe_x: NDArray = np.clip(potential_x, 0, self.simulation_size - 1)

        # Checks the matrix using the safe coordinates.
        is_empty_spot: NDArray = (self.agent_matrix[safe_y, safe_x] == self.config_agent_manager.EMPTY_MATRIX_ID)

        # 4. Updates the valid moves.
        new_df = new_df.with_columns(
            (pl.col("valid_move") & is_empty_spot).alias("valid_move")
        )

        # 5. Deletes the old position and updates the new.
        # Extract the polar columns as numpy arrays.
        old_pos_y: NDArray = new_df["y_pos"].to_numpy().astype(dtype=np.int16)
        old_pos_x: NDArray = new_df["x_pos"].to_numpy().astype(dtype=np.int16)

        valid_move: NDArray = new_df["valid_move"].to_numpy().astype(dtype=np.bool_)

        # Deletes the old positions.
        self.agent_matrix[old_pos_y, old_pos_x] = self.config_agent_manager.EMPTY_MATRIX_ID
        
        # Updates the new columns.
        new_df = new_df.with_columns(
            pl.when(pl.col("valid_move"))
            .then(
                pl.col("new_x_pos").alias("x_pos"),
            )
            .otherwise(
                pl.col("x_pos").alias("x_pos"),
            )
        )

        new_df = new_df.with_columns(
            pl.when(pl.col("valid_move"))
            .then(
                pl.col("new_y_pos").alias("y_pos")
            )
            .otherwise(
                pl.col("y_pos").alias("y_pos")
            )
        )

        # Extracts the newly updated y- and x-positions.
        collision_updated_y_pos: NDArray = new_df["y_pos"].to_numpy().astype(dtype=np.int16)
        collision_updated_x_pos: NDArray = new_df["x_pos"].to_numpy().astype(dtype=np.int16)

        # Updates the matrices.
        self.agent_matrix[collision_updated_y_pos, collision_updated_x_pos] = self.config_agent_manager.AGENT_MATRIX_ID

        return (valid_move, new_df)
    
    def agent_loop(
            self,
            map_temperature: int,
    ) -> None:
        df: DataFrame = self.agents_df
        cfg: ConfigAgentManager = self.config_agent_manager

        # A. MANDATORY ACTIONS.
        # 1. Cheks whether the agent dies.
        df = self._process_death(old_df=df)

        # 2. Updates the agents age.
        df = df.with_columns(
            (pl.col("age") + 1)
            .alias("age")
        )

        # 3. Calculates the effective resistance.
        df = df.with_columns(
            pl.when((pl.col("age")) > (pl.col("max_age") // 2)) # If the age threshold is met.
            .then((1.0 - ((pl.col("age") - pl.col("max_age") // 2) / (pl.col("max_age") - pl.col("max_age") // 2)) ** 2).clip(lower_bound=0.1))
            .otherwise(1.0)
            .cast(pl.Float32)
            .alias("effective_resistance")
        )

        # 4. Processes the metabolism.
        df = self._process_metabolism(
            old_df = df,
            map_temperature = map_temperature
        )

        # 5. Handles the disease, bleeding, gestation and waste accumulation states.
        df = self._process_vital_states(old_df=df)

        # 6. Calculates the observeble vision grids.
        cell_visions, corpse_visions, food_visions, waste_visions, root_visions = self._extract_vision_grid(old_df = df)

        # 7. Handles the gestation.
        df = self._process_gestation(
            old_df = df,
            cell_visions = cell_visions
        )

        # 8. Handles the waste accumulation.
        df = self._process_waste_accumulation(old_df=df)

        # 9. Updates the agents dataframe.
        self.agents_df = df

        # B. TAKEN ACTIONS.

        # 1. Moves the agents.
        #! TESTING
        rng = np.random.default_rng()
        direction_indices: NDArray = rng.integers(low=0, high=3+1, size=df["sex"].len())
        valid_move, df = self._move_agents(
            old_df = df,
            direction_indices = direction_indices
        )

        self.agents_df = df

    def get_agent_dataframe(self) -> DataFrame:
        return self.agents_df
