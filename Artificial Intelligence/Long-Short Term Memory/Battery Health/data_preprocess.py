
# data_preprocess.py

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
import os

import numpy as np
from numpy.typing import NDArray
import scipy.io
import torch
from torch import Tensor
from torch.utils.data import (
    DataLoader,
    TensorDataset
)
from typing import Any

@dataclass
class ConfigDataManager:
    """
    Configuration dataclass for the Battery DataManager.

    Attributes:
        stride (int): The step size for sliding windows or downsampling.
        data_directory (str): Path to the directory containing all .mat files (e.g., 'C:/data/').
        train_battery_ids (list[str]): List of battery IDs to use for training (e.g., ['B0005', 'B0006']).
        val_battery_ids (list[str]): List of battery IDs for validation (e.g., ['B0007']).
        test_battery_ids (list[str]): List of battery IDs for testing (e.g., ['B0018']).
        interpolation_length (int): The fixed length for every cycle's time-series.
        batch_size (int): The number of samples per batch in the DataLoaders.
    """
    stride: int
    data_directory: str
    train_battery_ids: list[str]
    val_battery_ids: list[str]
    test_battery_ids: list[str]
    interpolation_length: int
    batch_size: int

# Validation of input parameters.
    def __post_init__(self) -> None:
        """
        Validates configuration parameters to ensure they are within acceptable ranges and types.

        Raises:
            TypeError: If a parameter has the wrong type.
            ValueError: If a parameter is negative, empty, or out of bounds.
            FileNotFoundError: If the data_directory does not exist.
            NotADirectoryError: If the data_directory path is not a directory.
        """
        # - stride
        if not isinstance(self.stride, int):
            raise TypeError(f"stride must be an integer, got {type(self.stride).__name__}.")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}.")

        # - data_directory
        if not isinstance(self.data_directory, str):
            raise TypeError(f"data_directory must be a string, got {type(self.data_directory).__name__}.")
        if len(self.data_directory.strip()) == 0:
            raise ValueError("data_directory cannot be empty.")
        if not os.path.exists(self.data_directory):
            raise FileNotFoundError(f"The specified data_directory does not exist: {self.data_directory}")
        if not os.path.isdir(self.data_directory):
            raise NotADirectoryError(f"The specified data_directory is not a directory: {self.data_directory}")

        # - battery_id lists validation
        for name, id_list in [('train_battery_ids', self.train_battery_ids),
                              ('val_battery_ids', self.val_battery_ids),
                              ('test_battery_ids', self.test_battery_ids)]:
            if not isinstance(id_list, list):
                raise TypeError(f"{name} must be a list, got {type(id_list).__name__}.")
            if not id_list:
                raise ValueError(f"{name} cannot be empty.")
            for battery_id in id_list:
                if not isinstance(battery_id, str):
                    raise TypeError(f"All IDs in {name} must be strings, but found {type(battery_id).__name__}.")
                if not battery_id.strip():
                    raise ValueError(f"Battery IDs in {name} cannot be empty strings.")

        # - interpolation_length
        if not isinstance(self.interpolation_length, int):
            raise TypeError(f"interpolation_length must be an integer, got {type(self.interpolation_length).__name__}.")
        if self.interpolation_length <= 0:
            raise ValueError(f"interpolation_length must be positive, got {self.interpolation_length}.")

        # - batch_size
        if not isinstance(self.batch_size, int):
            raise TypeError(f"batch_size must be an integer, got {type(self.batch_size).__name__}.")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}.")


class DataManager():
    """
    Manages the ETL (Extract, Transform, Load) pipeline for NASA Battery PCoE datasets.

    This class handles:
    1. Parsing complex MATLAB (.mat) structures.
    2. Extracting specific discharge cycle features (Voltage, Current, Temp).
    3. Resampling variable-length cycles to fixed lengths.
    4. Splitting data into Train/Val/Test sets to prevent leakage.
    5. Normalizing features using training statistics.
    6. Wrapping data into PyTorch DataLoaders.
    """

    # Constants for the NASA Data Structure.
    KEY_CYCLE: str = "cycle"
    KEY_TYPE: str = "type"
    KEY_DATA: str = "data"

    VAL_DISCHARGE: str = "discharge"

    FIELD_VOLTAGE: str = "Voltage_measured"
    FIELD_CURRENT: str = "Current_measured"
    FIELD_TEMP: str = "Temperature_measured"
    FIELD_CAPACITY: str = "Capacity"

    def __init__(
            self,
            config_data_manager: ConfigDataManager
    ) -> None:
        self.cfg_dm: ConfigDataManager = config_data_manager

        # Initialize the data loaders.
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader |  None = None
        self.test_loader: DataLoader | None = None
         

    def _load_and_convert_data(
            self,
            battery_id: str
    ) -> tuple[list[NDArray], list[NDArray], list[NDArray], list[float]]:
        """
        Parses a single battery's .mat file and extracts 'discharge' cycle data.
        
        Args:
            battery_id (str): ID of the battery to load (e.g., 'B0005').

        Returns:
            A tuple of lists: (voltages, currents, temperatures, capacities).
        """
        # 1. Construct the full path to the .mat file.
        file_path = os.path.join(self.cfg_dm.data_directory, f"{battery_id}.mat")
        
        # 2. Load the matlab data.
        mat_data: dict[str, Any] = scipy.io.loadmat(file_name=file_path)

        # 3. Access the top-level dictionary.
        data_struct: NDArray = mat_data[battery_id] 

        # 4. Access the 'cycle'struct.
        cycles: NDArray = data_struct[0, 0][self.KEY_CYCLE]

        # 5. Extract the specific data.
        voltages: list[NDArray] = []
        currents: list[NDArray] = []
        temperatures: list[NDArray] = []
        capacities: list[float] = []

        for cycle in cycles[0]:
            operation_type: str = str(cycle[self.KEY_TYPE][0])

            if operation_type == self.VAL_DISCHARGE:
                # Access the 'data' field inside the cycle.
                raw_data: NDArray = cycle[self.KEY_DATA] 

                # Extract specific fields.
                voltage: NDArray = raw_data[0, 0][self.FIELD_VOLTAGE][0]
                current: NDArray = raw_data[0, 0][self.FIELD_CURRENT][0]
                temperature: NDArray = raw_data[0, 0][self.FIELD_TEMP][0]
                capacity: float = float(raw_data[0, 0][self.FIELD_CAPACITY][0])

                # Append the parameters to the lists.
                voltages.append(voltage)
                currents.append(current)
                temperatures.append(temperature)
                capacities.append(capacity)
        
        return (voltages, currents, temperatures, capacities)

    def _interpolate_arrays(
            self,
            arrays: list[NDArray]
    ) -> list[NDArray]:
        """
        Resizes a list of variable-length arrays to a fixed length using linear interpolation.

        This ensures that all battery cycles, regardless of their original duration,
        have the same number of time steps (defined by cfg.interpolation_length)
        so they can be stacked into a tensor.

        Args:
            arrays (list[NDArray]): A list of 1D NumPy arrays of varying lengths.

        Returns:
            list[NDArray]: A list of 1D NumPy arrays, all having length `interpolation_length`.
        """
        
        # 1. Iterate over all the arrays of the list.
        interp_arrays: list[NDArray] = []

        for arr in arrays:
            # 2. Define the old and new ranges.
            x_old: NDArray = np.arange(len(arr))
            x_new: NDArray = np.linspace(
                start = 0,
                stop = len(arr) - 1,
                num = self.cfg_dm.interpolation_length
            )

            # 3. Interpolate the array.
            interpo_array: NDArray = np.interp(
                x = x_new,
                xp = x_old,
                fp = arr
            )

            # 4. Append the interpolated array to the tracker.
            interp_arrays.append(interpo_array)

        return interp_arrays
    
    def _process_batteries(
            self,
            battery_ids: list[str] 
    ) -> tuple[NDArray, NDArray]:
        """
        Loads, processes and combines data for a given list of battery IDs.

        Args:
            battery_ids (list[str]): A list of battery identifiers to process.

        Returns:
            A tuple containing the combined stacked inputs (X) and outputs (y).
        """

        all_stacked_inputs: list[NDArray] = []
        all_outputs: list[NDArray] = []

        for battery_id in battery_ids:
            # 1. Load raw data for one battery.
            voltages, currents, temperatures, capacities = self._load_and_convert_data(battery_id=battery_id)

            # 2. Interpolate the time-series features.
            interp_voltages: list[NDArray] = self._interpolate_arrays(arrays=voltages)
            interp_currents: list[NDArray] = self._interpolate_arrays(arrays=currents)
            interp_temperatures: list[NDArray] = self._interpolate_arrays(arrays=temperatures)

            # 3. Convert lists of arrays into single 2D arrays.
            stacked_voltages: NDArray = np.vstack(interp_voltages)
            stacked_currents: NDArray = np.vstack(interp_currents)
            stacked_temperatures: NDArray = np.vstack(interp_temperatures)
            outputs: NDArray = np.array(capacities).reshape(-1, 1)

            # 4. Stack features into a tensor (cycles, time, features).
            stacked_inputs: NDArray = np.stack(
                (stacked_voltages, stacked_currents, stacked_temperatures),
                axis = 2
            )

            # 5. Append the processed data for this battery to the master lists.
            all_stacked_inputs.append(stacked_inputs)
            all_outputs.append(outputs)

        # 6. Concatenate data from all batteries into final X and y arrays.
        final_inputs: NDArray = np.concatenate(
            all_stacked_inputs, 
            axis = 0
        )
        final_outputs: NDArray = np.concatenate(
            all_outputs,
            axis = 0
        )

        return (final_inputs, final_outputs)


    def _scale_features(
            self,
            data: NDArray,
            min_val: NDArray | None = None,
            max_val: NDArray | None = None
    ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Applies Min-Max normalization to the data.

        If `min_val` and `max_val` are None, they are calculated from the input `data` (Fit).
        If provided, the input `data` is scaled using those values (Transform).
        This method supports 3D inputs (N, T, F) by calculating stats across axes (0, 1).

        Args:
            data (NDArray): The input array to scale.
            min_val (NDArray | None): Pre-computed minimum values. Defaults to None.
            max_val (NDArray | None): Pre-computed maximum values. Defaults to None.

        Returns:
            tuple[NDArray, NDArray, NDArray]: A tuple containing:
                - scaled_data: The normalized array with values between 0 and 1.
                - min_val: The minimum values used for scaling.
                - max_val: The maximum values used for scaling.
        """

        # 1. Calculate the min and maximum values if not present.
        if (min_val is None) or (max_val is None):
            min_val: NDArray = np.min(
                a = data,
                axis = (0, 1),
                keepdims = True # Keeps the original dimensions.
            )
            max_val: NDArray = np.max(
                a = data,
                axis = (0, 1),
                keepdims = True
            )

        # 2. Add epsilon to avoid division by zero.
        epsilon: float = 1e-8

        # 3. Apply scaling the scaling.
        scaled_data: NDArray = (data - min_val) / (max_val - min_val + epsilon)

        return (scaled_data, min_val, max_val)

    def _create_dataloader(
            self,
            x: NDArray,
            y: NDArray,
            shuffle: bool
    ) -> DataLoader:
        """
        Converts NumPy arrays into a PyTorch DataLoader.

        Casts inputs to float32 (standard for PyTorch models) and wraps them
        in a TensorDataset.

        Args:
            x (NDArray): Input features array.
            y (NDArray): Target labels array.
            shuffle (bool): Whether to shuffle the data (True for training, False for eval).

        Returns:
            DataLoader: A PyTorch DataLoader containing the dataset.
        """

        # 1. Change the datatype of the arrays to float32.
        x_float32: NDArray = x.astype(dtype=np.float32)
        y_float32: NDArray = y.astype(dtype=np.float32)

        # 2. Create tensors for the inputs and targets.
        x_tensor: Tensor = torch.from_numpy(x_float32)
        y_tensor: Tensor = torch.from_numpy(y_float32)

        # 3. Create the dataset.
        dataset = TensorDataset(x_tensor, y_tensor)

        # 4. Create the dataloader.
        loader = DataLoader(
            dataset = dataset,
            batch_size = self.cfg_dm.batch_size,
            shuffle = shuffle
        )

        return loader

    def prepare_data(self) -> tuple[float, float]:
        """
        Orchestrates the data pipeline using battery IDs for splitting.

        Steps:
            1. Processes lists of battery IDs to create Train, Val and Test sets.
            2. Normalizes data (fitting scalar on Train then applying to Val and Test).
            3. Instantiates PyTorch DataLoaders.

        Returns:
            (min_capacity, max_capacity) from training ser for denormalization.
        """

        # 1. Create datasets by processing the specified batteries for each split.
        train_x, train_y = self._process_batteries(battery_ids=self.cfg_dm.train_battery_ids)
        val_x, val_y = self._process_batteries(battery_ids=self.cfg_dm.val_battery_ids)
        test_x, test_y = self._process_batteries(battery_ids=self.cfg_dm.test_battery_ids)

        # 2. Normalize the input and output data.
        normalized_train_x, min_train_x, max_train_x = self._scale_features(data=train_x)
        normalized_train_y, min_train_y, max_train_y = self._scale_features(data=train_y)

        # Transform the validation and test sets using the training data's min/max values.
        normalized_val_x, _, _ = self._scale_features(
            data = val_x,
            min_val = min_train_x,
            max_val = max_train_x
        )
        normalized_val_y, _, _ = self._scale_features(
            data = val_y,
            min_val = min_train_y,
            max_val = max_train_y
        )

        normalized_test_x, _, _ = self._scale_features(
            data = test_x,
            min_val = min_train_x,
            max_val = max_train_x
        )
        normalized_test_y, _, _ = self._scale_features(
            data = test_y,
            min_val = min_train_y,
            max_val = max_train_y
        )

        # 3. Create Dataloaders from the normalized data.
        self.train_loader = self._create_dataloader(
            x = normalized_train_x,
            y = normalized_train_y,
            shuffle = True
        )

        self.val_loader = self._create_dataloader(
            x = normalized_val_x,
            y = normalized_val_y,
            shuffle = False
        )
        self.test_loader = self._create_dataloader(
            x = normalized_test_x,
            y = normalized_test_y,
            shuffle = False
        )

        # 4. Return scaling factors from the training set for re-scaling predictions.
        return (min_train_y.item(), max_train_y.item())
        

    def get_loaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Returns the prepared PyTorch DataLoaders.

        Returns:
            tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)

        Raises:
            RuntimeError: If called before `prepare_data()`.
        """

        if self.train_loader is None:
            raise RuntimeError("Data has not been prepared yet. Run prepare_data() first.")
        
        return (self.train_loader, self.val_loader, self.test_loader)
