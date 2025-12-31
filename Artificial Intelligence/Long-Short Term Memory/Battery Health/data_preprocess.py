
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
        stride (int): The step size for sliding windows (if applicable) or downsampling.
        mat_file_name (str): Path to the source .mat file (e.g., 'B0005.mat').
        battery_id (str): The specific battery key within the .mat structure (e.g., 'B0005').
        interpolation_length (int): The fixed length to which every cycle's time-series will be resized.
        train_split (float): Percentage of cycles to use for training (0.0 - 1.0).
        val_split (float): Percentage of cycles to use for validation (0.0 - 1.0).
        batch_size (int): The number of samples per batch in the DataLoaders.
    """

    stride: int
    mat_file_name: str
    battery_id: str # Battery option.
    interpolation_length: int # Length of every tracker.
    train_split: float
    val_split: float
    batch_size: int

    # Validation of input parameters.
    def __post__init__(self) -> None:
        """
        Validates configuration parameters to ensure they are within acceptable ranges and types.

        Raises:
            TypeError: If a parameter has the wrong type.
            ValueError: If a parameter is negative, empty, or out of bounds.
        """
        # - stride
        if not isinstance(self.stride, int):
            raise TypeError(f"stride must be an integer, got {type(self.stride).__name__}.")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}.")

        # - mat_file_name
        if not isinstance(self.mat_file_name, str):
            raise TypeError(f"mat_file_name must be a string, got {type(self.mat_file_name).__name__}.")
        if len(self.mat_file_name) == 0:
            raise ValueError("mat_file_name cannot be empty.")
        if not self.mat_file_name.endswith(".mat"):
            raise ValueError(f"mat_file_name must end with '.mat', got {self.mat_file_name}.")

        # - battery_id
        if not isinstance(self.battery_id, str):
            raise TypeError(f"battery_id must be a string, got {type(self.battery_id).__name__}.")
        if len(self.battery_id) == 0:
            raise ValueError("battery_id cannot be empty.")

        # - interpolation_length
        if not isinstance(self.interpolation_length, int):
            raise TypeError(f"interpolation_length must be an integer, got {type(self.interpolation_length).__name__}.")
        if self.interpolation_length <= 0:
            raise ValueError(f"interpolation_length must be positive, got {self.interpolation_length}.")

        # - train_split
        if not isinstance(self.train_split, float):
            raise TypeError(f"train_split must be a float, got {type(self.train_split).__name__}.")
        if not (0.0 < self.train_split < 1.0):
            raise ValueError(f"train_split must be between 0.0 and 1.0, got {self.train_split}.")

        # - val_split
        if not isinstance(self.val_split, float):
            raise TypeError(f"val_split must be a float, got {type(self.val_split).__name__}.")
        if not (0.0 < self.val_split < 1.0):
            raise ValueError(f"val_split must be between 0.0 and 1.0, got {self.val_split}.")

        # --- train_split and val_split combination
        if (self.train_split + self.val_split) >= 1.0:
            raise ValueError(f"train_split + val_split must be less than 1.0 (to leave room for test set), got {self.train_split + self.val_split}.")

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
    6. wrapping data into PyTorch DataLoaders.
    """

    def __init__(
            self,
            config_data_manager: ConfigDataManager
    ) -> None:
        self.cfg_dm: ConfigDataManager = config_data_manager

        # Initialize the data loaders.
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader |  None = None
        self.test_loader: DataLoader | None = None
         

    def _load_and_convert_data(self) -> tuple[list[NDArray], list[NDArray], list[NDArray], list[NDArray], list[float]]:
        """
        Parses the raw .mat file and extracts time-series data for 'discharge' operations.

        Iterates through the nested MATLAB struct to find cycles labeled as 'discharge'.
        Extracts Voltage, Current, Temperature, Time, and Capacity.

        Returns:
            tuple: A tuple containing five lists:
                - voltages (list[NDArray]): List of voltage arrays per cycle.
                - currents (list[NDArray]): List of current arrays per cycle.
                - temperatures (list[NDArray]): List of temperature arrays per cycle.
                - times (list[NDArray]): List of time arrays per cycle.
                - capacities (list[float]): List of scalar capacity values per cycle.
        """
        
        # 1. Load the matlab data.
        mat_data: dict[str, Any] = scipy.io.loadmat(file_name=self.cfg_dm.mat_file_name)

        # 2. Access the top-level dictionary.
        data_struct: NDArray = mat_data[self.cfg_dm.battery_id] 

        # 3. Access the 'cycle'struct.
        cycles: NDArray = data_struct[0, 0]["cycle"]

        # 4. Extract the specific data.
        voltages: list[NDArray] = []
        currents: list[NDArray] = []
        temperatures: list[NDArray] = []
        times: list[NDArray] = []
        capacities: list[float] = []

        for cycle in cycles[0]:
            operation_type: str = str(cycle["type"][0])

            if operation_type == "discharge":
                # Access the 'data' field inside the cycle.
                raw_data: NDArray = cycle["data"] 

                # Extract specific fields.
                voltage: NDArray = raw_data[0, 0]["Voltage_measured"][0]
                current: NDArray = raw_data[0, 0]["Current_measured"][0]
                temperature: NDArray = raw_data[0, 0]["Temperature_measured"][0]
                time: NDArray = raw_data[0, 0]["Time"][0]
                capacity: float = float(raw_data[0, 0]["Capacity"][0])

                # Append the parameters to the lists.
                voltages.append(voltage)
                currents.append(current)
                temperatures.append(temperature)
                times.append(time)
                capacities.append(capacity)
        
        return (voltages, currents, temperatures, times, capacities)

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
        x_tensor: Tensor = torch.from_numpy(ndarray=x_float32)
        y_tensor: Tensor = torch.from_numpy(ndarray=y_float32)

        # 3. Create the dataset.
        dataset = TensorDataset(x_tensor, y_tensor)

        # 4. Create the dataloader.
        loader = DataLoader(
            dataset = dataset,
            batch_size = self.cfg_dm.batch_size,
            shuffle = shuffle
        )

        return loader

    def prepare_data(self) -> None:
        """
        Orchestrates the complete data preprocessing pipeline.

        Steps:
        1. Loads raw data from the .mat file.
        2. Interpolates all time-series to a fixed length.
        3. Stacks features into a 3D Tensor (Cycles, Time, Features).
        4. Splits data into Train, Validation, and Test sets.
        5. Normalizes data (fitting scaler on Train, applying to Val/Test to avoid leakage).
        6. Instantiates PyTorch DataLoaders.
        """

        # 1. Load the matlab data and convert it into arrays.
        voltages, currents, temperatures, times, capacities = self._load_and_convert_data()

        # 2. Interpolate the arrays, so all of them has the same length.
        interp_voltages: list[NDArray] = self._interpolate_arrays(arrays=voltages)
        interp_currents: list[NDArray] = self._interpolate_arrays(arrays=currents)
        interp_temperatures: list[NDArray] = self._interpolate_arrays(arrays=temperatures)

        # 3. Convert the list of 1D arrays into 2D arrays.
        voltages_2D: NDArray = np.vstack(interp_voltages)
        currents_2D: NDArray = np.vstack(interp_currents)
        temperatures_2D: NDArray = np.vstack(interp_temperatures)

        # Conver the list of float to 2D array.
        capacities_np: NDArray = np.array(capacities)
        outputs: NDArray = capacities_np.reshape(-1, 1)

        # 4. Stack features along the 3rd dimension (Features axis).
        stacked_inputs: NDArray = np.stack(
            (voltages_2D, currents_2D, temperatures_2D),
            axis = 2
        ) # (Cyles, Time, Features) -> (168, 500, 3).

        # 5. Split the data into training, validation and test.
        # Calculate the sizes.
        data_size: int = stacked_inputs.shape[0]

        train_size: int = int(data_size * self.cfg_dm.train_split)
        val_size: int = int(data_size * self.cfg_dm.val_split)

        # Split the input data.
        train_x, val_x, test_x = np.split(
            ary = stacked_inputs,
            indices_or_sections = [train_size, train_size + val_size] # Cuts at two indexes, creates 3 arrays.
        )

        # Split the output data.
        train_y, val_y, test_y = np.split(
            ary = outputs,
            indices_or_sections = [train_size, train_size + val_size]
        )

        # 6. Normalize the input and output data.
        normalized_train_x, min_train_x, max_train_x = self._scale_features(
            data = train_x,
            min_val = None,
            max_val = None
        )
        normalized_train_y, min_train_y, max_train_y = self._scale_features(
            data = train_y,
            min_val = None,
            max_val = None
        )

        # In order to prevent the data leakeage, the validation and test data is normalized with the min and max value of training data.
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

        # 7. Create Dataloaders from the data.
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
