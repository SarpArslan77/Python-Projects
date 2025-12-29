
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
    stride: int
    mat_file_name: str
    battery_id: str # Battery option.
    interpolation_length: int # Length of every tracker.
    train_split: float
    val_split: float
    batch_size: int

    # Validation of input parameters.
    def __post__init__(self) -> None:
        pass


class DataManager():
    def __init__(
            self,
            config_data_manager: ConfigDataManager
    ) -> None:
        self.cfg_dm: ConfigDataManager = config_data_manager

        """# Trackers for the to-be-read parameter data.
        self.discharge_voltages: list[NDArray] = []
        self.discharge_currents: list[NDArray] = []
        self.discharge_times: list[NDArray] = []
        self.discharge_capacities: list[float] = []"""

        # Initialize the data loaders.
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader |  None = None
        self.test_loader: DataLoader | None = None
         

    def _load_and_convert_data(self) -> tuple[list[NDArray], list[NDArray], list[NDArray], list[NDArray], list[float]]:
        # 1. Load the matlab data.
        mat_data: dict[str, Any] = scipy.io.loadmat(file_name=self.cfg_dm.mat_file_name)

        # 2. Access the top-level dictionary.
        data_struct = mat_data[self.cfg_dm.battery_id] #TODO ATH

        # 3. Access the 'cycle'struct.
        cycles = data_struct[0, 0]["cycle"]

        # 4. Extract the specific data.
        discharge_voltages: list[NDArray] = []
        discharge_currents: list[NDArray] = []
        discharge_temperatures: list[NDArray] = []
        discharge_times: list[NDArray] = []
        discharge_capacities: list[float] = []

        for cycle in cycles[0]:
            operation_type: str = str(cycle["type"][0])

            if operation_type == "discharge":
                # Access the 'data' field inside the cycle.
                raw_data = cycle["data"] # TODO ATH

                # Extract specific fields.
                discharge_voltage: NDArray = raw_data[0, 0]["Voltage_measured"][0]
                discharge_current: NDArray = raw_data[0, 0]["Current_measured"][0]
                discharge_temperature: NDArray = raw_data[0, 0]["Temperature_measured"][0]
                discharge_time: NDArray = raw_data[0, 0]["Time"][0]
                discharge_capacity: NDArray = raw_data[0, 0]["Capacity"][0]

                # Append the parameters to the lists.
                discharge_voltages.append(discharge_voltage)
                discharge_currents.append(discharge_current)
                discharge_temperatures.append(discharge_temperature)
                discharge_times.append(discharge_time)
                discharge_capacities.append(discharge_capacity)
        
        return (discharge_voltages, discharge_currents, discharge_temperatures, discharge_times, discharge_capacities)

    def _interpolate_arrays(
            self,
            arrays: list[NDArray]
    ) -> list[NDArray]:
        # 1. Iterate over all the arrays of the list.
        interpolated_arrays: list[NDArray] = []

        for arr in arrays[:]:
            # 2. Define the old and new ranges.
            x_old: NDArray = np.arange(len(arr))
            x_new: NDArray = np.linspace(
                start = 0,
                stop = len(arr) - 1,
                num = self.cfg_dm.interpolation_length
            )

            # 3. Interpolate the array.
            interpolated_array: NDArray = np.interp(
                x = x_new,
                xp = x_old,
                fp = arr
            )

            # 4. Append the interpolated array to the tracker.
            interpolated_arrays.append(interpolated_array)

        return interpolated_arrays

    def _scale_features(
            self,
            data: NDArray,
    ) -> NDArray:

        # 1. Calculate the min and maximum values.
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

        # 3. Apply scaling the scaling and return.
        return () #TODO KEEP FROM HERE CHECK TEH METHOD WHETHER THEY ARE PROF
        
    
    def _normalize_list(
            self,
            list_data: list
    ) -> NDArray:
        # 1. Convert the list into an array.
        array: NDArray = np.asarray(list_data)

        # 2. Find the min and max of the array.
        min_value: float = array.min()
        max_value: float = array.max()

        # 3. Normalize the array.
        normalized_array: NDArray = (array - min_value) / (max_value - min_value)

        # 4. Reshape the array into a 2D.
        reshaped_array: NDArray = normalized_array.reshape(-1, 1) # (168, 1)

        return reshaped_array


    def _create_dataloaders(
            self,
            inputs: NDArray,
            targets: NDArray
    ) -> None:
        
        # 1. Change the datatype of the arrays to float32.
        inputs_float32: NDArray = inputs.astype(dtype=np.float32)
        targets_float32: NDArray = targets.astype(dtype=np.float32)

        # 2. Create tensors for the inputs and targets.
        inputs_tensor: Tensor = torch.from_numpy(ndarray=inputs_float32)
        targets_tensor: Tensor = torch.from_numpy(ndarray=targets_float32)

        # 3. Seperate the data into training, validation and test sets.
        dataset_size: int = inputs_tensor.shape[0]
        train_size: int = int(dataset_size * self.cfg_dm.train_split)
        val_size: int = int(dataset_size * self.cfg_dm.val_split)

        # Split the dataset.
        train_dataset = TensorDataset(inputs_tensor[:train_size, :, :], targets_tensor[:train_size, :])
        val_dataset = TensorDataset(inputs_tensor[train_size:train_size+val_size, :, :], targets_tensor[train_size:train_size+val_size, :])
        test_dataset = TensorDataset(inputs_tensor[train_size+val_size:, :, :], targets_tensor[train_size+val_size:, :])

        # 4. Create the dataloaders.
        self.train_loader = DataLoader(
            dataset = train_dataset,
            batch_size = self.cfg_dm.batch_size,
            shuffle = True
        )
        self.val_loader = DataLoader(
            dataset = val_dataset,
            batch_size = self.cfg_dm.batch_size,
            shuffle = False
        )
        self.test_loader = DataLoader(
            dataset = test_dataset,
            batch_size = self.cfg_dm.batch_size,
            shuffle = False
        )

    def prepare_data(self) -> None:
        # 1. Load the matlab data and convert it into arrays.
        discharge_voltages, discharge_currents, discharge_temperatures, discharge_times, discharge_capacities = self._load_and_convert_data()

        # 2. Interpolate the arrays, so all of them has the same length.
        interpolated_discharge_voltages: list[NDArray] = self._interpolate_arrays(arrays=discharge_voltages)
        interpolated_discharge_currents: list[NDArray] = self._interpolate_arrays(arrays=discharge_currents)
        interpolated_discharge_temperatures: list[NDArray] = self._interpolate_arrays(arrays=discharge_temperatures)

        # 3. Convert the list of 1D arrays into 2D arrays.
        discharge_voltages_2D: NDArray = np.vstack(interpolated_discharge_voltages)
        discharge_currents_2D: NDArray = np.vstack(interpolated_discharge_currents)
        discharge_temperatures_2D: NDArray = np.vstack(interpolated_discharge_temperatures)

        # 4. Stack features along the 3rd dimension (Features axis).
        stacked_discharge_inputs: NDArray = np.stack(
            (discharge_voltages_2D, discharge_currents_2D, discharge_temperatures_2D),
            axis = 2
        ) # (Cyles, Time, Features) -> (168, 500, 3).

        # 5. Normalize the input and output arrays.
        normalized_discharge_inputs: NDArray = self._scale_features(data=stacked_discharge_inputs)
        normalized_discharge_targets: NDArray = self._normalize_list(list_data=discharge_capacities)

        # 6. Create Dataloaders from the arrays.
        self._create_dataloaders(
            inputs = normalized_discharge_inputs,
            targets = normalized_discharge_targets
        )

    def get_loaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Returns the loaders. Raises error if data is not prepared.
        """

        if self.train_loader is None:
            raise RuntimeError("Data has not been prepared yet. Run prepare_data() first.")
        
        return (self.train_loader, self.val_loader, self.test_loader)