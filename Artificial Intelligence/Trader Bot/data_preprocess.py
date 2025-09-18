
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

READ_FILE_PATH: str = r"C:/Users/Besitzer/Desktop/Python/AI Projects/Trader Bot/NSEI 2015-2023.csv"

df: pd.DataFrame = pd.read_csv(READ_FILE_PATH)

# We will fix the missing dates, before using the corresponding values.
data_dates: list[str] = df["Date"].tolist()
# Read the data for OHLCV.
data_open: list[float] = df["Open"].tolist()
data_high: list[float] = df["High"].tolist()
data_low: list[float] = df["Low"].tolist()
data_close: list[float] = df["Close"].tolist()

data_dict: dict[str, list[str | float]] = {
    "Date" : data_dates,
    "Open" : data_open,
    "High" : data_high,
    "Low" : data_low,
    "Close" : data_close
}
new_df: pd.DataFrame = pd.DataFrame(data_dict)

# Convert 'Date' column to datetime objects.
new_df["Date"] = pd.to_datetime(df["Date"])
# Set the 'Date' column as index for time-series operations.
new_df.set_index("Date", inplace=True)

# Create a complete data range and reindex.
all_dates: pd.DatetimeIndex = pd.date_range(start=new_df.index.min(), end=new_df.index.max(), freq="D")
df_complete: pd.DataFrame = new_df.reindex(all_dates) # NaN's for missing dates

# Fill NaN values with the last known value.
fixed_df: pd.DataFrame = df_complete.ffill()

# Sequence length for test_data_date creation and sliding window method.
sequence_length: int = 30

# We calculate the split between train and test here, in order to prevent future data leakages.
data_len: int = len(all_dates) # All features has the same amount of rows, so the 'year' feature is here arbitrary.
train_percentage: float = 0.7
train_num_samples: int = round(data_len*train_percentage)
val_percentage: float = 0.1
val_num_samples: int = round(data_len * val_percentage)
# Create an numpy array of datetime64 object as test dates for graph.
test_data_dates: np.ndarray = np.array(all_dates[train_num_samples:])
# Convert the datetime64 array to string array.
string_test_data_dates: np.ndarray = test_data_dates.astype(str)
# Replace the trailing 'T00:00:00.000000000' from the dates with nothing, so it removes the trailing part compeletly.
test_data_dates = np.char.replace(string_test_data_dates, "T00:00:00.000000000", "")
# Remove the sequence length from the start to match with the X_test later.
sequence_test_data_dates: np.ndarray = test_data_dates[sequence_length:]

# Repeat the same for all dates.
fixed_data_dates: np.ndarray = np.array(all_dates)
string_fixed_data_dates: np.ndarray = fixed_data_dates.astype(str)
final_data_dates: np.ndarray = np.char.replace(string_fixed_data_dates, "T00:00:00.000000000", "")

# For data use the newly fixed ones.
fixed_data_open: list[float] = fixed_df["Open"].tolist()
fixed_data_high: list[float] = fixed_df["High"].tolist()
fixed_data_low: list[float] = fixed_df["Low"].tolist()
fixed_data_close: list[float] = fixed_df["Close"].tolist()

# Use vectorized feature engineering.
date_index: pd.DatetimeIndex = fixed_df.index

# Create all features at once using vectorized operations.
fixed_df["Year"] = date_index.year
fixed_df["Month"] = date_index.month
fixed_df["DayOfMonth"] = date_index.day
fixed_df["DayOfWeek"] = date_index.dayofweek # Monday=0, Sunday=6

# Create cyclical features directly on the DataFrame columns.
fixed_df["MonthSin"] = np.sin(2 * np.pi * fixed_df["Month"] / 12)
fixed_df["MonthCos"] = np.cos(2 * np.pi * fixed_df["Month"] / 12)
fixed_df["DayOfWeekSin"] = np.sin(2 * np.pi * fixed_df["DayOfWeek"] / 7)
fixed_df["DayOfWeekCos"] = np.cos(2 * np.pi * fixed_df["DayOfWeek"] / 7)

# Define the columns, we will need for the model as inputs.
feature_columns: np.ndarray[str] = np.array([
    "Year", "MonthSin", "MonthCos", "DayOfMonth", "DayOfWeekSin", "DayOfWeekCos",
    "Open", "High", "Low", "Close"
])
# Create a new DataFrame with only features we need.
model_data_df: pd.DataFrame = fixed_df[feature_columns]

# Convert to numpy array for processing.
all_data_np: np.ndarray = model_data_df.to_numpy()

# Split the data into train, test and validation sets before normalization.
train_data_unnormalized: np.ndarray = all_data_np[:train_num_samples]
val_data_unnormalized: np.ndarray = all_data_np[train_num_samples: train_num_samples+val_num_samples]
test_data_unnormalized: np.ndarray = all_data_np[train_num_samples+val_num_samples:]

# Define the normalization range.
scaler = MinMaxScaler(feature_range=(-1, 1))
# Scales goes only through the training data, to prevent data leakeage from test datas, in order to find the max and min values for each features aka column.
scaler.fit(train_data_unnormalized)

# Get the min and max vals for each feature for unnormalization in main.py
min_vals: np.ndarray = scaler.data_min_[-4:]
max_vals: np.ndarray = scaler.data_max_[-4:]

# Apply the normalization with scaler limits from training set on both datasets.
train_data: np.ndarray = scaler.transform(train_data_unnormalized)
test_data: np.ndarray = scaler.transform(test_data_unnormalized)
val_data: np.ndarray = scaler.transform(val_data_unnormalized)

# Create sequences using the sliding window method.
stride: int = 1 # Amount of steps to create the next sequence
output_columns: list[int] = [6, 7, 8, 9] # OHLCV datas

def create_sequences(
    data: np.ndarray
) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]]]:
    
    created_sequences: list[list[torch.Tensor]] = []
    targets: list[list[torch.Tensor]] = []

    for i in range(0, data.shape[0]-sequence_length, stride):
        input_sequence: np.ndarray = data[i:i+sequence_length, :]
        output_targets: np.ndarray = data[i+sequence_length, output_columns]
        # Turn the numpy arrays into tensors, before stacking them.
        #   Create them explicitly as float32
        input_sequence_tensor: torch.Tensor = torch.tensor(input_sequence, dtype=torch.float32)
        output_targets_tensor: torch.Tensor = torch.tensor(output_targets, dtype=torch.float32)

        created_sequences.append(input_sequence_tensor)
        targets.append(output_targets_tensor)

    X_sequences: torch.Tensor = torch.stack(created_sequences)
    y_targets: torch.Tensor = torch.stack(targets)

    return [X_sequences, y_targets]

X_train, y_train = create_sequences(train_data)
X_test, y_test = create_sequences(test_data)
X_val, y_val = create_sequences(val_data)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
val_dataset = TensorDataset(X_val, y_val)

batch_size: int = 160

train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True,

)
test_loader = DataLoader(
    dataset = test_dataset,
    batch_size = batch_size,
    shuffle = False,

)
val_loader = DataLoader(
    dataset = val_dataset,
    batch_size = batch_size,
    shuffle = False,

)

