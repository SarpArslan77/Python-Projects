import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Read the Excel file from the given path
#! Change the file path before uploading to GIT-Hub
read_file_path: str = r"C:/Users/Besitzer/Desktop/Python/AI_Projects/Mars_Weatherforecast/fixed_mars_weather.xlsx"

try:
    df = pd.read_excel(read_file_path)
    print("\nFile red successfully!")
    if df.empty:
        print("\nThe Excel sheet is empty.")
except FileNotFoundError:
    print(f"\nError: The file {read_file_path} was not found. Exitting the program.")
    exit()
    

mars_weather_data: dict = {}
# Turn the excel data into NumPy arrays in a dictionary
for column_name in df.columns:
    # Data is being represented by the sol, so we don't need it as an input to the Neural Network (NN).
    if column_name != "terrestrial_date":
        mars_weather_data[column_name] = df[column_name].to_numpy()

# Before the training optimize the parameters, if needed.
for key_name in list(mars_weather_data.keys()):

    # Turn the NumPy arrays into Pytorch tensors
    pytorch_tensor = torch.from_numpy(mars_weather_data[key_name]).float()
    # Update the dictionary.
    mars_weather_data[key_name] = pytorch_tensor

# Stack the tensors for the Long-Short-Time-Memory (LSTM) NN in a one unified 2D Tensor.
stacked_tensor = torch.stack(list(mars_weather_data.values()), dim=1)

# Create sequences using the sliding window method.
sequence_length: int = 30
stride: int = sequence_length # Overlapping amount
output_columns: list[int] = [1, 2, 3] # Output columns = min_temp, max_temp, pressure

created_sequences: list[list] = []
targets: list[list] = []

for i in range(0, (stacked_tensor.size(0) - sequence_length), stride):
    # Slice the input sequences and targets.
    X = stacked_tensor[i : i+sequence_length, :] # Shape: [30, num_features]
    y = stacked_tensor[i + sequence_length, output_columns] 

    created_sequences.append(X)
    targets.append(y)

X_sequences = torch.stack(created_sequences) # Shape: [num_samples, 30, num_features]
y_targets = torch.stack(targets) # Shape: [num_samples, 3]

# Split the data and create DataLoaders
training_data_percentage: int = 0.8 # 80%
total_num_samples: int = X_sequences.size(0)
training_data_num_samples: int = round(total_num_samples*training_data_percentage)

X_training = X_sequences[:training_data_num_samples, :, :]
X_test = X_sequences[training_data_num_samples:, :, :] # Rest are test samples.
y_training = y_targets[:training_data_num_samples, :]
y_test = y_targets[training_data_num_samples:, :]

# Create DataLoader compatible datasets.
training_dataset = TensorDataset(X_training, y_training)
test_dataset = TensorDataset(X_test, y_test)

batch_size: int = 5

# Use DataLoader for easer iterating for the sets.
train_loader = DataLoader(
    dataset = training_dataset,
    batch_size = batch_size,
    shuffle = True
)
test_loader = DataLoader(
    dataset = test_dataset,
    batch_size = batch_size,
    shuffle = False
)


        

