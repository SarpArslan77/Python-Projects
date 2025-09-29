
#! max_temp shows significant difference compared to other parameters, find the reason why

import torch
import torch.nn as nn

from prepare_data import (
    train_loader,
    test_loader
)
from lstm import LSTM

if __name__ == "__main__":
    # Model values
    input_size: int = 8 # Number of features
    hidden_size: int = 50
    num_layers: int = 2 # Number of stacked LSTM-layers
    output_size: int = 3 # Predicting 3 values: min_temp, max_temp, pressure
    # Training values
    num_epochs: int = 100
    learning_rate: float = 0.1
    batch_size: int = 32

    model = LSTM(
        input_size = input_size,
        hidden_size = hidden_size,
        num_layers = num_layers,
        output_size = output_size
    )

    # Define Loss Function and Optimizer.
    loss_func = nn.MSELoss() # Mean-Squared Error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train() # Open the training mode.
        for seqs, labels in train_loader:
            # Forward pass
            outputs = model(seqs)
            loss = loss_func(outputs, labels)
            # Backward-pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluation loop
    model.eval()
    with torch.no_grad():
        # Get one batch of data from the test loader.
        inputs, labels = next(iter(test_loader))

        # Initialize the LSTM Model.
        predictions = model(inputs)

        # Print the first few predictions and labels.
        for i in range(len(predictions)):
            pred = predictions[i]
            label = labels[i]
            print(f"\nSample {i+1}:")
            print(f"  -> Predicted: [min_temp: {pred[0]:.2f}, max_temp: {pred[1]:.2f}, pressure: {pred[2]:.2f}]")
            print(f"  -> Actual:    [min_temp: {label[0]:.2f}, max_temp: {label[1]:.2f}, pressure: {label[2]:.2f}]")
            print(f" --- The difference percentagewise is: --- " )
            print(f" i) min_temp: {abs(1 - pred[0]/label[0])*100:.2f}%")
            print(f" ii) max_temp: {abs(1 - pred[1]/label[1])*100:.2f}%")
            print(f" iii) pressure: {abs(1 - pred[2]/label[2])*100:.2f}%\n")
