
# 
import os
devnull = os.open(os.devnull, os.O_WRONLY)
old_stderr_fd = os.dup(2)
os.dup2(devnull, 2)
try:
    import intel_extension_for_pytorch as ipex
finally:
    os.dup2(old_stderr_fd, 2)
    os.close(devnull)
    os.close(old_stderr_fd)

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
#!from torch.xpu.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
import mplcursors

from data_preprocess import (
    train_loader,
    test_loader,
    val_loader,
    min_vals,
    max_vals,
    sequence_test_data_dates
)
from lstm import LSTM

if __name__ == "__main__":

    # Check whether Intel XE GPU is avaiable, if not use the CPU
    try:
        device = torch.device("xpu") 
    except:
        device = torch.device("cpu") 
    print(f"\nUsing device: {device}") 

    # Model variables.
    input_size: int = 10
    hidden_size: int = 50
    num_layers: int = 2 # Number of stacked LSTM-layers
    output_size: int = 4
    # Training variables.
    num_epochs: int = 10
    learning_rate: float = 1e-3

    model = LSTM(
        input_size = input_size,
        hidden_size = hidden_size,
        num_layers = num_layers,
        output_size = output_size
    ).to(device)

    # Define Loss Function and Optimizer with a scheduler.
    loss_func = nn.MSELoss() # Mean Squared Error for regresion.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode = "min", 
        factor = 0.5, 
        patience = 5, 
        min_lr = 1e-5
    )

    # Apply IPEX optimizations to the model and optimizer
    model, optimizer = ipex.optimize(model, optimizer=optimizer)
    model: LSTM
    optimizer: torch.optim.Adam

    # Turn the max & min numpy arrays into tensors.
    tensor_max_vals, tensor_min_vals = torch.Tensor(max_vals), torch.Tensor(min_vals)
    # Send the both min and max values tensors to the GPU.
    tensor_min_vals, tensor_max_vals = tensor_min_vals.to(device), tensor_max_vals.to(device)

    # Training loop
    print("\nStarting with training.")
    train_all_ep: list[list[float]] = []
    for epoch in range(num_epochs):
        model.train()
        print(f"\nEpoch: {epoch+1}/{num_epochs}")
        # Create empty lists to store tensors on the GPU.
        epoch_train_outputs: list[torch.Tensor] = []
        epoch_train_labels: list[torch.Tensor] = []
        for i, (train_seqs, train_labels) in enumerate(train_loader):
            train_seqs: torch.Tensor
            train_labels: torch.Tensor
            # Send the sequences and labels to to choosen device (ideally GPU).
            train_seqs, train_labels = train_seqs.to(device), train_labels.to(device)
            # Forward pass
            train_outputs: torch.Tensor  = model(train_seqs)
            train_loss: torch.Tensor  = loss_func(train_outputs, train_labels)

            # Backwardpass and optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Append the row tensors to the lists.
            epoch_train_outputs.append(train_outputs)
            epoch_train_labels.append(train_labels)

        # Post epoch analysis
        # Concatenate the list of batch tensors into two large tensors on GPU.
        all_train_outputs: torch.Tensor = torch.cat(epoch_train_outputs)
        all_train_labels: torch.Tensor = torch.cat(epoch_train_labels)

        # Perform all calculations on these large tensors on GPU.
        train_predicted_values: torch.Tensor = (all_train_outputs + 1) * (tensor_max_vals - tensor_min_vals) / 2 + tensor_min_vals
        train_real_values: torch.Tensor = (all_train_labels + 1) * (tensor_max_vals - tensor_min_vals) / 2 + tensor_min_vals
        train_error_percentages: torch.Tensor = torch.abs(train_predicted_values - train_real_values) / (train_real_values + 1e-8) * 100

        # Calculate the final mean and then move the small result to the CPU.
        mean_train_errors: torch.Tensor = torch.mean(train_error_percentages, axis=0).cpu().tolist()
        train_open_acc, train_high_acc, train_low_acc, train_close_acc = mean_train_errors

        print("Error Percentages for the outputs: ")
        print(f"- Opening price: {train_open_acc:.2f} %")
        print(f"- High price: {train_high_acc:.2f} %")
        print(f"- Low price: {train_low_acc:.2f} %")
        print(f"- Closing price: {train_close_acc:.2f} %")
        # Track all the individual accuracies for the graph.
        train_all_ep.append([train_open_acc, train_high_acc, train_low_acc, train_close_acc])
        
        # Calculate the validation loss for the scheduler.
        model.eval()
        with torch.no_grad():
            for j, (val_seqs, val_labels) in enumerate(val_loader):
                val_seqs: torch.Tensor
                val_labels: torch.Tensor
                val_seqs, val_labels = val_seqs.to(device), val_labels.to(device)

                val_outputs: torch.Tensor = model(val_seqs)
                val_loss: torch.Tensor = loss_func(val_outputs, val_labels)

                #?val_outputs, val_loss = val_outputs.to(device), val_loss.to(device)
                # Since this is a costly operation, only do it if absolutely needed.

            scheduler.step(val_loss)


    # Helper function to create graphs for the parameters.
    def plot_real_and_predicted_values(
            ax: matplotlib.axes.Axes,
            real_tracker: list[float],
            predicted_tracker: list[float],
            metric_name: str
    ) -> None:
        # Plot the training and test metric at the same graph.
        real_line, = ax.plot(
            real_tracker,
            color = "red",
            label = "Real"
        )
        predicted_line, = ax.plot(
            predicted_tracker,
            color = "blue",
            label = "Predicted"
        )
        # Change the unit of the values to scientific notation.
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
        # Change the attributes of the graph.
        ax.set_xlabel("Date")
        ax.set_ylabel(f"{metric_name}")
        ax.set_title(f"{metric_name}")
        ax.grid(True)
        mplcursors.cursor([real_line, predicted_line])

    # Test loop
    print("\nStarting with testing.")
    model.eval()
    test_ep_tracker: list[list[float]] = []
    test_real_open_tracker: list[float] = []
    test_predicted_open_tracker: list[float] = []
    test_mae_tracker: list[list[float]] = []
    test_rmse_tracker: list[list[float]] = []
    with torch.no_grad():
        for test_seqs, test_labels in test_loader:
            test_seqs: torch.Tensor
            test_labels: torch.Tensor
            test_seqs, test_labels = test_seqs.to(device), test_labels.to(device)

            test_outputs: torch.Tensor = model(test_seqs)

            test_predicted_values: torch.Tensor  = (test_outputs + 1) * (tensor_max_vals - tensor_min_vals) / 2 + tensor_min_vals
            test_real_values: torch.Tensor  = (test_labels + 1) * (tensor_max_vals - tensor_min_vals) / 2 + tensor_min_vals
            test_error_percentages: torch.Tensor = torch.Tensor.abs(test_predicted_values - test_real_values) / (test_real_values + 1e-8) * 100 # Add small epsilon 1e-8 to prevent division by zero.
            test_ep_tracker.extend(test_error_percentages.tolist())

            # Calculate mean absolute error and squared error.
            mae_batch: torch.Tensor = torch.mean(torch.abs(test_predicted_values - test_real_values), axis=0)
            test_mae_tracker.append(mae_batch.tolist())
            rmse_batch: torch.Tensor = torch.sqrt(torch.mean((test_predicted_values - test_real_values)**2, axis=0))
            test_rmse_tracker.append(rmse_batch.tolist())

            test_real_open_tracker.extend(test_real_values[:, 0])
            test_predicted_open_tracker.extend(test_predicted_values[:, 0])


        # Calculate the error percentage, mean absolute error and root mean squared error for all together.
        test_open_ep, test_high_ep, test_low_ep, test_close_ep= np.mean(test_ep_tracker, axis=0)
        test_open_mae, test_high_mae, test_low_mae, test_close_mae = np.mean(test_mae_tracker, axis=0)
        test_open_rmse, test_high_rmse, test_low_rmse, test_close_rmse = np.mean(test_rmse_tracker, axis=0)



    print("\nTest results are for the error percentages (EP), mean absolute eror (MAE) and root mean squared error (RMSE): ")
    print(f"- Opening Price -> EP: {test_open_ep:.2f} % | MAE: {test_open_mae:.2f} | RMSE: {test_open_rmse:.2f} |")
    print(f"- Closing Price -> EP: {test_close_ep:.2f} % | MAE: {test_close_mae:.2f} | RMSE: {test_close_rmse:.2f} |")
    print(f"- Lowest Price  -> EP: {test_low_ep:.2f} % | MAE: {test_low_mae:.2f} | RMSE: {test_low_rmse:.2f} |")
    print(f"- Highest Price -> EP: {test_high_ep:.2f} % | MAE: {test_high_mae:.2f} | RMSE: {test_high_rmse:.2f} |")

    print("\nCreating graphs for training accuracy and test outputs.")
    test_open_fig, (test_result_ax, train_progress_ax) = plt.subplots(2, 1, figsize=(10, 15))
    # Create the twin axes for the right y-axis
    test_result_ax: matplotlib.axes.Axes
    twin_test_open_ax: matplotlib.axes.Axes = test_result_ax.twinx()

    plot_real_and_predicted_values(
        test_result_ax,
        test_real_open_tracker,
        test_predicted_open_tracker, 
        "Opening Price"
    )
    print(len(test_real_open_tracker))
    # Create a numpy array version for slicing.
    test_ep_tracker_np = np.array(test_ep_tracker)
    window_size: int = 10
    # Create a rolling mean (rm) for the window size.
    test_rm_open = np.convolve(test_ep_tracker_np[:, 0], np.ones(window_size)/window_size, mode="valid")
    twin_ep_line, = twin_test_open_ax.plot(
        test_rm_open,
        color = "orange",
        linestyle = "--", # Dashed linestyle
        alpha = 0.8,
        label = "Error Percentage"
    )
    twin_test_open_ax.set_ylabel("Error (%)")
    # We will create a unified legend by getting the handles and labels from the first axis then second.
    test_open_handles, test_open_labels = test_result_ax.get_legend_handles_labels()
    twin_test_open_handles, twin_test_open_labels = twin_test_open_ax.get_legend_handles_labels()
    # Combine them.
    test_all_open_handles = test_open_handles + twin_test_open_handles
    test_all_open_labels = test_open_labels + twin_test_open_labels
    # Call the legend function once on of the axes with the combined handles and labels.
    test_result_ax.legend(test_all_open_handles, test_all_open_labels, loc="upper left")

    # Create the 5.graph for opening & closing price.
    def plot_training_error_percentages(
            ep_ax: matplotlib.axes.Axes,
            first_ep_list: np.ndarray,
            second_ep_list: np.ndarray,
            first_color: str,
            second_color: str,
            first_metric_name : str,
            second_metric_name: str
    ) -> None:
        first_line, = ep_ax.plot(
            first_ep_list,
            color = first_color,
            label = first_metric_name
        )
        second_line, = ep_ax.plot(
            second_ep_list,
            color = second_color,
            label = second_metric_name
        )
        ep_ax.set_xlabel("Epoch")
        ep_ax.set_ylabel("Error (%)")
        ep_ax.set_title(f"Training Error Percentages for {first_metric_name} & {second_metric_name} Price")
        ep_ax.grid(True)
        ep_ax.legend()
        mplcursors.cursor([first_line, second_line])

    # For slicing, turn the list into numpy array first.
    train_all_ep_np: np.ndarray = np.array(train_all_ep)
    train_open_ep_list: np.ndarray = train_all_ep_np[:, 0]
    train_high_ep_list: np.ndarray = train_all_ep_np[:, 1]
    train_low_ep_list: np.ndarray = train_all_ep_np[:, 2]
    train_close_ep_list: np.ndarray = train_all_ep_np[:, 3]

    plot_training_error_percentages(
        train_progress_ax,
        train_open_ep_list,
        train_close_ep_list,
        "green",
        "orange",
        "Opening",
        "Closing"
    )
    plot_training_error_percentages(
        train_progress_ax,
        train_high_ep_list,
        train_low_ep_list,
        "magenta",
        "brown",
        "Highest",
        "Lowest"
    )

    # Adjust the gap between the subplots.
    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    plt.show()