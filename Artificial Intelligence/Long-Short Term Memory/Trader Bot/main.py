
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
import mplcursors
import intel_extension_for_pytorch as ipex

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
    input_size: int = 18
    hidden_size: int = 50
    num_layers: int = 2 # Number of stacked LSTM-layers
    output_size: int = 4
    # Training variables.
    num_epochs: int = 50
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
        mean_train_error_percentages: torch.Tensor = torch.mean(train_error_percentages, axis=0).cpu().tolist()
        train_open_ep, train_high_ep, train_low_ep, train_close_ep = mean_train_error_percentages

        print("Closing Price error percentages (EP) and losses: ")
        print(f"- Training EP: {train_open_ep:.4f} %")
        # Track all the individual accuracies for the graph.
        train_all_ep.append([train_open_ep, train_high_ep, train_low_ep, train_close_ep])
        
        # Calculate the validation loss for the scheduler.
        model.eval()
        with torch.no_grad():
            val_loss_tracker: list[torch.Tensor] = []
            val_outputs_tracker: list[torch.Tensor] = []
            val_labels_tracker: list[torch.Tensor] = []
            for j, (val_seqs, val_labels) in enumerate(val_loader):
                val_seqs: torch.Tensor
                val_labels: torch.Tensor
                val_seqs, val_labels = val_seqs.to(device), val_labels.to(device)

                val_outputs: torch.Tensor = model(val_seqs)
                val_loss: float = loss_func(val_outputs, val_labels)

                #?val_outputs, val_loss = val_outputs.to(device), val_loss.to(device)
                # Since this is a costly operation, only do it if absolutely needed.

                # Track the validation loss for scheduler and outputs & labels for e.p. calculation.
                val_loss_tracker.append(val_loss)
                val_outputs_tracker.append(val_outputs)
                val_labels_tracker.append(val_labels)

            #  Stacking creates a 1D Tensor from the list of 0D Tensors.
            all_val_loss: torch.Tensor = torch.stack(val_loss_tracker)
            avg_val_loss: float = torch.mean(all_val_loss)
            scheduler.step(avg_val_loss)

            # Repeat the same e.p. calculation as in training loop.
            all_val_outputs: torch.Tensor = torch.cat(val_outputs_tracker)
            all_val_labels: torch.Tensor = torch.cat(val_labels_tracker)

            val_predicted_values: torch.Tensor = (all_val_outputs + 1) * (tensor_max_vals - tensor_min_vals) / 2 + tensor_min_vals
            val_real_values: torch.Tensor = (all_val_labels + 1) * (tensor_max_vals - tensor_min_vals) / 2 + tensor_min_vals
            val_error_percentages: torch.Tensor = torch.abs(val_predicted_values - val_real_values) / (val_real_values + 1e-8) * 100

            mean_val_error_percentages: torch.Tensor = torch.mean(val_error_percentages, axis=0).cpu().tolist()
            val_open_ep, val_high_ep, val_low_ep, val_close_ep = mean_val_error_percentages

            print(f"- Validation EP: {val_close_ep:.4f} % | Validation Loss: {avg_val_loss:.4f}")


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
    # These will store the final CPU-based numbers for plotting.
    test_real_tracker: list[float] = []
    test_predicted_tracker: list[float] = []
    with torch.no_grad():
        # Accumulate the raw tensors on the GPU.
        epoch_test_outputs: list[torch.Tensor] = []
        epoch_test_labels: list[torch.Tensor] = []
        for test_seqs, test_labels in test_loader:
            test_seqs: torch.Tensor
            test_labels: torch.Tensor
            test_seqs, test_labels = test_seqs.to(device), test_labels.to(device)

            test_outputs: torch.Tensor = model(test_seqs)

            # Append the raw GPU tensors to the temporary lists.
            epoch_test_outputs.append(test_outputs)
            epoch_test_labels.append(test_labels)
        
        # Concatenate all batch results into single, large GPU tensors.
        all_test_outputs: torch.Tensor = torch.cat(epoch_test_outputs)
        all_test_labels: torch.Tensor = torch.cat(epoch_test_labels)

        # Perform denormalization on the large GPU tensors.
        test_predicted_values: torch.Tensor = (all_test_outputs + 1) * (tensor_max_vals - tensor_min_vals) / 2 + tensor_min_vals
        test_real_values: torch.Tensor = (all_test_labels + 1) * (tensor_max_vals - tensor_min_vals) / 2 + tensor_min_vals

        # Convert the required data to CPU lists for Matplotlib
        #! (only 'close' is choosen rn for plotting)
        test_real_tracker.extend(test_real_values[:, 3].cpu().tolist())
        test_predicted_tracker.extend(test_predicted_values[:, 3].cpu().tolist())

        # Calculate the other metrics and move them to CPU for numpy.
        test_error_percentages: torch.Tensor = torch.abs(test_predicted_values - test_real_values) / (test_real_values + 1e-8) * 100
        test_ep_tracker: list[list[float]] = test_error_percentages.cpu().tolist()

        test_mae: list[float] = torch.mean(torch.abs(test_predicted_values - test_real_values), axis=0).cpu().tolist()
        test_rmse: list[float] = torch.sqrt(torch.mean((test_predicted_values - test_real_values)**2, axis=0)).cpu().tolist()

        test_open_ep, test_high_ep, test_low_ep, test_close_ep = np.mean(test_ep_tracker, axis=0)
        test_open_mae, test_high_mae, test_low_mae, test_close_mae = test_mae
        test_open_rmse, test_high_rmse, test_low_rmse, test_close_rmse = test_rmse

    print("\nTest results are for the error percentages (EP), mean absolute eror (MAE) and root mean squared error (RMSE): ")
    print(f"- Closing Price -> EP: {test_close_ep:.4f} % | MAE: {test_close_mae:.4f} | RMSE: {test_close_rmse:.4f} |")

    print("\nCreating graphs for training accuracy and test outputs.")
    test_fig, (test_result_ax, train_progress_ax) = plt.subplots(2, 1, figsize=(10, 15))
    # Create the twin axes for the right y-axis
    test_result_ax: matplotlib.axes.Axes
    twin_test_ax: matplotlib.axes.Axes = test_result_ax.twinx()

    plot_real_and_predicted_values(
        test_result_ax,
        test_real_tracker,
        test_predicted_tracker, 
        "Opening Price"
    )
    print(len(test_real_tracker))
    # Create a numpy array version for slicing.
    test_ep_tracker_np = np.array(test_ep_tracker)
    window_size: int = 10
    # Create a rolling mean (rm) for the window size.
    test_rm = np.convolve(test_ep_tracker_np[:, 3], np.ones(window_size)/window_size, mode="valid")
    twin_ep_line, = twin_test_ax.plot(
        test_rm,
        color = "orange",
        linestyle = "--", # Dashed linestyle
        alpha = 0.8,
        label = "Error Percentage"
    )
    twin_test_ax.set_ylabel("Error (%)")
    # We will create a unified legend by getting the handles and labels from the first axis then second.
    test_handles, test_labels = test_result_ax.get_legend_handles_labels()
    twin_test_handles, twin_test_labels = twin_test_ax.get_legend_handles_labels()
    # Combine them.
    test_all_handles = test_handles + twin_test_handles
    test_all_labels = test_labels + twin_test_labels
    # Call the legend function once on of the axes with the combined handles and labels.
    test_result_ax.legend(test_all_handles, test_all_labels, loc="upper left")

    # Create the 5.graph for opening & closing price.
    def plot_training_error_percentages(
            ep_ax: matplotlib.axes.Axes,
            first_ep_list: np.ndarray,
            first_color: str,
            first_metric_name : str,
    ) -> None:
        first_line, = ep_ax.plot(
            first_ep_list,
            color = first_color,
            label = first_metric_name
        )
        ep_ax.set_xlabel("Epoch")
        ep_ax.set_ylabel("Error (%)")
        ep_ax.set_title(f"Training Error Percentages for {first_metric_name} Price")
        ep_ax.grid(True)
        ep_ax.legend()
        mplcursors.cursor(first_line)

    # For slicing, turn the list into numpy array first.
    train_all_ep_np: np.ndarray = np.array(train_all_ep)
    train_open_ep_list: np.ndarray = train_all_ep_np[:, 0]
    train_high_ep_list: np.ndarray = train_all_ep_np[:, 1]
    train_low_ep_list: np.ndarray = train_all_ep_np[:, 2]
    train_close_ep_list: np.ndarray = train_all_ep_np[:, 3]

    plot_training_error_percentages(
        train_progress_ax,
        train_close_ep_list,
        "orange",
        "Opening",
    )

    # Adjust the gap between the subplots.
    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    plt.show()