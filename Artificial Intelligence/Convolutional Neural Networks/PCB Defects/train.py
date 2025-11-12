
#! Custom TODO notes:
#TODO AD: Add docstring.
#TODO ATH: Add type hint.
#TODO CTH: Check type hint.
#TODO FTH: Fix the hardcoding.
#TODO HPE: Handle possible error.
#TODO AC: Add comment.

from datetime import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
#! import intel_extension_for_pytorch as ipex
import torch
import torch.optim as optim
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from data_preprocess import create_dataloaders
from graph import plot_graph

if __name__ == "__main__":

    # Load the pre-trained model.
    model: torchvision.models.detection.faster_rcnn.FasterRCNN = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Get the number of input features for the classifier.
    in_channels: int = model.roi_heads.box_predictor.cls_score.in_features
    # Define the number of classes: 6 types of defects + 1 background = 7
    num_classes: int = 7 #TODO FTH
    # Create a new, untrained classifier head.
    new_head = FastRCNNPredictor(
        in_channels = in_channels,
        num_classes = num_classes
    )
    # Replace the old box_predictor with our new custom one.
    model.roi_heads.box_predictor = new_head

    # Check whether the Intel XE GPU is available, if not use the CPU.
    """device = torch.device("xpu" if torch.xpu.is_available() else "cpu")"""
    device = torch.device("cpu")
    print(f"\nActive device: {device}")

    # Move the final model to the device.
    #! model.to(device)

    # Define an optimizer and scheduler.
    optimizer = optim.Adam(
        params = model.parameters(),
        lr = 1e-3, #TODO FTH
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer = optimizer,
        mode = "min", #TODO FTH
        factor = 5e-1, #TODO FTH
        patience = 5, #TODO FTH
        min_lr = 1e-5 #TODO FTH
    )

    """# Apply IPEX optimizations to the model and optimizer. #TODO Fix the IPEX optimization.
    model, optimizer = ipex.optimize( #TODO HPE
        model = model,
        optimizer = optimizer
    )
    optimizer: optim.Adam"""

    parent_folder: str = "C:/Users/Besitzer/Desktop/Python/AI Projects/Convolutional Neural Networks/PCB Defects/PCB_DATASET/"
    # Create the dataloaders for the training and test loops.
    train_loader, val_loader, test_loader = create_dataloaders( #TODO FTH
        parent_folder = parent_folder,
        images_folder_name = "images",
        images_format = "jpg",
        annotations_folder_name = "Annotations",
        random_state_seed = 69,
        batch_size = 16,
        num_workers = os.cpu_count() // 2
    )

    # Start the training loop.
    print("\nStarting Training!")
    # Create a timestamped folder.
    training_start_time = datetime.now()
    trial_folder_name: str = training_start_time.strftime("%Y-%m-%d_%H-%M")
    MODEL_SAVE_PATH: str = r"C:/Users/Besitzer/Desktop/Python/AI Projects/Convolutional Neural Networks/PCB Defects/Trials"
    trial_folder_path: str = os.path.join(MODEL_SAVE_PATH, trial_folder_name)
    try: # Try to create a folder for the new trial.
        os.makedirs(trial_folder_path)
        print(f"\nDirectory {trial_folder_path} has been created succesfully!")
    except PermissionError:
        print("New directory creation doesn't have the necessary permissions to write to that file.")
    except FileExistsError:
        print("A Directory with an exact name exists.")

    # Tracking parameters for the graph.
    lr_tracker: list[float] = []
    train_loss_tracker: list[float] = []
    val_loss_tracker: list[float] = []

    num_epochs = 500 #TODO FTH
    for epoch in range(num_epochs): 
        print(f"\nEpoch : {epoch+1} / {num_epochs}")
        # Set the model into training mode.
        epoch_train_losses: list[float] = []
        model.train()

        for i, (train_images, train_annotations) in enumerate(train_loader):
            train_images: torch.Tensor
            train_annotations: list[dict[str, torch.Tensor]]
            print(i)

            # Move the images and annotations to the device first.
            #! train_images = train_images.to(device)
            """train_annotations: list[dict[str, torch.Tensor]] = [
                {train_key: train_value for train_key, train_value in train_annotation_dicts.items()} for train_annotation_dicts in train_annotations
            ] #!  train_value.to(device)"""

            # Forward pass.
            train_outputs: dict[str, torch.Tensor] = model(train_images, train_annotations)
            # Sum the losses.
            batch_train_loss: torch.Tensor = sum(loss for loss in train_outputs.values())
            epoch_train_losses.append(batch_train_loss.item())

            # Backward pass and optimization.
            optimizer.zero_grad()
            batch_train_loss.backward()
            optimizer.step()

        # Calculate the average epoch loss for training.
        avg_epoch_train_loss: float = sum(epoch_train_losses) / len(epoch_train_losses)
        train_loss_tracker.append(avg_epoch_train_loss)

        # Validation phase.
        if (epoch + 1) % 10 == 0: #TODO FTH
            epoch_val_losses: list[float] = []
            model.eval()
            with torch.no_grad():
                for val_images, val_annotations in val_loader:
                    val_images: torch.Tensor
                    val_annotations: list[dict[str, torch.Tensor]]

                    #! val_images = val_images.to(device)
                    val_annotations: list[dict[str, torch.Tensor]] = [
                        {val_key: val_value for val_key, val_value in val_annotation_dicts.items()} for val_annotation_dicts in val_annotations
                    ] #! val_value.to(device)

                    # Temporarily set the model to training mode to enable loss calculation.
                    model.train()
                    val_outputs: dict[str, torch.Tensor] = model(val_images, val_annotations)
                    # Set the model back to validation mode.
                    model.eval()

                    batch_val_loss: torch.Tensor = sum(loss for loss in val_outputs.values())
                    epoch_val_losses.append(batch_val_loss.item())

            # Step the scheduler with the validation loss.
            avg_epoch_val_loss: float = sum(epoch_val_losses) / len(epoch_val_losses)
            scheduler.step(avg_epoch_val_loss)

            # Track the average epoch validation loss and new learning rate.
            val_loss_tracker.append(avg_epoch_val_loss)
            new_lr: float = optimizer.param_groups[0]["lr"]
            lr_tracker.append(new_lr)

        print(f"  Avg Train Loss: {avg_epoch_train_loss:.4f} | Avg Validation Loss: {avg_epoch_val_loss:.4f} | New LR: {new_lr}")

    # Calculate the training time.
    training_end_time = datetime.now() 
    training_elapsed_seconds: float = (training_end_time - training_start_time).total_seconds()
    training_elapsed_minutes, training_seconds = divmod(training_elapsed_seconds, 60)
    training_hours, training_minutes = divmod(training_elapsed_minutes, 60)
    # Pack them all in a tuple for the graph.
    training_time: tuple[float, float, float] = (training_hours, training_minutes, training_seconds)
    print(f"\nFinished Training!\nTraining Time: {int(training_hours):02d}:{int(training_minutes):02d}:{int(training_seconds):02d}")

    # Create a graph for the trial.
    print("\nCreating and saving a graph for the loss and learning rate progress...")
    plot_graph(
        num_epochs = num_epochs,
        training_time = training_time,
        train_loss_history = np.array(train_loss_tracker),
        val_loss_history = np.array(val_loss_tracker),
        lr_history = np.array(lr_tracker)
    )
    try: # Try to create a graph for the loss and learning rate variables.
        graph_path: str = os.path.join(trial_folder_path, "loss_and_lr_progress.png") 
        plt.savefig(graph_path)
        plt.show()
    except PermissionError:
        print("The Graph doesn't have the necessary permissions to write to that file...")
    except:
        print("The saving of ratio graph has failed due to an unknown error...")

