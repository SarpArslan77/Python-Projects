
#TODO Add testing loop at the end.

#! Custom TODO notes:
#TODO AD: Add docstring.
#TODO ATH: Add type hint.
#TODO CTH: Check type hint.
#TODO FTH: Fix the hardcoding.
#TODO HPE: Handle possible error.
#TODO AC: Add comment.

from datetime import datetime
import os
from typing import Any

import numpy as np
from numpy.typing import NDArray
#! import intel_extension_for_pytorch as ipex
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch.optim as optim
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from data_preprocess import create_dataloaders
from graph import plot_graph

def train(
        optimizer_params: dict[str, Any],
        dataloader_params: dict[str, Any],
        train_and_val_params: dict[str, int]
) -> None:
    """
    Trains and validates a Faster R-CNN object detection model.

    This function handles the complete training pipeline, including loading a
    pre-trained model, setting up the optimizer and data loaders, and running
    the training loop. It periodically evaluates the model on a validation set
    using mean Average Precision (mAP), adjusts the learning rate accordingly,
    and saves model checkpoints. Finally, it plots the training history.

    Args:
        optimizer_params (dict[str, Any]): Parameters for the optimizer and scheduler.
        dataloader_params (dict[str, Any]): Parameters for data paths and loaders.
        train_and_val_params (dict[str, int]): Parameters for the training loop,
            like number of epochs and validation frequency.
    """
    
    # Unpack the parameters.
    lr: float = optimizer_params["lr"]
    factor: float = optimizer_params["factor"]
    patience: int = optimizer_params["patience"]
    min_lr: float = optimizer_params["min_lr"]

    model_save_path: str = dataloader_params["model_save_path"] # Not really for the dataloader.
    parent_folder_path: str = dataloader_params["parent_folder_path"]
    images_folder_name: str = dataloader_params["images_folder_name"]
    images_format: str = dataloader_params["images_format"]
    annotations_folder_name: str = dataloader_params["annotations_folder_name"]
    device: torch.device = dataloader_params["device"]
    random_state_seed: int = dataloader_params["random_state_seed"]
    batch_size: int = dataloader_params["batch_size"]
    num_workers: int = dataloader_params["num_workers"]

    num_epochs: int = train_and_val_params["num_epochs"]
    train_progress_print_frequency: int = train_and_val_params["train_progress_print_frequency"]
    val_frequency: int = train_and_val_params["val_frequency"]
    checkpoint_save_frequency: int = train_and_val_params["checkpoint_save_frequency"]


    # Load the pre-trained model.
    model: torchvision.models.detection.faster_rcnn.FasterRCNN = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # JIT-compile the model into optimized kernels.
    #! model = torch.compile(model)

    # Get the number of input features for the classifier.
    in_channels: int = model.roi_heads.box_predictor.cls_score.in_features
    # Define the number of classes: 6 types of defects + 1 background = 7
    num_classes: int = 7 
    # Create a new, untrained classifier head.
    new_head = FastRCNNPredictor(
        in_channels = in_channels,
        num_classes = num_classes
    )
    # Replace the old box_predictor with our new custom one.
    model.roi_heads.box_predictor = new_head

    # Move the final model to the device.
    #! model.to(device)

    # Define an optimizer and scheduler.
    optimizer = optim.Adam(
        params = model.parameters(),
        lr = lr, 
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer = optimizer,
        # Determines the condition under which the lr is reduced.
        mode = "max",  # lr will be reduced when the monitored quantity has stopped increasing for a number of episodes (patience).
        factor = factor, 
        patience = patience, 
        min_lr = min_lr 
    )

    # Apply IPEX optimizations to the model and optimizer. #TODO Fix the IPEX optimization.
    """
    model, optimizer = ipex.optimize( #TODO HPE
        model = model,
        optimizer = optimizer
    )
    optimizer: optim.Adam
    """

    # Create the dataloaders for the training and test loops.
    train_loader, val_loader, test_loader = create_dataloaders( 
        parent_folder_path = parent_folder_path,
        images_folder_name = images_folder_name,
        images_format = images_format,
        annotations_folder_name = annotations_folder_name,
        device = device,
        random_state_seed = random_state_seed,
        batch_size = batch_size,
        num_workers = num_workers
    )

    # Initialize the metric.
    val_metric = MeanAveragePrecision( 
        box_format = "xyxy",
        iou_type = "bbox"
    )
    test_metric = MeanAveragePrecision(
        box_format = "xyxy",
        iou_type = "bbox"
    )

    # Start the training loop.
    print("\nStarting Training!")
    # Create a timestamped folder. #TODO Change the directory creating after the training is done.
    training_start_time = datetime.now()
    trial_folder_name: str = training_start_time.strftime("%Y-%m-%d_%H-%M")
    trial_folder_path: str = os.path.join(model_save_path, trial_folder_name)
    try: # Try to create a folder for the new trial.
        os.makedirs(trial_folder_path)
        print(f"\nDirectory {trial_folder_path} has been created succesfully!")
    except PermissionError:
        print("New directory creation doesn't have the necessary permissions to write to that file.")
    except FileExistsError:
        print("A Directory with an exact name exists.")

    # Tracking parameters for the graph.
    lr_tracker: list[float] = []
    train_overall_loss_tracker: list[float] = []
    train_loss_objectness_tracker: list[float] = []
    train_loss_rpn_box_reg_tracker: list[float] = []
    train_loss_classifier_tracker: list[float] = []
    train_loss_box_reg_tracker: list[float] = []

    val_map_tracker: list[float] = [] # Mean Average Precision.

    for epoch in range(num_epochs): 
        print(f"\nEpoch : {epoch+1} / {num_epochs}")
        # Set the model into training mode.
        epoch_overall_train_losses: list[float] = []
        epoch_loss_objectness_losses: list[float] = []
        epoch_loss_rpn_box_reg_losses: list[float] = []
        epoch_loss_classifier_losses: list[float] = []
        epoch_loss_box_reg_losses: list[float] = []

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

            # Sum and seperate the losses.
            batch_train_loss: torch.Tensor = sum(loss for loss in train_outputs.values())
            epoch_overall_train_losses.append(batch_train_loss.item())

            epoch_loss_objectness_losses.append(train_outputs["loss_objectness"])
            epoch_loss_rpn_box_reg_losses.append(train_outputs["loss_rpn_box_reg"])
            epoch_loss_classifier_losses.append(train_outputs["loss_classifier"])
            epoch_loss_box_reg_losses.append(train_outputs["loss_box_reg"])

            # Backward pass and optimization.
            optimizer.zero_grad()
            batch_train_loss.backward()
            optimizer.step()

        # Calculate the average epoch loss for training.
        avg_epoch_train_loss: float = sum(epoch_overall_train_losses) / len(epoch_overall_train_losses)
        train_overall_loss_tracker.append(avg_epoch_train_loss)

        avg_epoch_loss_objectness_loss: float = sum(epoch_loss_objectness_losses) / len(epoch_loss_objectness_losses)
        train_loss_objectness_tracker.append(avg_epoch_loss_objectness_loss)
        avg_epoch_loss_rpn_box_reg_loss: float = sum(epoch_loss_rpn_box_reg_losses) / len(epoch_loss_rpn_box_reg_losses)
        train_loss_rpn_box_reg_tracker.append(avg_epoch_loss_rpn_box_reg_loss)
        avg_epoch_loss_classifier_loss: float = sum(epoch_loss_classifier_losses) / len(epoch_loss_classifier_losses)
        train_loss_classifier_tracker.append(avg_epoch_loss_classifier_loss)
        avg_epoch_loss_box_reg_loss: float = sum(epoch_loss_box_reg_losses) / len(epoch_loss_box_reg_losses)
        train_loss_box_reg_tracker.append(avg_epoch_loss_box_reg_loss)

        if (epoch + 1) % train_progress_print_frequency == 0:
            print(f"   Overall Train Loss: {avg_epoch_train_loss:.4f}")
            print(f"   Objectness Loss: {avg_epoch_loss_objectness_loss:.4f}")
            print(f"   Proposal Bounding Box Loss: {avg_epoch_loss_rpn_box_reg_loss:.4f}")
            print(f"   Final Classification Loss: {avg_epoch_loss_classifier_loss:.4f}")
            print(f"   Final Bounding Box Loss: {avg_epoch_loss_box_reg_loss:.4f}")

        # Validation phase.
        if (epoch + 1) % val_frequency == 0: 
            model.eval()
            with torch.no_grad():
                for val_images, val_annotations in val_loader:
                    val_images: torch.Tensor
                    val_annotations: list[dict[str, torch.Tensor]]

                    # Get the model predictions.
                    val_predictions: dict[str, torch.Tensor] = model(val_images)

                    # Update the metric with the predictions and ground truths for the current batch.
                    val_metric.update(
                        preds = val_predictions,
                        target = val_annotations
                    )

            # Compute the mAP over all validation batches.
            val_results: dict[str, torch.Tensor] = val_metric.compute()
            val_map: float = val_results["map"].item() # Get the main mAP score.
            val_map_tracker.append(val_map)

            # Reset the metric for the next validation run.
            val_metric.reset()

            # Step the scheduler according to mAP value.
            scheduler.step(val_map)

            # Track the new learning rate.
            new_lr: float = optimizer.param_groups[0]["lr"]
            lr_tracker.append(new_lr)

            print(f"  Avg Validation mAP: {val_map:.4f} | New LR: {new_lr}")

        # Save the model every specified epoch count.
        if (epoch + 1) % checkpoint_save_frequency == 0: 
            checkpoint_path: str = os.path.join(trial_folder_path, f"model_epoch_{epoch+1}.pth")
            
            # Create a dictionary to hold all the states.
            checkpoint: dict[str, Any] = {
                "epoch" : epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss_tracker": train_overall_loss_tracker,
                "val_map_tracker": val_map_tracker,
                "lr_tracker": lr_tracker
            }
            # Save the checkpoint dictionary.
            torch.save(
                obj = checkpoint,
                f = checkpoint_path
            )
            
            print(f"\nCheckpoint saved at epoch {epoch+1} to 'Trials' folder with the name model_epoch_{epoch+1}.pth !")

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
        train_loss_history = np.array(train_overall_loss_tracker),
        val_map_history = np.array(val_map_tracker),
        trial_folder_path = trial_folder_path,
        show_graph = True
    )

    # Test phase.
    print("\nStarting with testing...")
    # The test parameters are:
    #   1) Mean Average Precision (mAP): 
    #   2) Per-Class Performance: 
    model.eval()
    with torch.no_grad():
        # Trackers for the test parameters.
        true_labels: list[int] = []
        predicted_labels: list[int] = []

        for test_images, test_annotations in test_loader:
            test_images: torch.Tensor
            test_annotations: list[dict[str, torch.Tensor]]

            # Add the true labels to the tracker.
            [true_labels.extend(torch.Tensor.tolist(test_annotation["labels"])) for test_annotation in test_annotations]

            # Get model predictions.
            test_predictions: dict[str, torch.Tensor] = model(test_images)

            # Add the predicted labels to the tracker.
            [predicted_labels.extend(torch.Tensor.tolist(test_prediction["labels"])) for test_prediction in test_predictions]

            # Update the metric with the predictions and ground truths for the current batch.
            test_metric.update(
                preds = test_predictions,
                target = test_annotations
            )

        # 1) Compute the mAP.
        test_results: dict[str, torch.Tensor] = test_metric.compute()
        test_map: float = test_results["map"].item() # Get the main mAP score.

        # Reset the test metric, just in case.
        test_metric.reset()

        print(f"\n Avg Test mAP: {test_map:.4f}")

        # 2) Compute the Per-Class Performance.
        y_true: NDArray = np.array(true_labels, dtype=torch.int8)
        y_predicted: NDArray = np.array(predicted_labels, dtype=torch.int8)

        # Create an classification report.
        classification_report_summary = classification_report(
            y_true,
            y_predicted,
            output_dict = True
        )
        # Convert the report to DataFrame and e





    



if __name__ == "__main__":
    
    # Prepare the parameters.
    # Check whether the Intel XE GPU is available, if not use the CPU.
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    print(f"\nActive device: {device}")

    # Pack the parameters to seperate dictionaries.
    optimizer_params: dict[str, Any] = {
        "lr": 1e-4,
        "factor": 5e-1,
        "patience": 5,
        "min_lr": 1e-5
    }
    dataloader_params: dict[str, Any] = {
        "model_save_path": "C:/Users/Besitzer/Desktop/Python/AI Projects/Convolutional Neural Networks/PCB Defects/Trials",
        "parent_folder_path": "C:/Users/Besitzer/Desktop/Python/AI Projects/Convolutional Neural Networks/PCB Defects/PCB_DATASET/",
        "images_folder_name": "images",
        "images_format": "jpg",
        "annotations_folder_name": "Annotations",
        "device": device,
        "random_state_seed": 69,
        "batch_size": 8,
        "num_workers": 0 # Must be set to Null, so preloading the images can work.
    }
    train_and_val_params: dict[str, int] = {
        "num_epochs": 20,
        "train_progress_print_frequency": 1,
        "val_frequency": 20,
        "checkpoint_save_frequency": 5
    }

    # Initialize the training method.
    train(
        optimizer_params,
        dataloader_params,
        train_and_val_params
    )


