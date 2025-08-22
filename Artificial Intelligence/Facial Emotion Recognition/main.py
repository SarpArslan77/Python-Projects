
#! Fix the new custom dataset, it broke the code.
#TODO Create a custom dataloader, in order to use the data augmentation only on disgust.
#TODO Try to optimize the training, make it faster

import torch
from sklearn.metrics import classification_report, confusion_matrix
import intel_extension_for_pytorch as ipex
import matplotlib.pyplot as plt
import mplcursors
import os
from datetime import datetime
import numpy as np
import copy
import seaborn as sns

from data import (
    test_loader,
    train_loader,
    val_loader,
    BATCH_SIZE,
)
from cnn import (
    device,
    model,
    criterion,
    optimizer,
    scheduler,
    STARTING_LEARNING_RATE,
)

# Hyperparameters
NUM_EPOCHS: int = 10

# Apply IPEX optimizations to the model and optimizer
model, optimizer = ipex.optimize(model, optimizer=optimizer)

# Training Loop
print("\nStarting Training!")
training_start_time = datetime.now()
# Tracking parameters for the graph.
train_acc_tracker: list[float] = []
val_acc_tracker: list[float] = []
train_loss_tracker: list[float] = []
val_loss_tracker: list[float] = []
lr_tracker: list[float] = []
# Track the best model with the lowest validation loss.
lowest_val_loss: float = np.inf
best_model_parameters = np.nan
# Stores the epoch, val_acc and val_loss for the best model, to be marked in the graph.
best_model_val_values: list = [np.nan, np.nan, np.nan] # [epoch, val_acc, val_loss]
for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch : {epoch+1}/{NUM_EPOCHS}")
    model.train()
    running_loss: float = 0.0 # Tracking loss for optimizing the learning rate.
    n_correct: int = 0
    n_total: int = 0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Calculate the accuracy.
        _, predicted = torch.max(outputs.data, 1)
        n_total += labels.size(0)
        # All tensors have to be on the same device.
        n_correct += (predicted.cpu() == labels.cpu()).sum().item()

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the loss.
        running_loss += loss.item() * images.size(0)

    train_acc: float = 100.0 * n_correct / n_total
    epoch_train_loss: float = running_loss / len(train_loader.dataset)
    print(f"    Train accuracy : {train_acc:.2f}% | Train Loss: {epoch_train_loss:.4f}")
    train_acc_tracker.append(train_acc)
    train_loss_tracker.append(epoch_train_loss)

    # Validation Phase
    model.eval()
    val_loss: float = 0.0
    val_correct: int = 0
    val_total: int = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted.cpu() == labels.cpu()).sum().item()
    
    val_acc: float = 100.0 * val_correct / val_total
    epoch_val_loss: float = val_loss / len(val_loader.dataset)
    print(f"    Validation accuracy : {val_acc:.2f}% | Val Loss: {epoch_val_loss:.4f}")
    val_acc_tracker.append(val_acc)
    val_loss_tracker.append(epoch_val_loss)

    # Save the learning rate from the optimizer.
    new_lr: float = optimizer.param_groups[0]["lr"]
    print(f"    Learning Rate: {new_lr}")
    lr_tracker.append(new_lr)

    # Update the best model and validation loss if the new model is better.
    if epoch_val_loss < lowest_val_loss:
        lowest_val_loss = epoch_val_loss
        best_model_parameters = copy.deepcopy(model.state_dict())
        best_model_val_values = [epoch, val_acc, epoch_val_loss]

    # Step the scheduler with validation loss
    scheduler.step(epoch_val_loss)
print("\nFinished Training.")
# Calculate the training time.
training_end_time = datetime.now()
elapsed_seconds = (training_end_time - training_start_time).total_seconds()
seconds = elapsed_seconds % 60
minutes = (elapsed_seconds % 3600) // 60
hours = elapsed_seconds // 3600
print(f"\nTotal Training Time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

# Create a graph for tracking the accuracy, loss and learning rate.
fig, ((ax_acc, ax_loss) , (ax_lr, ax_cm)) = plt.subplots(2, 2, figsize=(10, 10))

# Unpack the best model values for the graph marking.
best_model_epoch = best_model_val_values[0]
best_model_acc = best_model_val_values[1]
best_model_loss = best_model_val_values[2]

# First create for the both accuracy.
# Training accuracy
line_train_acc, = ax_acc.plot(
    train_acc_tracker,
    color = "red",
)
# Validation accuracy
line_val_acc, = ax_acc.plot(
    val_acc_tracker,
    color = "blue"
)

# Add the star marking for the best model.
ax_acc.plot(
    best_model_epoch,
    best_model_acc,
    marker = "*", 
    # marker argument tells the .plot function, which automatically draws lines,
    #   to draw only one point in the graph.
    color = "gold",
    markersize = 12,
    markeredgecolor = "black",
    markeredgewidth = 1,
    label = "Best Model Accuracy"
)

# Accuracy plot settings
ax_acc.set_title("Training & Validation Accuracy Progression")
ax_acc.set_xlabel("Epoch")
ax_acc.set_ylabel("Accuracy")
ax_acc.grid(True)
ax_acc.legend(
    [line_train_acc, line_val_acc], 
    ["Training Accuracy", "Validation Accuracy"], 
    loc = "lower right",
)

# Add interactive cursor
mplcursors.cursor(line_train_acc)
mplcursors.cursor(line_val_acc)

# Repeat for loss.
line_train_loss, = ax_loss.plot(
    train_loss_tracker,
    color = "red",
    linestyle="--",
)
line_val_loss, = ax_loss.plot(
    val_loss_tracker,
    color = "blue",
    linestyle = "--",
)

ax_loss.plot(
    best_model_epoch,
    best_model_loss,
    marker = "*",
    color = "gold",
    markersize = 12,
    markeredgecolor = "black",
    markeredgewidth = 1,
    label = "Best Model Loss"
)

ax_loss.set_title("Training & Validation Loss Progression")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Loss")
ax_loss.grid(True)
ax_loss.legend(
    [line_train_loss, line_val_loss], 
    ["Training Loss", "Validation Loss"], 
    loc = "upper right",
)

mplcursors.cursor(line_train_loss)
mplcursors.cursor(line_val_loss)

# Repeat for learning rate.
line_lr, = ax_lr.plot(
    lr_tracker,
    color = "orange",
)

ax_lr.set_title("Learning Rate Progression")
ax_lr.set_xlabel("Epoch")
ax_lr.set_ylabel("Learning Rate")
ax_lr.grid(True)
ax_lr.legend(
    [line_lr],
    ["Learning Rate"],
    loc = "upper right",
)

mplcursors.cursor(line_lr)

#! Unmark these part, whenever you not want to save the model parameters and accuracy graph.

# Create a Timestamped directory for saving

# Get the current date and time.
dir_creation_time = datetime.now()

# Format the timestamp for the directory name: e.g. "12/08/2025_10:30"
new_dir_name: str = dir_creation_time.strftime("%d-%m-%Y_%H-%M")

# Create the new directory path.
MODEL_SAVE_PATH: str = r"C:/Users/Besitzer/Desktop/Python/AI Projects/Facial Emotion Recognition/Runs/"
new_dir_path: str = os.path.join(MODEL_SAVE_PATH, new_dir_name)

try: # Create a new directory.
    os.makedirs(new_dir_path,)
    print(f"\nDirectory {new_dir_name} created succesfully.")
except PermissionError:
    print("New directory creation doesn't have the necessary permissions to write to that file.")
except FileExistsError:
    print("A Directory with an exact name exists.")
else:
    # This else-block only runs if the directory was created successfully.

    # Define the file paths within the new directory.
    best_model_parameters_path = os.path.join(new_dir_path, "best_model_parameters.pth")
    last_model_parameters_path = os.path.join(new_dir_path, "last_model_parameters.pth")
    acc_graph_path = os.path.join(new_dir_path, "acc_loss_lr_graph.png")

    try: # Save the model's best state_dict.
        torch.save(best_model_parameters, best_model_parameters_path)
    except PermissionError:
        print("Model parameter saving doesn't have the necessary permissions to write to that file.")
    try: # Save the model's last state.
        torch.save(model.state_dict(), last_model_parameters_path)
    except PermissionError:
        print("Model parameter saving doesn't have the necessary permissions to write to that file.")

    try: # Save the Matplotlib.graph.
        plt.savefig(acc_graph_path)
    except PermissionError:
        print("The Graph doesn't have the necessary permissions to write to that file.")

emotion_labels: list[str] = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
# Track all the model predictions and correct labels.
all_predictions: list = []
all_labels: list = []

# Testing loop
model.eval()
print("\nStarting Testing!")
with torch.no_grad():
    n_correct: int = 0
    n_samples: int = 0
    for j, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        # Max returns (value, index)
        _, predicted = torch.max(outputs.data, 1)

        # Append the predictions and labels from this batch to our master lists.
        #       We move them to the CPU to use with numpy/sklearn.
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        n_samples += labels.cpu().size(0)
        n_correct += (predicted.cpu() == labels.cpu()).sum().item()

    accuracy: float = n_correct / n_samples * 100.0
    print(f"\nAccuracy of the network is: {accuracy:.2f} %")

    # This report includes precision, recall and f1-score for each class.
    classification_report_summary = classification_report(
        all_labels,
        all_predictions,
        target_names=emotion_labels,
    )
    print("\nClassification Report:\n")
    print(classification_report_summary)

    # Create the confusion matrix, to see which emotions are between each are confused by the model.
    confusion_matrix_report = confusion_matrix(all_labels, all_predictions)
    # Add the cm to the graphs.
    sns.heatmap(
        confusion_matrix_report,
        annot = True, # Display the number in each cell.
        fmt = "d", # Format the numbers as integers.
        cmap = "Blues", # Use Red-And-Blue colormap.
        xticklabels = emotion_labels,
        yticklabels = emotion_labels,
        ax = ax_cm,
    )

    ax_cm.set_title("Confusion Matrix")
    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("True Label")
    # Rotate the y-achsis labels to be horizontal
    ax_cm.set_yticklabels(ax_cm.get_yticklabels(), rotation="horizontal")

    # Automatically adjust the gaps between graphs, to prevent text overlapping.
    plt.tight_layout(pad=3.0)
    # plt.show has to be here, because it resets the canvas after its closed.
    plt.show()

    # Track all the layers for the summary.
    layer_rankings: list[str] = []
    for name, module in model.named_modules():
        layer_type = type(module).__name__
        layer_rankings.append(layer_type)
    # Get the name of the optimizer, scheduler and loss function used for the cnn.
    optimizer_type = type(optimizer).__name__
    loss_function_type = type(criterion).__name__
    scheduler_type = type(scheduler).__name__

    try: # Save the report as .txt file to the newly created directory.
        report_path: str = os.path.join(new_dir_path, "summary.txt")
        with open(report_path, "w") as f:
            f.write("Classification Report:\n")
            f.write("\n")
            f.write(classification_report_summary)
            f.write(f"\nTotal Training Time: {int(hours)}h {int(minutes)}m {int(seconds)}s, on device: {device}\n")
            f.write("\nParameters are:\n")
            f.write(f" - Number of epochs: {NUM_EPOCHS}\n")
            f.write(f" - Batch size: {BATCH_SIZE}\n")
            f.write(f" - Started with the learning rate {STARTING_LEARNING_RATE} and ended with {new_lr}\n")
            f.write(f" - Best performing model at epoch {best_model_epoch} with accuracy {best_model_acc:.2f}% and loss {best_model_loss:.4f}\n")
            f.write("\n")
            # Write the layer structure.
            layer_count: int = 1
            for i, layer_name in enumerate(layer_rankings):
                # First layer is the module name, write it with its properties.
                if i == 0:
                    f.write(f"Class {layer_name}:")
                    f.write(f"\n - Optimizer: {optimizer_type}")
                    f.write(f"\n - Loss Function: {loss_function_type}")
                    f.write(f"\n - Scheduler: {scheduler_type}")
                    f.write("\n")
                # Sequential marks the beginning of a new layer.
                elif layer_name == "Sequential":
                    f.write(f"\nLayer {layer_count}:")
                    layer_count += 1
                    f.write("\n ")
                else: 
                    f.write(f" -> {layer_name}")
            f.write("\n\n")

        print("\nReport also saved in the new directory.")
    except FileNotFoundError:
        print("File couldn't be found, so the saving of the report was unsuccesful.")

# Camera implementation
"""
# Create the VideoCapture object.
camera = cv2.VideoCapture(0) # 0 = default camera

# Main loop to get continiues data.
while True:
    ret, frame = camera.read()
    # ret: is a boolean value True if the frame was red successfully
    # frame: actual image captured

    # Display the frame.
    cv2.imshow("PC-Camera", frame)

    # End the program if 'q' pressed.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
"""
