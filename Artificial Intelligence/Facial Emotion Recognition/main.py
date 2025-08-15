


import torch
from sklearn.metrics import classification_report
import intel_extension_for_pytorch as ipex
import matplotlib.pyplot as plt
import mplcursors
import os
from datetime import datetime

from data import (
    test_loader,
    train_loader,
    val_loader,
)
from cnn import (
    device,
    model,
    criterion,
    optimizer,
    scheduler,
)

# Hyperparameters
NUM_EPOCHS: int = 2

# Apply IPEX optimizations to the model and optimizer
model, optimizer = ipex.optimize(model, optimizer=optimizer)

# Training Loop
print("\nStarting Training!")
train_acc_tracker: list[float] = []
val_acc_tracker: list[float] = []
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

    # Step the scheduler with validation loss
    scheduler.step(epoch_val_loss)
print("\nFinished Training.")

# Create a graph for tracking the training accuracy and learning_rate
fig, ax = plt.subplots()

# Training accuracy
line_train, = ax.plot(
    train_acc_tracker,
    color = "red",
)
# Validation accuracy
line_val, = ax.plot(
    val_acc_tracker,
    color = "blue"
)

# Plot settings
ax.set_title("Training & Validation Accuracy Progression")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.grid(True)
ax.legend(
    [line_train, line_val], 
    ["Training Accuracy", "Validation Accuracy"], 
    loc = "lower right",
)

# Add interactive cursor
mplcursors.cursor(line_train)
mplcursors.cursor(line_val)

#! Unmark these part, whenever you not want to save the model parameters and accuracy graph.


# Create a Timestamped directory for saving

# Get the current date and time.
now = datetime.now()

# Format the timestamp for the directory name: e.g. "12/08/2025_10:30"
new_dir_name: str = now.strftime("%d-%m-%Y_%H-%M")

# Create the new directory path.
MODEL_SAVE_PATH: str = r"C:/Users/Besitzer/Desktop/Python/AI Projects/Facial Emotion Recognition/Runs/"
new_dir_path: str = os.path.join(MODEL_SAVE_PATH, new_dir_name)

# Create the directory.
os.makedirs(new_dir_path,)
print(f"\nDirectory {new_dir_name} created succesfully.")

# Define the file paths within the new directory.
model_parameters_path = os.path.join(new_dir_path, "model_parameters.pth")
acc_graph_path = os.path.join(new_dir_path, "accuracy_graph.png")

# Save the model's state_dict.
torch.save(model.state_dict(), model_parameters_path)
# Save the Matplotlib.graph.
plt.savefig(acc_graph_path)


# plt.show has to be here, because it resets the canvas after its closed.
plt.show()

emotion_labels: list[str] = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

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
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=emotion_labels,
    )
    print("\nClassification Report: ")
    print(report)

    # Save the report as .txt file to the newly created directory.
    report_path: str = os.path.join(new_dir_path, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("Classification Report:\n")
        f.write("\n")
        f.write(report)
    print("\nReport also saved in the new directory.")

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
