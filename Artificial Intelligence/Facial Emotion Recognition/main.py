
import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
import intel_extension_for_pytorch as ipex

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
NUM_EPOCHS: int = 50

# Apply IPEX optimizations to the model and optimizer
model, optimizer = ipex.optimize(model, optimizer=optimizer)

# Training Loop
print("\nStarting Training!")
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
    print(f"\nTrain accuracy : {train_acc:.2f}% | Train Loss: {epoch_train_loss:.4f}")

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
    print(f"Validation accuracy : {val_acc:.2f}% | Val Loss: {epoch_val_loss:.4f}")

    # Step the scheduler with validation loss
    scheduler.step(epoch_val_loss)

print("\nFinished Training.")

#! Unmark these part, whenever you want to override and save the model parameters.

# Save the model parameters, also weights and biases.
MODEL_SAVE_PATH: str = r"C:/Users/Besitzer/Desktop/Python/AI Projects/Facial Emotion Recognition/trained_model.pth"
# Save the state_dict.
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Saved the model parameters.")


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
