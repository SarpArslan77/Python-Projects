
import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from data import (
    test_loader,
    train_loader,
    val_loader,
)
import intel_extension_for_pytorch as ipex

# Neural Network implementation
"""
from nn import (
    NeuralNetwork,
    input_size,
    hidden_node_amount,
    num_classes,
    num_epochs,
    batch_size,
    learning_rate,
    device
)

# Create the NN-Model.
model = NeuralNetwork(input_size, hidden_node_amount, num_classes).to(device)
criterion = nn.CrossEntropyLoss() # includes Softmax Activation
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(training_loader):
        # Reshape the images from (batch_size, 1, 48, 48) to (batch_size, 2304)
        images = images.reshape(-1, 48*48).to(device)
        labels = labels.to(device)

        # Forward Pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and Optimize
        optimizer.zero_grad() # Clear gradients from the previous step.
        loss.backward() # Compute the gradients.
        optimizer.step() # Update the weights and biases
    
    print(f"Epoch [{epoch}/{num_epochs-1}], Loss: {loss.item():.4f}")
print("\nFinished Training!")

# Test Loop
print("\nStarting with the Test.")
model.eval()
with torch.no_grad(): 
# We use torch.no_grad() to save memory and computations, as we don't need gradients here
    total_n_correct: int = 0
    total_n_samples: int = 0
    # TP: True Positive, TN: True Negative, FP: False Positive, FN: False Negative
    emotion_counts: list = [
        [0 for _ in range(2)] for _ in range(7)
    ] # Is built like: [[TP_angry, angry_count], [TP_disgust, disgust_count], ...]

    for images, labels in test_loader:
        images = images.reshape(-1, 48*48).to(device)
        labels = labels.to(device) # Shape: (512)
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        # Calculate the overall accuracy. 
        total_n_samples += labels.size(0)
        total_n_correct += (predicted == labels).sum().item()

        # Calculate the precision for each emotion.
        emotion_counts[0][0] += torch.logical_and(labels == 0, predicted == 0).sum().item()
        emotion_counts[0][1] += (predicted == 0).sum().item()
        emotion_counts[1][0] += torch.logical_and(labels == 1, predicted == 1).sum().item()
        emotion_counts[1][1] += (predicted == 1).sum().item()
        emotion_counts[2][0] += torch.logical_and(labels == 2, predicted == 2).sum().item()
        emotion_counts[2][1] += (predicted == 2).sum().item()
        emotion_counts[3][0] += torch.logical_and(labels == 3, predicted == 3).sum().item()
        emotion_counts[3][1] += (predicted == 3).sum().item()
        emotion_counts[4][0] += torch.logical_and(labels == 4, predicted == 4).sum().item()
        emotion_counts[4][1] += (predicted == 4).sum().item()
        emotion_counts[5][0] += torch.logical_and(labels == 5, predicted == 5).sum().item()
        emotion_counts[5][1] += (predicted == 5).sum().item()
        emotion_counts[6][0] += torch.logical_and(labels == 6, predicted == 6).sum().item()
        emotion_counts[6][1] += (predicted == 6).sum().item()
        
        print(emotion_counts[0][1])
        print(emotion_counts[1][1])
        print(emotion_counts[2][1])
        print(emotion_counts[3][1])
        print(emotion_counts[4][1])
        print(emotion_counts[5][1])
        print(emotion_counts[6][1])
        print()

    accuracy = round(total_n_correct/total_n_samples * 100, 2)
    print(f"\nOverall accuracy on the test set: {accuracy} %")

    precision_angry: float = round(emotion_counts[0][0] / emotion_counts[0][1], 2)
    precision_disgust: float = round(emotion_counts[1][0] / emotion_counts[1][1], 2)
    precision_fear: float = round(emotion_counts[2][0] / emotion_counts[2][1], 2)
    precision_happy: float = round(emotion_counts[3][0] / emotion_counts[3][1], 2)
    precision_neutral: float = round(emotion_counts[4][0] / emotion_counts[4][1], 2)
    precision_sad: float = round(emotion_counts[5][0] / emotion_counts[5][1], 2)
    precision_surprise: float = round(emotion_counts[6][0] / emotion_counts[6][1], 2)
    print(f"\nPrecision on the Emotions:")
    print(f"angry: {precision_angry} %")
    print(f"disgust: {precision_disgust} %")
    print(f"fear: {precision_fear} %")
    print(f"happy: {precision_happy} %")
    print(f"neutral: {precision_neutral} %")
    print(f"sad: {precision_sad} %")
    print(f"surprise: {precision_surprise} %")
"""

from cnn import (
    device,
    model,
    criterion,
    optimizer,
    scheduler,
)

# Hyperparameters
NUM_EPOCHS: int = 10

# Apply IPEX optimizations to the model and optimizer
model, optimizer = ipex.optimize(model, optimizer=optimizer)

# Training Loop
print("\nStarting Training!")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss: float = 0.0 # Tracking loss for optimizing the learning rate.
    for i, (images, labels) in enumerate(train_loader):
        print(i)
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the loss
        running_loss += loss.item() * images.size(0)

    epoch_train_loss: float = running_loss / len(train_loader.dataset)

    # Validation Phase
    model.eval()
    val_loss: float = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
    
    epoch_val_loss: float = val_loss / len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

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
        print(j)
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
