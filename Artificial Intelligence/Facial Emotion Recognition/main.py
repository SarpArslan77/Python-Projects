
#TODO: fix precision. the total count of each emotion is 0, so it causes problem for the division
#TODO:      all guesses are 3?

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data import (
    test_loader,
    training_loader,
)
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
        emotion_counts[0][0] += (labels == 0).sum().item()
        emotion_counts[0][1] += (predicted == 0).sum().item()
        emotion_counts[1][0] += (labels == 1).sum().item()
        emotion_counts[1][1] += (predicted == 1).sum().item()
        emotion_counts[2][0] += (labels == 2).sum().item()
        emotion_counts[2][1] += (predicted == 2).sum().item()
        emotion_counts[3][0] += (labels == 3).sum().item()
        emotion_counts[3][1] += (predicted == 3).sum().item()
        emotion_counts[4][0] += (labels == 4).sum().item()
        emotion_counts[4][1] += (predicted == 4).sum().item()
        emotion_counts[5][0] += (labels == 5).sum().item()
        emotion_counts[5][1] += (predicted == 5).sum().item()
        emotion_counts[6][0] += (labels == 6).sum().item()
        emotion_counts[6][1] += (predicted == 6).sum().item()
        
        """print(emotion_counts[0][1])
        print(emotion_counts[1][1])
        print(emotion_counts[2][1])
        print(emotion_counts[3][1])
        print(emotion_counts[4][1])
        print(emotion_counts[5][1])
        print(emotion_counts[6][1])"""

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
