
#TODO: it shows 100 % Accuracy and 0.0 Loss, which is obviously false

import cv2
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
    n_correct: int = 0
    n_samples: int = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 48*48).to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
    
    accuracy = 100 * n_correct/n_samples
    print(f"Accuracy on the test set: {accuracy:.2f} %")




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