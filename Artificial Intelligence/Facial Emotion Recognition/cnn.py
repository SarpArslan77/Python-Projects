import torch
import torch.nn as nn
import torch.optim as optim

# Check whether Intel XE GPU is avaiable, if not use the CPU
try:
    device = torch.device("xpu") 
except:
    device = torch.device("cpu") 
print(f"\nUsing device: {device}") 

# Hyperparameters
STARTING_LEARNING_RATE: float = 5e-4

class ConvNet(nn.Module):
    def __init__(self, num_class: int = 7) -> None:
        super(ConvNet, self).__init__()
        # Input shape: (1, 48, 48)

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output shape: (32, 24, 24)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # (64, 12, 12)
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # (128, 6, 6)
        )

        # After two max pools, the 48x48 image is now 12x12
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=128*6*6, out_features=256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=256, out_features=num_class),
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = x.view(x.size(0), -1)

        return self.fc_layer(x)
    
    
# Convolutional Neural Network implementation
model = ConvNet().to(device)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=STARTING_LEARNING_RATE)

# Scheduler Definition
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode = "min", # It will look for the metric to stop decreasing.
    factor = 0.1, # By which the learning rate will be reduced.
    patience = 3, # Number of epochs with no improvement after which learning rate will be reduced.
    min_lr = 1e-4, # Minimum learning rate, that the optimizer can drop to.
)


