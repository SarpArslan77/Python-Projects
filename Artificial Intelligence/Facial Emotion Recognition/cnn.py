import torch
import torch.nn as nn
import torch.optim as optim
import intel_extension_for_pytorch as ipex


device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print(f"\nUsing device: {device}") # <-- ADD THIS LINE

# Hyperparameters
LEARNING_RATE: float = 0.001

class ConvNet(nn.Module):
    def __init__(self) -> None:
        super(ConvNet, self).__init__()
        # Input shape: (1, 48, 48)

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output shape: (64, 24, 24)
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
        )
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output shape: (128, 12, 12)
        )
        self.conv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
        )
        self.conv_layer6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output shape: (256, 6, 6)
        )

        # After two max pools, the 48x48 image is now 12x12
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=256*6*6, out_features=1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 7) # 7 classes = 7 emotions
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        x = self.conv_layer6(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)

        return x
    
# Convolutional Neural Network implementation
model = ConvNet().to(device)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

# Scheduler Definition
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode = "min", # It will look for the metric to stop decreasing.
    factor = 0.1, # By which the learning rate will be reduced.
    patience = 3, # Number of epochs with no improvement after which learning rate will be reduced
)
