
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, num_class: int = 7) -> None:
        super(ConvNet, self).__init__()
        # Input shape: (1, 48, 48)

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32), # Batch Normalization for 4D tensors (batch_size, number_of_channels, height, width)
            nn.MaxPool2d(kernel_size=2, stride=2),
        ) # Output shape: (32, 24, 24)
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),     
            nn.ReLU(),
            nn.BatchNorm2d(64), 
            nn.MaxPool2d(kernel_size=2, stride=2),
        ) # Output shape: (64, 12, 12)
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2), 
        ) # Output shape: (128, 6, 6)

        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2), 
        ) # Output shape: (128, 3, 3)

        self.fc_layer = nn.Sequential(
            # The input features are calculated based on the output of conv_layer4
            nn.Linear(in_features=128 * 3 * 3, out_features=256),
            #?nn.BatchNorm1d(256), # 2D Tensor (batch_size, number_of_features)
            nn.ReLU(),
            nn.Dropout(p=0.5), # Dropout to prevent overfitting
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=num_class), 
        )

    def forward(self, x):
        # Pass the input through all four convolutional layers
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        
        # Flatten the output of the convolutional layers
        x = x.view(x.size(0), -1)
        
        # Pass the flattened tensor to the fully connected layers
        return self.fc_layer(x)



