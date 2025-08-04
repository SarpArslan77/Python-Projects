import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the network and the parameters.
input_size: int = 2304 # 48 x 48 pixels
hidden_node_amount: int = 32 # Number of nodes in each hidden layer
num_classes: int = 7 # 7 Emotions = angry, disgust, fear, happy, neutral, sad, surprise
num_epochs: int = 11 # Amount of repeating.
batch_size: int = 128
learning_rate: float = 0.2

class NeuralNetwork(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_node_amount: int,
            num_classes: int
    ) -> None:
        super(NeuralNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_node_amount),
            nn.ReLU(),
            nn.Linear(hidden_node_amount, hidden_node_amount),
            nn.ReLU(),
            nn.Linear(hidden_node_amount, hidden_node_amount),
            nn.ReLU(),
            nn.Linear(hidden_node_amount, num_classes),
        )

    def forward(self, x):
        return self.network(x)
    

        