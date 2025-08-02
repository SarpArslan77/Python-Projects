
import torch
import torch.nn as nn

# Create the LSTM Model
class LSTM(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            output_size: int
    ) -> None:
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the LSTM layer.
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # batch_first=True makes the input and output tensors have shape (batch_size, sequences, features).

        # Define the fully connected linear output layer.
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden(short-time memory) and cell states(long-time memory) 
        #   with zeros with shape : (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # The LSTM returns the output and the final hidden & cell states
        lstm_out, _ = self.lstm(x, (h0, c0))
        # lstm_out : (batch_size, seq_length, hidden_size)

        # We only need the output from the last time step for our prediction.
        last_time_step_out = lstm_out[:, -1, :]

        # Pass the last output through the last linear layer.
        out = self.fc(last_time_step_out)

        return out