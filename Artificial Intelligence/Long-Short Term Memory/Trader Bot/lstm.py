
import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int      
    ):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first = True, # Makes the input and output tensors have shape (batch_size, sequences, features).
            bidirectional = True, # Allows the model the process the sequence in both directions.
        )

        # Define a seperate dropout layer.
        self.dropout = nn.Dropout(0.2)

        # Define fully connected layer.
        self.fc = nn.Linear(hidden_size * 2, output_size) # Because of bidirectionality, the output size of LSTM is doubled.

    def forward(self, x):
        # Initialize hidden(short_time memory) and cell states(long-time memory) with zeros. shape: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device) # Because of bidirectionality, the output size of LSTM is doubled.
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # The LSTM returns the output and the final hidden & ceel states.
        lstm_out, _ = self.lstm(x, (h0, c0))
        # lstm_out: (batch_size, seq_length, hidden_size)

        # We only need the output from the last step for our prediction.
        lstm_output = lstm_out[:, -1, :]

        #!out = self.dropout(lstm_output)

        # Pass the last output through the linear layer.
        out = self.fc(lstm_output)

        return out