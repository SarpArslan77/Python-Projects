
# lstm.py

#! Custom TODO notes:
#TODO AD: Add docstring.
#TODO ATH: Add type hint.
#TODO CTH: Check type hint.
#TODO FTH: Fix the hardcoding.
#TODO HPE: Handle possible error.
#TODO AC: Add comment.
#TODO AIC: Add input control.

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

@dataclass
class ConfigLSTM:
    """
    Configuration dataclass for the LSTM model.

    Attributes:
        input_size (int): The number of expected features in the input (e.g., 3 for Volt/Curr/Temp).
        hidden_size (int): The number of features in the hidden state `h`.
        num_layers (int): Number of recurrent layers (stacking LSTMs).
        output_size (int): The size of the final output (e.g., 1 for regression).
        bidirectional (bool): If True, becomes a bidirectional LSTM.
        dropout_p (float): Dropout probability (0.0 to 1.0).
    """

    input_size: int
    hidden_size: int
    num_layers: int
    output_size: int
    bidirectional: bool
    dropout_p: float

    def __post_init__(self) -> None:
        # - input_size
        if not isinstance(self.input_size, int):
            raise TypeError(f"input_size must be an integer, got {type(self.input_size).__name__}.")
        if self.input_size <= 0:
            raise ValueError(f"input_size must be positive, got {self.input_size}.")

        # - hidden_size
        if not isinstance(self.hidden_size, int):
            raise TypeError(f"hidden_size must be an integer, got {type(self.hidden_size).__name__}.")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}.")
        
        # - num_layers
        if not isinstance(self.num_layers, int):
            raise TypeError(f"num_layers must be an integer, got {type(self.num_layers).__name__}.")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}.")
        
        # - output_size
        if not isinstance(self.output_size, int):
            raise TypeError(f"output_size must be an integer, got {type(self.output_size).__name__}.")
        if self.output_size <= 0:
            raise ValueError(f"output_size must be positive, got {self.output_size}.")
        
        # - bidirectional
        if not isinstance(self.bidirectional, bool):
            raise TypeError(f"bidirectional must be a boolean, got {type(self.bidirectional).__name__}.")
        
        # - dropout_p
        if not isinstance(self.dropout_p, float):
            raise TypeError(f"dropout_p must be a float, got {type(self.dropout_p).__name__}.")
        if not (0.0 <= self.dropout_p <= 1.0):
            raise ValueError(f"dropout_p must be between 0.0 and 1.0, got {self.dropout_p}.")


class LSTM(nn.Module):
    """
    Standard LSTM (Long Short-Term Memory) regression model.

    This architecture follows a Many-to-One pattern:
    1. Processes the entire input sequence using LSTM layers.
    2. Extracts the hidden state of the **last** time step.
    3. Applies Dropout for regularization.
    4. Projects to the target dimension using a Linear layer.
    """   

    def __init__(
            self,
            config_lstm: ConfigLSTM
    ) -> None:     
        super().__init__()
        self.cfg_lstm: ConfigLSTM = config_lstm

        # Define the device.
        self.device = torch.device("cpu")

        # Define the Long-Short-Time-Memory (LSTM) architecture.
        self.lstm = nn.LSTM(
            input_size = self.cfg_lstm.input_size,
            hidden_size = self.cfg_lstm.hidden_size,
            num_layers = self.cfg_lstm.num_layers,
            batch_first = True, # Forces the input and output the shape: (batch_size, sequences, features).
            bidirectional = self.cfg_lstm.bidirectional
        )

        # Define the fully connected layer (output layer).
        self.fc = nn.Linear(
            in_features = self.cfg_lstm.hidden_size * (1 + int(self.cfg_lstm.bidirectional)),
            out_features = self.cfg_lstm.output_size
        )

        # Define the dropout layer.
        self.dropout = nn.Dropout(p=self.cfg_lstm.dropout_p)

    def forward(
            self,
            x: Tensor
    ) -> Tensor:
        """
        Forward propagation for the LSTM.

        Args:
            x (Tensor): Input data.
            
        Returns:
            output (Tensor): Output data.
        """

        # 1. Initialize the hidden-(short time memory) and cell states (long time memory).
        h0: Tensor = torch.zeros(
            size = (
                self.cfg_lstm.num_layers * (1 + int(self.cfg_lstm.bidirectional)),
                x.size(0),
                self.cfg_lstm.hidden_size
            )
        ).to(self.device) # (num_layers, batch_size, hidden_size)
        c0: Tensor = torch.zeros(
            size = (
                self.cfg_lstm.num_layers * (1 + int(self.cfg_lstm.bidirectional)),
                x.size(0),
                self.cfg_lstm.hidden_size
            )
        ).to(self.device)

        # 2. Returns the output and the final hidden- and cell states.
        lstm_output, _ = self.lstm(x, (h0, c0)) # (batch_size, seq_length, hidden_size).

        # Only the last output from the last step is needed for the prediction.
        lstm_last_output: Tensor = lstm_output[:, -1, :]
        
        # 3. Pass the last output through a dropout layer.
        lstm_last_output_dropped: Tensor = self.dropout(lstm_last_output)

        # 4. Pass the last dropped output through the linear output layer.
        output: Tensor = self.fc(lstm_last_output_dropped)

        return output