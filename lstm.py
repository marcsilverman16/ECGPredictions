import torch
import torch.nn as nn
import torch.nn.init as init

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, output_size=3, dropout=0.3):
        """
        Args:
            input_size (int): The number of input features per time step.
            hidden_size (int): The number of features in the hidden state h.
            num_layers (int): The number of stacked LSTM layers.
            output_size (int): The number of output features.
            dropout (float): The dropout probability.
        """
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

        # Initialize hidden and cell states
        self.init_hidden_cell = self._init_hidden_cell

    def _init_hidden_cell(self, batch_size, device):
        init_weight = 0.1
        h0 = torch.randn(self.num_layers, batch_size, self.hidden_size) * init_weight
        c0 = torch.randn(self.num_layers, batch_size, self.hidden_size) * init_weight
        return h0.to(device), c0.to(device)

    def forward(self, x, device=None):
        x = x.transpose(1, 2)  # Change to (batch_size, seq_length, features)

        # Get device from the first parameter tensor if device is not provided
        if device is None:
            device = next(self.parameters()).device
        else:
            x = x.to(device)

        # Initialize hidden and cell states
        h0, c0 = self.init_hidden_cell(x.size(0), device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out