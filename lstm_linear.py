import torch
import torch.nn as nn


class LSTMWithLinearLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, device: str = "cpu"):
        super().__init__()

        self._lstm = nn.LSTM(input_size, hidden_size)
        self._linear = nn.Linear(hidden_size, output_size)
        self.device = device

    def forward(self, x_t: torch.Tensor):
        _, (h_t, _) = self._lstm(x_t)

        return self._linear(h_t)

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params
    
    def to_device(self):
        """Move model to a specified device."""
        self.to(self.device)