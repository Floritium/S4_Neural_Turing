"""LSTM Controller."""
import torch
from torch import nn
from torch.nn import Parameter
import numpy as np


class LSTMController(nn.Module):
    """An NTM controller based on LSTM."""
    def __init__(self, num_inputs, num_outputs, num_layers, device):
        super(LSTMController, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTM(input_size=num_inputs,
                            hidden_size=num_outputs,
                            num_layers=num_layers,
                            batch_first=False).to(device)

        # The hidden state is a learned parameter
        self.lstm_h_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05).to(device)
        self.lstm_c_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05).to(device)

        self.reset_parameters()

    def create_new_state(self, batch_size):
        # Dimension: (num_layers * num_directions, batch, hidden_size)
        lstm_h = self.lstm_h_bias.clone().repeat(1, batch_size, 1)
        lstm_c = self.lstm_c_bias.clone().repeat(1, batch_size, 1)
        return lstm_h, lstm_c

    def reset_parameters(self):
        for p in self.lstm.parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.num_inputs +  self.num_outputs))
                nn.init.uniform_(p, -stdev, stdev)

    def size(self):
        return self.num_inputs, self.num_outputs

    def forward(self, x, prev_state):
        if x.dim() == 2: # use this for sequence length = 1, where each item in the sequence is a single time step and gets fed in
            x = x.unsqueeze(0)
        outp, state = self.lstm(x, prev_state)
        return outp.squeeze(0), state

    # def to_device(self, device):
    #     """Move model and its parameters to a specified device."""
    #     self.to(device)
    #     self.lstm_h_bias.data = self.lstm_h_bias.data.to(device)
    #     self.lstm_c_bias.data = self.lstm_c_bias.data.to(device)
