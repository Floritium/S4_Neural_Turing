import torch
from torch import nn
import torch.nn.functional as F

class Controller_LSTM(nn.Module):
    
    def __init__(self, input_dim, ctrl_dim, output_dim, read_data_size, device):
        super(Controller_LSTM, self).__init__()
        
        self.input_size = input_dim
        self.ctrl_dim = ctrl_dim
        self.output_size = output_dim
        self.read_data_size = read_data_size
        self.device = device
        
        # Controller neural network
        self.controller_net = nn.LSTMCell(input_dim, ctrl_dim).to(self.device)
        # Output neural network
        self.out_net = nn.Linear(read_data_size, output_dim).to(self.device)
        # Initialize the weights of output net
        nn.init.kaiming_uniform_(self.out_net.weight)
        
        # Learnable initial hidden and cell states
        self.h_state = torch.zeros([1, ctrl_dim]).to(self.device)
        self.c_state = torch.zeros([1, ctrl_dim]).to(self.device)
        # Layers to learn init values for controller hidden and cell states
        self.h_bias_fc = nn.Linear(1, ctrl_dim).to(self.device)
        self.c_bias_fc = nn.Linear(1, ctrl_dim).to(self.device)
        
        # Reset
        self.reset()
        
    def forward(self, x, prev_reads):
        '''Returns the hidden and cell states'''
        x = torch.cat([x, *prev_reads], dim=1)
        # Get hidden and cell states
        self.h_state, self.c_state = self.controller_net(x, (self.h_state, self.c_state))
        
        return self.h_state, self.c_state
    
    def output(self, reads):
        '''Returns the external output from the read vectors'''
        out_state = torch.cat([self.h_state, *reads], dim=1)
        # Compute output
        output = self.out_net(out_state) # logits

        return output
        
    def reset(self, batch_size=1):
        '''Reset/initialize the controller states'''

        # Dummy input
        in_data = torch.tensor([[0.]]).to(self.device)
        # Hidden state
        h_bias = self.h_bias_fc(in_data)
        self.h_state = h_bias.repeat(batch_size, 1)
        # Cell state
        c_bias = self.c_bias_fc(in_data)
        self.c_state = c_bias.repeat(batch_size, 1)