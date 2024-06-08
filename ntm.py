import torch
from torch import nn
import torch.nn.functional as F

from modules import Memory, Head, Controller_LSTM

class NTM(nn.Module):
    
    def __init__(self,
                 input_dim,
                 output_dim,
                 ctrl_dim,
                 memory_units,
                 memory_unit_size,
                 num_heads,
                 device='cpu'):
        super(NTM, self).__init__()

        self.device = device
        print('NTM Using device:', self.device)

        # Create controller
        self.ctrl_dim = ctrl_dim
        self.controller = Controller_LSTM(input_dim + num_heads * memory_unit_size,
                                     ctrl_dim, 
                                     output_dim,
                                     ctrl_dim + num_heads * memory_unit_size, device).to(device)
        
        # Create memory
        self.memory = Memory(memory_units, memory_unit_size, device)
        self.memory_unit_size = memory_unit_size # M
        self.memory_units = memory_units # N
        
        # Create Heads
        self.num_heads = num_heads
        self.heads = nn.ModuleList([])
        for head in range(num_heads):
            self.heads += [
                Head('r', ctrl_dim, memory_unit_size, device),
                Head('w', ctrl_dim, memory_unit_size, device)
            ]
        
        # Init previous head weights and read vectors
        self.prev_head_weights = []
        self.prev_reads = []
        # Layers to initialize the weights and read vectors
        self.head_weights_fc = nn.Linear(1, self.memory_units).to(self.device)
        self.reads_fc = nn.Linear(1, self.memory_unit_size).to(self.device)
        
        self.reset()
        
        
    def forward(self, x, memorize=True):
        '''Returns the output of the Neural Turing Machine'''
        
        # use the s4 to process the input @todo
        output = None
        
        # Get controller states
        ctrl_hidden, ctrl_cell = self.controller(x, self.prev_reads)
        
        # Read, and Write
        reads = []
        head_weights = []
        
        for head, prev_head_weights in zip(self.heads, self.prev_head_weights):
            # Read
            if head.mode == 'r':
                weights, read_vec = head(ctrl_cell, prev_head_weights, self.memory)
                reads.append(read_vec)
            # Write    
            elif head.mode == 'w':
                weights, _ = head(ctrl_cell, prev_head_weights, self.memory)
            
            head_weights.append(weights)
        
        # Compute output
        if memorize != True:
            output = self.controller.output(reads)

        # use the s4 after the output @todo
        
        self.prev_head_weights = head_weights
        self.prev_reads = reads
            
        return output

    def to_device():
        self.to(self.device)
    
    def reset(self, batch_size=1):
        '''Reset/initialize NTM parameters'''
        # Reset memory and controller
        self.memory.reset(batch_size)
        self.controller.reset(batch_size)

        self.controller.to(self.device)
        self.memory.to(self.device)
        
        # Initialize previous head weights (attention vectors)
        self.prev_head_weights = []
        for i in range(len(self.heads)):
            # prev_weight = torch.zeros([batch_size, self.memory.n])
            prev_weight = F.softmax(self.head_weights_fc(torch.Tensor([[0.]]).to(self.device)), dim=1)
            self.prev_head_weights.append(prev_weight)
        
        # Initialize previous read vectors
        self.prev_reads = []
        for i in range(self.num_heads):
            # prev_read = torch.zeros([batch_size, self.memory.m])
            # nn.init.kaiming_uniform_(prev_read)
            prev_read = self.reads_fc(torch.Tensor([[0.]]).to(self.device))
            self.prev_reads.append(prev_read)