#!/usr/bin/env python
import torch
from torch import nn
import torch.nn.functional as F
import random

class NTM_S4D(nn.Module):
    """A Neural Turing Machine."""
    def __init__(self, num_inputs, num_outputs, controller, memory, heads, use_memory, n_layers, device):
        """Initialize the NTM.

        :param num_inputs: External input size.
        :param num_outputs: External output size.
        :param controller: :class:`s4-controller`
        :param memory: :class:`NTMMemory`
        :param heads: list of :class:`NTMReadHead` or :class:`NTMWriteHead`

        Note: This design allows the flexibility of using any number of read and
              write heads independently, also, the order by which the heads are
              called in controlled by the user (order in list)
        """
        super(NTM, self).__init__()

        # Save arguments
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller = controller
        self.memory = memory
        self.heads = heads
        self.use_memory = use_memory

        self.N, self.M = memory.size()
        _, self.controller_size = controller.size()

        # Initialize the initial previous read values to random biases
        self.num_read_heads = 0
        self.init_r = []
        for head in heads:
            if head.is_read_head():
                init_r_bias = torch.randn(1, self.M) * 0.01
                self.register_buffer("read{}_bias".format(self.num_read_heads), init_r_bias.data.to(device))
                self.init_r += [init_r_bias.to(device)]
                self.num_read_heads += 1

        assert self.num_read_heads > 0, "heads list must contain at least a single read head"

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.prenorm = prenorm
        self.encoder = nn.Linear(self.num_inputs, d_model)

        # Stack S4 layers as residual blocks
        self.controller = controller
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                self.controller
                #S4(d_model, dropout=dropout, transposed=True, lr=min(0.001, args.lr), mode='s4d', init='diag-lin', bidirectional=False, disc='bilinear', real_transform='exp')
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        self.decoder = nn.Linear(self.controller_size + self.num_read_heads * self.M, self.num_outputs)
        
        self.device = device
        self.reset_parameters()

    def create_new_state(self, batch_size):
        init_r = [r.clone().repeat(batch_size, 1) for r in self.init_r]
        controller_state = self.s4_layers[-1].default_state(batch_size)
        heads_state = [head.create_new_state(batch_size) for head in self.heads]

        return init_r, controller_state, heads_state

    def reset_parameters(self):
        # Initialize the linear layer
        nn.init.xavier_uniform_(self.fc.weight, gain=1)
        nn.init.normal_(self.fc.bias, std=0.01)

    def forward(self, x, prev_state):
        """NTM forward function.

        :param x: input vector (batch_size x num_inputs)
        :param prev_state: The previous state of the NTM
        """
        # Unpack the previous state
        prev_reads, prev_controller_state, prev_heads_states = prev_state

        # Use the controller to get an embeddings
        inp = torch.cat([x] + prev_reads, dim=1)
        controller_outp, controller_state = self.controller(inp, prev_controller_state)
        interact_with_memory = random.random() < self.use_memory

        # Read/Write from the list of heads
        if interact_with_memory:
            reads = []
            heads_states = []
            for head, prev_head_state in zip(self.heads, prev_heads_states):
                if head.is_read_head():
                    # input the hidden cell state of the controller
                    r, head_state = head(controller_state.squeeze(0), prev_head_state) # output r=(batch_size x M), head_state=(batch_size x N)
                    reads += [r]
                else:
                    # input the hidden cell state of the controller
                    head_state = head(controller_state[-1].squeeze(0), prev_head_state)
                heads_states += [head_state]
        else:
            # hold in cache for the next iteration
            reads = prev_reads
            heads_states = prev_heads_states

        # Generate Output
        inp2 = torch.cat([controller_state[-1].squeeze(0)] + reads, dim=1)
        o = self.fc(inp2)

        # Pack the current state
        state = (reads, controller_state, heads_states)

        return o, state