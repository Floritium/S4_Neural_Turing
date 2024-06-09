"""All in one NTM. Encapsulation of all components."""
import torch
from torch import nn
from .ntm import NTM
from .controller import LSTMController
from .head import NTMReadHead, NTMWriteHead
from .memory import NTMMemory


class EncapsulatedNTM(nn.Module):

    def __init__(self, num_inputs, num_outputs,
                 controller_size, controller_layers, num_heads, N, M, device):
        """Initialize an EncapsulatedNTM.

        :param num_inputs: External number of inputs.
        :param num_outputs: External number of outputs.
        :param controller_size: The size of the internal representation.
        :param controller_layers: Controller number of layers.
        :param num_heads: Number of heads.
        :param N: Number of rows in the memory bank.
        :param M: Number of cols/features in the memory bank.
        """
        super(EncapsulatedNTM, self).__init__()

        # Save args
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller_size = controller_size
        self.controller_layers = controller_layers
        self.num_heads = num_heads
        self.N = N
        self.M = M
        self.device = device

        # Create the NTM components
        memory = NTMMemory(N, M, device)
        controller = LSTMController(num_inputs + M*num_heads, controller_size, controller_layers, device)
        heads = nn.ModuleList([])
        for i in range(num_heads):
            heads += [
                NTMReadHead(memory, controller_size, device),
                NTMWriteHead(memory, controller_size, device)
            ]

        self.ntm = NTM(num_inputs, num_outputs, controller, memory, heads, device)
        self.memory = memory

    def init_sequence(self, batch_size):
        """Initializing the state."""
        self.batch_size = batch_size
        self.memory.reset(batch_size)
        self.previous_state = self.ntm.create_new_state(batch_size)

    def forward(self, x=None):
        if x is None:
            x = torch.zeros(self.batch_size, self.num_inputs).to(self.device)

        o, self.previous_state = self.ntm(x, self.previous_state)
        return o, self.previous_state

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params

    def to_device(self):
        """Move model to a specified device."""
        self.to(self.device)
        self.memory.to(self.device)
        for head in self.ntm.heads:
            head.to(self.device)
        self.ntm.controller.to(self.device)