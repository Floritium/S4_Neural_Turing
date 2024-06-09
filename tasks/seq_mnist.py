import random

from attr import attrs, attrib, Factory
import torch
from torch import nn
from torch import optim
import numpy as np

from ntm.aio import EncapsulatedNTM
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader

class SequentialMNIST(Dataset):
    def __init__(self, task_params, train=True):
        self.resize_resolution = task_params["resize_resolution"]

        # Define the transformation to be applied to the data
        self.transform = transforms.Compose([
            transforms.Resize((self.resize_resolution, self.resize_resolution)),  # Resize to higher resolution, e.g., 56x56
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Normalize the tensor to have pixel values in [-1, 1]
            transforms.Lambda(lambda x: x.view(1, self.resize_resolution * self.resize_resolution).t())  # Adjust view for new resolution
        ])

        # Load the MNIST dataset
        self.dataset = torchvision.datasets.MNIST(
            root="./data/sequential_mnist",
            train=train,
            download=True,
            transform=self.transform,
        )

        # hardcoded as MNIST images are 28x28, but here we resize
        self.seq_len = self.resize_resolution * self.resize_resolution

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_seq = self.dataset[idx][0]
        target_label = self.dataset[idx][1]
        return input_seq, torch.tensor([target_label])

# def dataloader(num_batches,
#                batch_size,
#                max_len,
#                dataset):
#     """Generator of random sequences for the copy task."""
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     for batch_num in range(dataloader):




@attrs
class SeqMNISTParams(object):
    name = attrib(default="seq_mnist-task")
    controller_size = attrib(default=100)
    controller_layers = attrib(default=1)
    num_heads = attrib(default=1)
    sequence_width = attrib(default=14)
    input_dim = attrib(default=1)
    output_dim = attrib(default=10)
    memory_n = attrib(default=128)
    memory_m = attrib(default=20)
    num_batches = attrib(default=250000)
    batch_size = attrib(default=64)
    rmsprop_lr = attrib(default=1e-4)
    rmsprop_momentum = attrib(default=0.9)
    rmsprop_alpha = attrib(default=0.95)
    device = attrib(default="cpu")


@attrs
class SeqMNISTModelTraining(object):
    params = attrib(default=Factory(SeqMNISTParams))
    net = attrib()
    dataloader = attrib()
    criterion = attrib()
    optimizer = attrib()

    @net.default
    def default_net(self):
        # We have 1 additional input for the delimiter which is passed on a
        # separate "control" channel
        net = EncapsulatedNTM(
            self.params.input_dim,
            self.params.output_dim,
            self.params.controller_size,
            self.params.controller_layers,
            self.params.num_heads,
            self.params.memory_n,
            self.params.memory_m,
            self.params.device,
        )
        return net

    @dataloader.default
    def default_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            SequentialMNIST({"resize_resolution": self.params.sequence_width}),
            batch_size=self.params.batch_size,
            shuffle=False,
        )
        self.params.num_batches = len(dataloader)
        return dataloader

    @criterion.default
    def default_criterion(self):
        return nn.CrossEntropyLoss()

    @optimizer.default
    def default_optimizer(self):
        return optim.RMSprop(
            self.net.parameters(),
            momentum=self.params.rmsprop_momentum,
            alpha=self.params.rmsprop_alpha,
            lr=self.params.rmsprop_lr,
        )
