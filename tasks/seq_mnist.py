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
from lstm_linear import LSTMWithLinearLayer

class SequentialMNIST(Dataset):
    def __init__(self, task_params, train=True):
        self.resize_resolution = task_params["resize_resolution"]

        # Define the transformation to be applied to the data
        self.transform = transforms.Compose([
            transforms.Resize((self.resize_resolution, self.resize_resolution)),  # Resize to higher resolution or lower, e.g., 56x56
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


@attrs
class SeqMNISTParams_ntm(object):
    name = attrib(default="seq-mnist-ntm")
    model_name = "ntm"
    controller_size = attrib(default=100)
    controller_layers = attrib(default=1)
    num_heads = attrib(default=1)
    resize_resolution = attrib(default=16)
    input_dim = attrib(default=1)
    output_dim = attrib(default=10)
    memory_n = attrib(default=128)
    memory_m = attrib(default=20)
    num_batches = attrib(default=0)
    batch_size = attrib(default=64)
    rmsprop_lr = attrib(default=1e-4)
    rmsprop_momentum = attrib(default=0.9)
    rmsprop_alpha = attrib(default=0.95)
    device = attrib(default="cpu")
    fraction = attrib(default=0.5)
    use_memory = attrib(default=1.0)
    seq_len = attrib(default=0)

@attrs
class SeqMNISTParams_ntm_s4d(object):
    name = attrib(default="seq-mnist-ntm-s4d")
    model_name = "ntm_s4d"
    controller_size = attrib(default=1344)
    controller_layers = attrib(default=1)
    num_heads = attrib(default=1)
    resize_resolution = attrib(default=16)
    input_dim = attrib(default=1)
    output_dim = attrib(default=10)
    memory_n = attrib(default=128)
    memory_m = attrib(default=20)
    num_batches = attrib(default=0)
    batch_size = attrib(default=64)
    rmsprop_lr = attrib(default=1e-4)
    rmsprop_momentum = attrib(default=0.9)
    rmsprop_alpha = attrib(default=0.95)
    device = attrib(default="cpu")
    fraction = attrib(default=0.5)
    use_memory = attrib(default=1.0)
    seq_len = attrib(default=0)

@attrs
class SeqMNISTParams_ntm_cache(object):
    # seq_len = num_head for NTM variant, where we dont interact with the memory each time step.
    name = attrib(default="seq-mnist-ntm-cache")
    model_name = "ntm_cache"
    controller_size = attrib(default=100)
    controller_layers = attrib(default=1)

    # both must be the aligned, i.e. 8*8 sequence len = num_heads
    resize_resolution = attrib(default=8)
    num_heads = attrib(default=0)

    input_dim = attrib(default=1)
    output_dim = attrib(default=10)
    memory_n = attrib(default=128)
    memory_m = attrib(default=20)
    num_batches = attrib(default=0)
    batch_size = attrib(default=32)
    rmsprop_lr = attrib(default=1e-4)
    rmsprop_momentum = attrib(default=0.9)
    rmsprop_alpha = attrib(default=0.95)
    device = attrib(default="cpu")
    fraction = attrib(default=0.8)
    seq_len = attrib(default=0)

@attrs
class SeqMNISTModelTraining_ntm(object):
    params = attrib(default=Factory(SeqMNISTParams_ntm))
    net = attrib()
    dataloader = attrib()
    criterion = attrib()
    optimizer = attrib()

    @net.default
    def default_net(self):
        # We have 1 additional input for the delimiter which is passed on a
        # separate "control" channel
        self.params.seq_len = self.params.resize_resolution**2

        net = EncapsulatedNTM(
            num_inputs=self.params.input_dim,
            num_outputs=self.params.output_dim,
            controller_size=self.params.controller_size,
            controller_layers=self.params.controller_layers,
            num_heads=self.params.num_heads,
            N=self.params.memory_n,
            M=self.params.memory_m,
            device=self.params.device,
            model_architecture=self.params.model_name,
            use_memory=self.params.use_memory,
        )
        return net

    @dataloader.default
    def default_dataloader(self):

        # init the dataset
        dataset = SequentialMNIST({"resize_resolution": self.params.resize_resolution})
        
        # split the dataset into train and val
        size = len(dataset) * self.params.fraction
        train_indices = np.random.choice(len(dataset), int(size), replace=False)
        val_indices = np.setdiff1d(np.arange(len(dataset)), train_indices)
        train_ds = torch.utils.data.Subset(dataset, train_indices)
        val_ds = torch.utils.data.Subset(dataset, val_indices)
        
        # create the train dataloader
        train_dataloader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.params.batch_size,
            shuffle=True,
        )

        # create the validation dataloader
        val_dataloader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=self.params.batch_size,
            shuffle=False,  # No need to shuffle for validation
        )

        self.params.num_batches = len(train_dataloader)
        return train_dataloader, val_dataloader

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

@attrs
class SeqMNISTModelTraining_ntm_cache(object):
    params = attrib(default=Factory(SeqMNISTParams_ntm_cache))
    net = attrib()
    dataloader = attrib()
    criterion = attrib()
    optimizer = attrib()

    @net.default
    def default_net(self):
        # We have 1 additional input for the delimiter which is passed on a
        # separate "control" channel
        self.params.seq_len = self.params.resize_resolution**2
        self.params.num_heads = self.params.seq_len 

        net = EncapsulatedNTM(
            self.params.input_dim,
            self.params.output_dim,
            self.params.controller_size,
            self.params.controller_layers,
            self.params.num_heads,
            self.params.memory_n,
            self.params.memory_m,
            self.params.device,
            self.params.model_name,
            self.params.seq_len,
        )
        return net

    @dataloader.default
    def default_dataloader(self):

        # init the dataset
        dataset = SequentialMNIST({"resize_resolution": self.params.resize_resolution})
        
        # split the dataset into train and val
        size = len(dataset) * self.params.fraction
        train_indices = np.random.choice(len(dataset), int(size), replace=False)
        val_indices = np.setdiff1d(np.arange(len(dataset)), train_indices)
        train_ds = torch.utils.data.Subset(dataset, train_indices)
        val_ds = torch.utils.data.Subset(dataset, val_indices)
        
        # create the train dataloader
        train_dataloader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.params.batch_size,
            shuffle=True,
        )

        # create the validation dataloader
        val_dataloader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=self.params.batch_size,
            shuffle=False,  # No need to shuffle for validation
        )

        self.params.num_batches = len(train_dataloader)
        return train_dataloader, val_dataloader

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


@attrs
class SeqMNISTModelTraining_lstm(object):
    params = attrib(default=Factory(SeqMNISTParams_ntm))
    net = attrib()
    dataloader = attrib()
    criterion = attrib()
    optimizer = attrib()

    @net.default
    def default_net(self):
        # We have 1 additional input for the delimiter which is passed on a
        # separate "control" channel
        self.params.seq_len = self.params.resize_resolution**2
        
        net = LSTMWithLinearLayer(
            self.params.input_dim,
            self.params.controller_size,
            self.params.output_dim,
            self.params.device,
        )
        return net

    @dataloader.default
    def default_dataloader(self):
        # init the dataset
        dataset = SequentialMNIST({"resize_resolution": self.params.resize_resolution})
        
        # split the dataset into train and val
        size = len(dataset) * self.params.fraction
        train_indices = np.random.choice(len(dataset), int(size), replace=False)
        val_indices = np.setdiff1d(np.arange(len(dataset)), train_indices)
        train_ds = torch.utils.data.Subset(dataset, train_indices)
        val_ds = torch.utils.data.Subset(dataset, val_indices)
        
        # create the train dataloader
        train_dataloader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.params.batch_size,
            shuffle=True,
        )

        # create the validation dataloader
        val_dataloader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=self.params.batch_size,
            shuffle=False,  # No need to shuffle for validation
        )

        self.params.num_batches = len(train_dataloader)
        return train_dataloader, val_dataloader

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
