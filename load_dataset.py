import json

import json
from tqdm import tqdm
import numpy as np
import os
import argparse

import torch
from torch import nn, optim
from torchvision import transforms as T
import json

from tasks.seq_mnist import SeqMNISTModelTraining, SeqMNISTParams, SequentialMNIST

# load test dataset
task_params = {
    "resize_resolution": 14,
}
test_dataset = SequentialMNIST(task_params, train=True)
