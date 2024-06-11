import argparse
import json
import logging
import time
import random
import re
import sys
import os
from arg_parser import init_arguments

import attr
import argcomplete
import torch
import numpy as np
from tqdm import tqdm
import wandb
import datetime

LOGGER = logging.getLogger(__name__)


from tasks.copytask import CopyTaskModelTraining, CopyTaskParams
from tasks.repeatcopytask import RepeatCopyTaskModelTraining, RepeatCopyTaskParams
from tasks.seq_mnist import (
    SeqMNISTModelTraining_ntm,
    SeqMNISTModelTraining_lstm,
    SeqMNISTParams_ntm,
    SeqMNISTModelTraining_ntm_cache,
    SeqMNISTParams_ntm_cache,
)


def evaluate_ntm(net, val_loader, criterion, args):
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for X, Y in val_loader:
            # reset the input sequence and target sequence
            X = X.permute(1, 0, 2)
            Y = Y.squeeze(1)

            batch_size = X.size(1)
            net.init_sequence(batch_size)
            inp_seq_len = X.size(0)

            # Feed the sequence + delimiter
            for i in range(inp_seq_len):
                outputs, _ = net(X[i])

            loss = criterion(outputs, Y)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += Y.size(0)
            correct += (predicted == Y).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def evaluate_lstm(net, val_loader, criterion, args):
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for X, Y in val_loader:
            X = X.permute(1, 0, 2)
            Y = Y.squeeze(1)

            outputs = net(X)
            loss = criterion(outputs, Y)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += Y.size(0)
            correct += (predicted == Y).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def evaluate(net, criterion, X, Y):
    """Evaluate a single batch (without training)."""
    inp_seq_len = X.size(0)
    outp_seq_len, batch_size, _ = Y.size()

    # New sequence
    net.init_sequence(batch_size)

    # Feed the sequence + delimiter
    states = []
    for i in range(inp_seq_len):
        o, state = net(X[i])
        states += [state]

    # Read the output (no input given)
    y_out = torch.zeros(Y.size())
    for i in range(outp_seq_len):
        y_out[i], state = net()
        states += [state]

    loss = criterion(y_out, Y)

    y_out_binarized = y_out.clone().data
    y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)

    # The cost is the number of error bits per sequence
    cost = torch.sum(torch.abs(y_out_binarized - Y.data))

    result = {
        "loss": loss.data[0],
        "cost": cost / batch_size,
        "y_out": y_out,
        "y_out_binarized": y_out_binarized,
        "states": states,
    }

    return result
