import torch

import argparse
import json
import logging
import time
import random
import re
import sys
import os
import attr
import argcomplete
import torch
import numpy as np
from tqdm import tqdm
import wandb
import datetime 

# Initialize the logger
from utils import progress_bar, progress_clean, get_ms, init_seed, save_checkpoint, clip_grads, update_model_params

def train_batch_ntm(net, criterion, optimizer, X, Y, args):
    """Trains a single batch."""


    # Transfer to GPU
    X = X.to(net.device)
    Y = Y.to(net.device)

    # reset the input sequence and target sequence
    if args.task == 'seq-mnist-ntm' or args.task == "seq-mnist-ntm-s4d":
        X = X.permute(1, 0, 2)
        Y = Y.squeeze(1)
    
    batch_size = X.size(1)
    net.init_sequence(batch_size)
    
    optimizer.zero_grad()
    inp_seq_len = X.size(0)

    # get the size of the output sequence for copy and recall task
    if args.task == 'copy' or args.task == 'repeat-copy':
        outp_seq_len, batch_size, _ = Y.size()
    
    # Feed the sequence + delimiter
    for i in range(inp_seq_len):
        y_out, _ = net(X[i])

    # Read the output (no input given)
    if args.task == 'copy' or args.task == 'repeat-copy':
        y_out = torch.zeros(Y.size())
        for i in range(outp_seq_len):
            out, _ = net()
            y_out[i] = torch.sigmoid(out)

    loss = criterion(y_out, Y)
    loss.backward()
    clip_grads(net)
    optimizer.step()

    # Compute the cost of the output when binary values are expected
    cost = torch.tensor(0)
    if args.task == 'copy' or args.task == 'repeat-copy':
        y_out_binarized = y_out.clone().data
        y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)

        # The cost is the number of error bits per sequence
        cost = torch.sum(torch.abs(y_out_binarized - Y.data))

    return loss.item(), cost.item() / batch_size

def train_batch_ntm_cache(net, criterion, optimizer, X, Y, args):
    """Trains a single batch."""


    # Transfer to GPU
    X = X.to(net.device)
    Y = Y.to(net.device)

    # reset the input sequence and target sequence
    if args.task == 'seq-mnist-ntm-cache':
        X = X.permute(1, 0, 2)
        Y = Y.squeeze(1)
    
    batch_size = X.size(1)
    net.init_sequence(batch_size)
    
    optimizer.zero_grad()

    # Feed the sequence + delimiter
    y_out, _ = net(X)

    loss = criterion(y_out, Y)
    loss.backward()
    clip_grads(net)
    optimizer.step()

    # Compute the cost of the output when binary values are expected
    cost = torch.tensor(0)
 
    return loss.item()

def train_batch_lstm(net, criterion, optimizer, X, Y, args):
    """Trains a single batch."""

    optimizer.zero_grad()

    # Transfer to GPU
    X = X.to(net.device)
    Y = Y.to(net.device)

    X = X.permute(1, 0, 2)
    Y = Y.squeeze(1)

    # Forward
    Y_pred = net(X)

    # Compute the loss
    loss = criterion(Y_pred, Y)
    loss.backward()

    # Clip gradients
    clip_grads(net)

    # Update parameters
    optimizer.step()

    return loss.item()

def train_batch_ntms4d(net, criterion, optimizer, X, Y, args):
    """Trains a single batch."""

    optimizer.zero_grad()

    # Transfer to GPU
    X = X.to(net.device)
    Y = Y.to(net.device)
    Y = Y.squeeze(1)

    batch_size = X.size(0)
    net.init_sequence(batch_size)

    # Forward
    Y_pred, _ = net(X)

    # Compute the loss
    loss = criterion(Y_pred, Y)
    loss.backward()

    # Clip gradients
    clip_grads(net)

    # Update parameters
    optimizer.step()

    return loss.item(), 0
