#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Training for the Copy Task in Neural Turing Machines."""

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

# Initialize the logger
LOGGER = logging.getLogger(__name__)
from evaluate import evaluate_ntm, evaluate_lstm, evaluate
from utils import progress_bar, progress_clean, get_ms, init_seed, save_checkpoint, clip_grads, update_model_params

# Import the tasks
from tasks.copytask import CopyTaskModelTraining, CopyTaskParams
from tasks.repeatcopytask import RepeatCopyTaskModelTraining, RepeatCopyTaskParams
from tasks.seq_mnist import SeqMNISTModelTraining_ntm, SeqMNISTModelTraining_lstm, SeqMNISTParams_ntm, SeqMNISTModelTraining_ntm_cache, SeqMNISTParams_ntm_cache, SeqMNISTParams_ntm_s4d

TASKS = {
    'copy': (CopyTaskModelTraining, CopyTaskParams),
    'repeat-copy': (RepeatCopyTaskModelTraining, RepeatCopyTaskParams),
    'seq-mnist-ntm-cache': (SeqMNISTModelTraining_ntm_cache, SeqMNISTParams_ntm_cache),
    'seq-mnist-ntm': (SeqMNISTModelTraining_ntm, SeqMNISTParams_ntm), # its basically also cache, as use_memory can be set between [0,1]
    'seq-mnist-lstm': (SeqMNISTModelTraining_lstm, SeqMNISTParams_ntm)
    'seq-mnist-ntm-s4d': (SeqMNISTModelTraining_ntm, SeqMNISTParams_ntm_s4d)
}


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
    if args.task != 'seq-mnist-ntm': or args.task != "seq-mnist-ntm-s4d"
        outp_seq_len, batch_size, _ = Y.size()

    # Feed the sequence + delimiter
    for i in range(inp_seq_len):
        y_out, _ = net(X[i])

    # Read the output (no input given)
    if args.task != 'seq-mnist-ntm' or or args.task != "seq-mnist-ntm-s4d":
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

def train_model(model, args):
    # Get the number of batches
    num_batches = model.params.num_batches
    batch_size = model.params.batch_size
    num_samples = num_batches * batch_size
    LOGGER.info("Training model for %d batches (batch_size=%d - num_samples=%d)...",
                num_batches, batch_size, num_samples)

    # Initialize the progress bar
    losses = []
    costs = []
    seq_lengths = []
    start_ms = get_ms()
    val_accuracy_list = []
    time = ''.join(str(datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")).split())

    # make path dependent on run
    args.checkpoint_path = args.checkpoint_path + "/" + time + "/" + args.task + "-" + str(model.params.seq_len)
    os.makedirs(args.checkpoint_path, exist_ok=True)

    # get the dataloaders
    train_loader, val_loader = model.dataloader

    model.net.to_device() # move the model to the device (must be optimized!)
    for epoch in range(1, args.epochs+1):
        model.net.train() # set the model to training mode
        for batch_num, (x, y) in enumerate(tqdm(train_loader)):
            if args.task == 'seq-mnist-ntm' or args.task == 'copy' or args.task == 'repeat-copy':
                loss, cost = train_batch_ntm(model.net, model.criterion, model.optimizer, x, y, args)
            if args.task == 'seq-mnist-ntm-cache':
                loss = train_batch_ntm_cache(model.net, model.criterion, model.optimizer, x, y, args)
                cost = loss
            elif args.task == 'seq-mnist-lstm':
                loss = train_batch_lstm(model.net, model.criterion, model.optimizer, x, y, args)
                cost = loss
            
            losses += [loss]
            costs += [cost]
            seq_lengths += [y.size(0)]

            # Update the progress bar
            if not isinstance(model.dataloader, torch.utils.data.DataLoader):
                progress_bar(batch_num, args.report_interval, loss)

            # Report
            if batch_num % args.report_interval == 0:
                mean_loss = np.array(losses[-args.report_interval:]).mean()
                mean_cost = np.array(costs[-args.report_interval:]).mean()
                mean_time = int(((get_ms() - start_ms) / args.report_interval) / batch_size)
                progress_clean()
                LOGGER.info("Batch %d Loss: %.6f Cost: %.2f Time: %d ms/sequence",
                            batch_num * x.size(0), mean_loss, mean_cost, mean_time)
                start_ms = get_ms()

            # Checkpoint
            if (args.checkpoint_interval != 0) and (batch_num % args.checkpoint_interval == 0):
                save_checkpoint(model.net, args, model.params, batch_num, losses, costs, seq_lengths, val_accuracy_list, time, epoch)
        
        if (epoch % args.validation_interval) == 0 and args.validation_interval > 0:
            model.net.eval() # set the model to evaluation mode
            print("Validating the model...")
            if args.task == 'seq-mnist-lstm':
                val_accuracy = evaluate_lstm(model.net, val_loader, model.criterion, args)
            elif args.task == "seq-mnist-ntm":
                val_accuracy = evaluate_ntm(model.net, val_loader, model.criterion, args)
            
            val_accuracy_list += [val_accuracy]
            LOGGER.info(f"Validation accuracy: {val_accuracy}%")

    # last ceckpoint
    save_checkpoint(model.net, args, model.params, batch_num, losses, costs, seq_lengths, val_accuracy_list, time, epoch)
    LOGGER.info("Done training.")

def init_model(args):
    LOGGER.info("Training for the **%s** task", args.task)

    model_cls, params_cls = TASKS[args.task]
    params = params_cls()
    params = update_model_params(params, args.param)

    LOGGER.info(params)

    wandb.init(project="nn-seminar", 
        config=vars(params),
        mode="online" if args.log else "disabled"
    )

    model = model_cls(params=params)
    return model

def init_logging():
    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(name)s]  %(message)s',
                        level=logging.DEBUG)


def main():
    init_logging()

    # Create the checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)

    # Initialize arguments
    args = init_arguments()

    # Initialize random
    init_seed(args.seed)

    # Initialize the model
    model = init_model(args)

    LOGGER.info("Total number of parameters: %d", model.net.calculate_num_params())
    train_model(model, args)


if __name__ == '__main__':
    main()
