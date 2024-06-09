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

LOGGER = logging.getLogger(__name__)


from tasks.copytask import CopyTaskModelTraining, CopyTaskParams
from tasks.repeatcopytask import RepeatCopyTaskModelTraining, RepeatCopyTaskParams
from tasks.seq_mnist import SeqMNISTModelTraining, SeqMNISTParams

TASKS = {
    'copy': (CopyTaskModelTraining, CopyTaskParams),
    'repeat-copy': (RepeatCopyTaskModelTraining, RepeatCopyTaskParams),
    'seq-mnist': (SeqMNISTModelTraining, SeqMNISTParams)
}


def get_ms():
    """Returns the current time in miliseconds."""
    return time.time() * 1000


def init_seed(seed=None):
    """Seed the RNGs for predicatability/reproduction purposes."""
    if seed is None:
        seed = int(get_ms() // 1000)

    LOGGER.info("Using seed=%d", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def progress_clean():
    """Clean the progress bar."""
    print("\r{}".format(" " * 80), end='\r')


def progress_bar(batch_num, report_interval, last_loss):
    """Prints the progress until the next report."""
    progress = (((batch_num-1) % report_interval) + 1) / report_interval
    fill = int(progress * 40)
    print("\r[{}{}]: {} (Loss: {:.4f})".format(
        "=" * fill, " " * (40 - fill), batch_num, last_loss), end='')


def save_checkpoint(net, name, args, model_parms, batch_num, losses, costs, seq_lengths, time):
    progress_clean()

    basename = "{}/{}-{}-batch-{}-{}".format(args.checkpoint_path, name, args.seed, batch_num, time)
    model_fname = basename + ".pth"
    LOGGER.info("Saving model checkpoint to: '%s'", model_fname)
    torch.save(net.state_dict(), model_fname)

    # Save the training history
    train_fname = basename + ".json"
    LOGGER.info("Saving model training history to '%s'", train_fname)
    content = {
        "loss": losses,
        "cost": costs,
        "seq_lengths": seq_lengths,
        "parameters_model": vars(model_parms)
    }
    open(train_fname, 'wt').write(json.dumps(content))


def clip_grads(net):
    """Gradient clipping to the range [10, 10]."""
    parameters = list(filter(lambda p: p.grad is not None, net.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-1, 1)


def train_batch(net, criterion, optimizer, X, Y, args):
    """Trains a single batch."""

    # Transfer to GPU
    X = X.to(net.device)
    Y = Y.to(net.device)

    # reset the input sequence and target sequence
    if args.task == 'seq-mnist':
        X = X.permute(1, 0, 2)
        Y = Y.squeeze(1)
    
    optimizer.zero_grad()
    inp_seq_len = X.size(0)

    # get the size of the output sequence for copy and recall task
    if args.task != 'seq-mnist':
        outp_seq_len, batch_size, _ = Y.size()

    # New sequence
    batch_size = X.size(1)
    net.init_sequence(batch_size)
    # net.to_device()

    # Feed the sequence + delimiter
    for i in range(inp_seq_len):
        y_out, _ , = net(X[i])

    # Read the output (no input given)
    if args.task != 'seq-mnist':
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
        'loss': loss.data[0],
        'cost': cost / batch_size,
        'y_out': y_out,
        'y_out_binarized': y_out_binarized,
        'states': states
    }

    return result


def train_model(model, args):
    num_batches = model.params.num_batches
    batch_size = model.params.batch_size
    num_samples = num_batches * batch_size

    LOGGER.info("Training model for %d batches (batch_size=%d - num_samples=%d)...",
                num_batches, batch_size, num_samples)

    losses = []
    costs = []
    seq_lengths = []
    start_ms = get_ms()

    time = ''.join(str(datetime.datetime.now()).split())

    for batch_num, (x, y) in enumerate(tqdm(model.dataloader)):
        loss, cost = train_batch(model.net, model.criterion, model.optimizer, x, y, args)
        
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
            save_checkpoint(model.net, model.params.name, args, model.params, batch_num, losses, costs, seq_lengths, time)

    LOGGER.info("Done training.")



def update_model_params(params, update):
    """Updates the default parameters using supplied user arguments."""

    update_dict = {}
    for p in update:
        m = re.match("(.*)=(.*)", p)
        if not m:
            LOGGER.error("Unable to parse param update '%s'", p)
            sys.exit(1)


        k, v = m.groups()
        print(getattr(params, k))
        update_dict[k] = int(v) if isinstance(getattr(params, k), int) else v

    try:
        params = attr.evolve(params, **update_dict)
    except TypeError as e:
        LOGGER.error(e)
        LOGGER.error("Valid parameters: %s", list(attr.asdict(params).keys()))
        sys.exit(1)

    return params

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
