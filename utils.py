

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
from evaluate import evaluate_ntm, evaluate_lstm, evaluate

from tasks.copytask import CopyTaskModelTraining, CopyTaskParams
from tasks.repeatcopytask import RepeatCopyTaskModelTraining, RepeatCopyTaskParams
from tasks.seq_mnist import SeqMNISTModelTraining_ntm, SeqMNISTModelTraining_lstm, SeqMNISTParams_ntm, SeqMNISTModelTraining_ntm_cache, SeqMNISTParams_ntm_cache


###### Utils for programm logging and progress tracking ######
def progress_bar(batch_num, report_interval, last_loss):
    """Prints the progress until the next report."""
    progress = (((batch_num-1) % report_interval) + 1) / report_interval
    fill = int(progress * 40)
    print("\r[{}{}]: {} (Loss: {:.4f})".format(
        "=" * fill, " " * (40 - fill), batch_num, last_loss), end='')

def progress_clean():
    """Clean the progress bar."""
    print("\r{}".format(" " * 80), end='\r')


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

def save_checkpoint(net, args, model_parms, batch_num, losses, costs, seq_lengths, val_accuracy_list, time, epoch):
    progress_clean()

    basename = "{}/{}--seed-{}-epoch-{}-batch-{}-{}".format(args.checkpoint_path, args.task, args.seed, epoch, batch_num, time)
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
        "val_accuracy_list": val_accuracy_list,
        "parameters_model": vars(model_parms)
    }
    open(train_fname, 'wt').write(json.dumps(content))

###### Utils for model related tasks ######
def clip_grads(net):
    """Gradient clipping to the range [10, 10]."""
    parameters = list(filter(lambda p: p.grad is not None, net.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-1, 1)

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
        if k == 'use_memory':
            update_dict[k] = float(v)
        elif k == "batch_size":
            update_dict[k] = int(v)
        else:
            pass

    try:
        params = attr.evolve(params, **update_dict)
    except TypeError as e:
        LOGGER.error(e)
        LOGGER.error("Valid parameters: %s", list(attr.asdict(params).keys()))
        sys.exit(1)

    return params