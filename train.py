import json
from tqdm import tqdm
import numpy as np
import os
import argparse

import torch
from torch import nn, optim

from ntm import NTM
from task_generator import CopyDataset, AssociativeDataset, SequentialMNIST
from argparser import get_args

import wandb
import datetime

# ==== Arguments ====
args = get_args()

# ==== Set the device ====
# Check if MPS (Apple Silicon) is available
if args.device == True and torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device('mps')
# Check if CUDA (NVIDIA GPU) is available
elif args.device == True and torch.cuda.is_available():
    device = torch.device('cuda')
# Fall back to CPU
else:
    device = torch.device('cpu')

print('Using device:', device)

# ==== seed ====
def seed_everything(seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(args.seed)

# Step 1: Open the JSON file
with open(args.task_json, 'r') as file:
    # Step 2: Load the JSON data
    task_params = json.load(file)


def progress_clean():
    """Clean the progress bar."""
    print("\r{}".format(" " * 80), end='\r')

def save_checkpoint(net, task_params, args, step, losses, costs, seq_lengths, time):
    progress_clean()

    basename = "{}/{}-{}-{}-batch-{}-{}".format(args.checkpoint_path, task_params["model"], task_params["task"], args.seed, step, time)
    model_fname = basename + ".pth"
    print(f"Saving model checkpoint to: {model_fname}")
    torch.save(net.state_dict(), model_fname)

    # Save the training history
    train_fname = basename + ".json"
    print(f"Saving model training history to {train_fname}")
    content = {
        "loss": losses,
        "cost": costs,
        "seq_lengths": seq_lengths,
        "parameters_model": vars(args),
        "parameters_task": task_params, # task_params is already a type dict
    }
    open(train_fname, 'wt').write(json.dumps(content))

# ==== Create Dataset / task ====
task_params = json.load(open(args.task_json)) # Load task parameters

# Create dataset
if task_params['task'] == 'copy':
    dataset = CopyDataset(task_params)
elif task_params['task'] == 'associative':
    dataset = AssociativeDataset(task_params)
elif task_params['task'] == 'seq_mnist':
    dataset = SequentialMNIST(task_params, train=True)
    print('Dataset size:', len(dataset))

# ==== Create NTM ====
if task_params['task'] == 'copy' or task_params['task'] == 'associative':
    ntm = NTM(input_dim=task_params['seq_width'] + 2,
              output_dim=task_params['seq_width'],
              ctrl_dim=task_params['controller_size'],
              memory_units=task_params['memory_units'],
              memory_unit_size=task_params['memory_unit_size'],
              num_heads=task_params['num_heads'], device=device).to(device)
elif task_params['task'] == 'seq_mnist':
    ntm = NTM(input_dim=task_params['seq_width'],
            output_dim=task_params['output_dim'],
            ctrl_dim=task_params['controller_size'],
            memory_units=task_params['memory_units'],
            memory_unit_size=task_params['memory_unit_size'],
            num_heads=task_params['num_heads'], device=device).to(device)

if args.resume_training:
    file_name = args.saved_model

    task_params = json.load(open(file_name))
    task_params = task_params['parameters_task']

    ntm = NTM(input_dim=task_params['seq_width'],
            output_dim=task_params['output_dim'],
            ctrl_dim=task_params['controller_size'],
            memory_units=task_params['memory_units'],
            memory_unit_size=task_params['memory_unit_size'],
            num_heads=task_params['num_heads'], device=device).to(device)

    # Add the new extension
    base = os.path.splitext(file_name)[0]
    new_file_name = base + ".pth"

    ntm.load_state_dict(torch.load(new_file_name))
    print("Model loaded from", new_file_name)


# ==== Training Settings ====
# Loss Function
if task_params['task'] == 'seq_mnist':
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.BCELoss()

# optimizer = optim.RMSprop(ntm.parameters(),
#                           lr=args.lr,
#                           alpha=args.alpha,
#                           momentum=args.momentum)

optimizer = optim.Adam(ntm.parameters(), lr=args.lr,
                       betas=(args.beta1, args.beta2))


# === WandB ===

# Initialize wandb and set the configuration
wandb.init(project="nn-seminar", 

    config={
    "task_json": args.task_json,
    "saved_model": args.saved_model,
    "batch_size": args.batch_size,
    "num_steps": args.num_steps,
    "learning_rate": args.lr,
    "momentum": args.momentum,
    "alpha": args.alpha,
    "beta1": args.beta1,
    "beta2": args.beta2,
    "seed": args.seed,
    "device": args.device,
    "eval_steps": args.eval_steps,
    "checkpoint_path": args.checkpoint_path,
    "checkpoint_interval": args.checkpoint_interval,
    },    
    mode="online" if args.log else "disabled"
)


# ==== Training ====

# log time for chekpoints
time = ''.join(str(datetime.datetime.now()).split())

losses = []
errors = []
ntm.to(device)
ntm.train()

for step in tqdm(range(args.num_steps)):
    
    optimizer.zero_grad()
    ntm.reset()
    
    # Sample data
    data = dataset[step]
    inputs, target = data['input'], data['target']
    inputs, target = inputs.to(device), target.to(device)
    
    # Tensor to store outputs
    out = torch.zeros(target.size() if task_params['task'] != 'seq_mnist' else task_params['output_dim'])
    
    # Process the inputs through NTM for memorization
    for i in range(inputs.size()[0]):
        # Forward passing all sequences for read
        # print(inputs[i].unsqueeze(0).shape, target.shape)
        ntm(inputs[i].unsqueeze(0), memorize=True)
    

    zero_inputs = torch.zeros(inputs.size()[1]).unsqueeze(0).to(device) # dummy inputs for reading memory

    # Get the outputs from memory without real inputs
    if task_params['task'] == 'seq_mnist':
        for i in range(inputs.size()[0]):
            out = ntm(zero_inputs, memorize=False) # logits for cross entropy loss criterion
    elif task_params['task'] == 'copy' or task_params['task'] == 'associative':
        for i in range(target.size()[0]):
            out[i] = torch.sigmoid(ntm(zero_inputs, memorize=False)) # sigmoid the logits
    
    # Compute loss, backprop, and optimize
    out = out.to(device)
    loss = criterion(out, target)
    losses.append(loss.item())
    loss.backward()
    nn.utils.clip_grad_value_(ntm.parameters(), 0.5)
    optimizer.step()
    
    # Calculate binary outputs
    binary_output = out.clone()
    binary_output = binary_output.cpu().detach().apply_(lambda x: 0 if x < 0.5 else 1)
    
    # Sequence prediction error is calculted in bits per sequence
    error = torch.sum(torch.abs(binary_output.cpu() - target.cpu()))
    errors.append(error.item())
        
    # Print Stats
    if step % args.eval_steps == 0:
        print('Step {} == Loss {:.3f} == Error {} bits per sequence'.format(step, np.mean(losses[-args.eval_steps:]), np.mean(errors[-args.eval_steps:])))
        wandb.log({"Training/Loss": np.mean(losses[-args.eval_steps:]), "Training/Error": np.mean(errors[-args.eval_steps:])}, step=step)

    # save checkpoint
    if (args.checkpoint_interval != 0) and (step % args.checkpoint_interval == 0):
        save_checkpoint(ntm, task_params, args, step, losses, errors, inputs.size()[0], time)
    
    