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

def save_checkpoint(net, name, args, step, losses, costs, seq_lengths):
    progress_clean()

    basename = "{}/{}-{}-batch-{}".format(args.checkpoint_path, name, args.seed, step)
    model_fname = basename + ".pth"
    print(f"Saving model checkpoint to: {model_fname}")
    torch.save(net.state_dict(), model_fname)

    # Save the training history
    train_fname = basename + ".json"
    print(f"Saving model training history to {train_fname}")
    content = {
        "loss": losses,
        "cost": costs,
        "seq_lengths": seq_lengths
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
    dataset = SequentialMNIST(task_params)

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


# ==== Training Settings ====
# Loss Function
if task_params['task'] == 'seq_mnist':
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.BCELoss()

optimizer = optim.RMSprop(ntm.parameters(),
                          lr=args.lr,
                          alpha=args.alpha,
                          momentum=args.momentum)

# optimizer = optim.Adam(ntm.parameters(), lr=args.lr,
#                        betas=(args.beta1, args.beta2))

# ==== Training ====
losses = []
errors = []

for step in tqdm(range(args.num_steps)):
    
    optimizer.zero_grad()
    ntm.reset()
    ntm.to(device)
    
    # Sample data
    data = dataset[step]
    inputs, target = data['input'], data['target']
    inputs, target = inputs.to(device), target.to(device)
    
    # Tensor to store outputs
    out = torch.zeros(target.size() if task_params['task'] != 'seq_mnist' else task_params['output_dim'])
    
    # Process the inputs through NTM for memorization
    for i in range(inputs.size()[0]):
        # Forward passing all sequences for read
        ntm(inputs[i].unsqueeze(0))
        
    # Get the outputs from memory without real inputs
    if task_params['task'] == 'seq_mnist':
        zero_inputs = torch.zeros(inputs.size()[1]).unsqueeze(0).to(device)
        for i in range(inputs.size()[0]):
            out = ntm(zero_inputs) # logits for cross entropy loss criterion
    
    elif task_params['task'] == 'copy' or task_params['task'] == 'associative':
        zero_inputs = torch.zeros(inputs.size()[1]).unsqueeze(0).to(device) # dummy inputs for reading memory
        for i in range(target.size()[0]):
            out[i] = torch.sigmoid(ntm(zero_inputs)) # sigmoid the logits
    
    # Compute loss, backprop, and optimize
    out = out.to(device)
    loss = criterion(out, target)
    losses.append(loss.item())
    loss.backward()
    nn.utils.clip_grad_value_(ntm.parameters(), 1)
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
    
    # save checkpoint
    if (args.checkpoint_interval != 0) and (step % args.checkpoint_interval == 0):
        save_checkpoint(ntm, task_params['task'], args, step, losses, errors, inputs.size()[0])
    
    