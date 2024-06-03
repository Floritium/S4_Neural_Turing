import json
from tqdm import tqdm
import numpy as np
import os
import argparse

import torch
from torch import nn, optim

from ntm import NTM
from task_generator import CopyDataset, AssociativeDataset
from argparser import get_args

# ==== Arguments ====
args = get_args()


# ==== Create Dataset ====
# Copy Task
task_params = json.load(open(args.task_json))
dataset = CopyDataset(task_params)
# Associative Recall Task
# dataset = AssociativeDataset(task_params)

# ==== Create NTM ====
ntm = NTM(input_dim=task_params['seq_width'] + 2,
          output_dim=task_params['seq_width'],
          ctrl_dim=task_params['controller_size'],
          memory_units=task_params['memory_units'],
          memory_unit_size=task_params['memory_unit_size'],
          num_heads=task_params['num_heads'])

# ==== Training Settings ====
# Loss Function
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
    
    # Sample data
    data = dataset[step]
    inputs, target = data['input'], data['target']
    
    # Tensor to store outputs
    out = torch.zeros(target.size())
    
    # Process the inputs through NTM for memorization
    for i in range(inputs.size()[0]):
        # Forward passing all sequences for read
        ntm(inputs[i].unsqueeze(0))
        
    # Get the outputs from memory without real inputs
    zero_inputs = torch.zeros(inputs.size()[1]).unsqueeze(0) # dummy inputs
    for i in range(target.size()[0]):
        out[i] = ntm(zero_inputs)
    
    # Compute loss, backprop, and optimize
    loss = criterion(out, target)
    losses.append(loss.item())
    loss.backward()
    nn.utils.clip_grad_value_(ntm.parameters(), 10)
    optimizer.step()
    
    # Calculate binary outputs
    binary_output = out.clone()
    binary_output = binary_output.detach().apply_(lambda x: 0 if x < 0.5 else 1)
    
    # Sequence prediction error is calculted in bits per sequence
    error = torch.sum(torch.abs(binary_output - target))
    errors.append(error.item())
    
    # Print Stats
    if step % 200 == 0:
        print('Step {} == Loss {:.3f} == Error {} bits per sequence'.format(step, np.mean(losses), np.mean(errors)))
        losses = []
        errors = []
        
# Save model
torch.save(ntm.state_dict(), args.saved_model)
    
    