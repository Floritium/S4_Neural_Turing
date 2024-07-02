# Structured state space neural turing machine

## What is S4D-NTM?
The structured state space neural turing machine (S4D-NTM) is a neural network model that uses structured matrices derived from the state-space model. Specifically the model is based on the structured state space model (S4D) and is designed to handle long-range dependencies in sequences via the hidden state from the structured matrices, which is stored on the Memory bank for later retrieval. The S4D-NTM model is a variant of the NTM model that uses structured matrices to improve the performance of the model on tasks that require long-range dependencies.

Ideas: https://app.diagrams.net/#G1cHGAA4ybM5Cu00LdYxlF6Wx5bUun8e5k#%7B%22pageId%22%3A%22vaXz-TmzTlR2uNQX9Hex%22%7D

Credit: adapted code from https://github.com/loudinthecloud/pytorch-ntm

## Task
- [ ] Understand the structured state space neural turing machine (S4D-NTM) and its application in the context of the S4 model.
- [ ] Implement the S4D-NTM model in PyTorch.
- [ ] Train the model on a simple task to understand its working.
  - [x] Baseline plain NTM model.
    - [x] Implement the copy task.
    - [x] Implement the sequential task.
  - [x] S4D-NTM.
    - [ ] Implement the copy task.
    - [x] Implement the sequential tasks.
  - [x] LSTM on the same tasks (maybe use reference chart from the S4 paper).
    - [x] Implement the copy task.
    - [x] Implement the sequential tasks.

## Temp
- [ ] NTM-S4D, 256-seq, 5 epochs 
- [x] NTM-S4D, 784-seq, 5 epochs (checkpoints/2024-07-0211-36-16/seq-mnist-ntm-s4d--seed-1000-epoch-5-batch-5399-2024-07-0211-36-16.json)
- [ ] S4D, 256-seq, 5 epochs
- [ ] S4D, 784-seq, 5 epochs

## Learning material: Matrix computations and operations
- [LOW-RANK MATRICES](https://www.ethanepperly.com/index.php/2021/10/26/big-ideas-in-applied-math-low-rank-matrices/)
- [NYSTRÖM APPROXIMATION](https://www.ethanepperly.com/index.php/2022/10/11/low-rank-approximation-toolbox-nystrom-approximation/)

## Training
Train the model using the following tasks:
```bash
python train.py --task seq-mnist-ntm --checkpoint_interval 20 --report_interval 10 -pbatch_size=64 --epochs=1 --validation_interval=0 --seed 1000
...
train.py --task seq-mnist-ntm --checkpoint_interval 100 --report_interval 100 -pbatch_size=10 -puse_memory=1 --epochs=3 --validation_interval=1 --seed 1000
...
python train.py --task seq-mnist-lstm --checkpoint_interval 20 --report_interval 10 -pbatch_size=64 --epochs=100 --validation_interval=0 --seed 1000
```