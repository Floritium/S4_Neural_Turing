# Structured state space neural turing machine

## What is S4D-NTM?
The structured state space neural turing machine (S4D-NTM) is a neural network model that uses structured matrices derived from the state-space model. Specifically the model is based on the structured state space model (S4D) and is designed to handle long-range dependencies in sequences via the hidden state from the structured matrices, which is stored on the Memory bank for later retrieval. The S4D-NTM model is a variant of the NTM model that uses structured matrices to improve the performance of the model on tasks that require long-range dependencies.

## Task
- [ ] Understand the structured state space neural turing machine (S4D-NTM) and its application in the context of the S4 model.
- [ ] Implement the S4D-NTM model in PyTorch.
- [ ] Train the model on a simple task to understand its working.
  - [x] Baseline plain NTM model.
    - [x] Implement the copy task.
    - [x] Implement the sequential task.
  - [ ] S4D-NTM.
    - [ ] Implement the copy task.
    - [ ] Implement the sequential tasks.
  - [ ] LSTM on the same tasks (maybe use reference chart from the S4 paper).
    - [ ] Implement the copy task.
    - [ ] Implement the sequential tasks.