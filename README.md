# Structured state space models

## What are SSMs?

![Representation of the linear state-space equations](image.png)

The mathematical basis of structured state space models (SSMs) originates from control theory and signal processing, where the **system dynamics are often described using continuous differential equations.**


## S4 (Structured State Space sequence model)
![Alt text](image-1.png)
![Alt text](image-2.png) (https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state#footnote-3-141228095)

S4 uses structured matrices (A Hippo) derived from the state-space model to approximate the continuous-time convolution operator efficiently. A promising recent approach proposed modeling sequences by simulating the fundamental state space model (SSM) x′(t) = Ax(t) + Bu(t), y(t) = Cx(t) + Du(t), and showed that for appropriate choices of the state matrix A, this system could handle long-range dependencies mathematically and empirically.

> Remark: SSMs are a intersection of RNNs and CNNs!!!

### Views of SMMs in S4
SSMs are models with three views. A continuous view, and when discretized, a recurrent as well as a convolutive view. One of S4 strengths is its ability to handle very long sequences, generally with a lower number of parameters than other models (ConvNet or transformers), while still being very fast. Given a sequence during inference, S4 makes
a autoregressive predicition of the next "token". But during training, the model sees the sequence at ones (across all timesteps at the sametime) and can therefore switch to convolution setting at train in parallel. RNN and CNN mode in S4 are about computation, but for modeling perspective (i.e. problem solvability in handling continues-time signals) the inductive bias in S4 is really low, hence the model performs better than other specialized models. Hence develop NN-models which have a mathematical framework (her SSM), which have general low inductive bias in a certain application area. S4 is just a layer (building block) for Neural network models.

### Differences between existing SSM architectures
the main differences between the various existing SSM architectures lie in the way the basic SSM equation is discretized, or in the definition of the, A matrix.

### Why continues-time to Discrete-time, when computers store, dicrete data?
**Short:** We need to compute the discrete equivalents by approximating the continuous differential equations $\rightarrow$ (That's why there are all these mathematical transformations).

**Remark:**
Why does the S4 architekture for strucutred state space models, deals with the discretization of continues-time, when the data stored on the computer is discrete?
1. The continuous formulation provides a robust mathematical framework and discretization allows accurate approximation of continuous dynamics.
2. Discretizing continuous-time equations allows for a better representation of the underlying continuous dynamics when processing discrete data. 
3. The careful discretization ensures that S4 can leverage the strengths of continuous signal processing while remaining practical for discrete computation.
4. Traditional RNNs and other sequence models expect data points at regular intervals. When data is irregularly sampled, these models either need data imputation or zero-padding, which can introduce biases and artifacts!!!
5. The linar ODEs, gets translated to Discrete-Time ODEs: It efficiently translates the continuous-time dynamics into discrete time, maintaining the long-range dependency characteristics. 

## Objective in this work 
Different discretization methods could offer improvements in:
- **Numerical Stability:** Important for very long sequences to prevent state explosion or decay.
- **Accuracy:** Preserving the continuous-time dynamics accurately for better modeling of long-range dependencies.
- **Computational Efficiency:** Faster computation of the state matrices, especially for real-time or large-scale applications.
- **Remark:** Use a simpler version of S4, i.e. instead of the heavy math proposed in the S4 paper to derive the Kernel, instead use S4D (S4 Diagnonal state space using diag.approx) implementation (implement your own).


"Hidden" objective for us: understand the re-implmenent this step (from the annotated S4 blog/paper: https://srush.github.io/annotated-s4/#part-2-implementing-s4)


# Quick links
- <u>_State Space models_</u>
  - [x] https://huggingface.co/blog/lbourdois/get-on-the-ssm-train
  - [ ] https://srush.github.io/annotated-s4/ (Annotated S4)
  - [x] https://hazyresearch.stanford.edu/blog/2022-01-14-s4-1
  - [x] https://hazyresearch.stanford.edu/blog/2022-01-14-s4-2
  - [x] https://hazyresearch.stanford.edu/blog/2022-01-14-s4-3
  - [x] https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state (intro to mambda build from s4 -> really good)
  - [ ] https://youtu.be/H0KrEZC9eyA?si=M4XWBjLZT6hcyn-X (Video of s4 from a seminar)
  - [x] https://www.youtube.com/watch?v=EvQ3ncuriCM (Video of s4 by the author itself)
  - [x] https://www.youtube.com/watch?v=OpJMn8T7Z34&t=465s (good video of S4 and its variants from albert gu)




- <u>_RNNs / Transformers_</u>
  - [x] https://www.youtube.com/watch?v=ZVN14xYm7JA&list=PL1J3bsLH2E_xJM46XPOwuUgJulU9J_-zm (Sequence Modeling with RNNs. Goodfellow.)
  - [x] https://karpathy.github.io/2015/05/21/rnn-effectiveness/ (Recurrent Neural Networks from andrej kaparthy)
  - [ ] https://colah.github.io/posts/2015-08-Understanding-LSTMs/ (LSTMs)
  - [ ] https://nlp.seas.harvard.edu/2018/04/03/attention.html (Annotated Tranformer)
  - [ ] https://www.youtube.com/watch?v=AIiwuClvH6k (Attention mechanims)