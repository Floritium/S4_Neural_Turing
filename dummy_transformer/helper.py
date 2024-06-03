import torch

def positional_encoding(n_embd, seq_len):
  # create a matrix of shape (seq_len, n_embd) filled with zeros
  pe = torch.zeros(seq_len, n_embd)

  # create an array of shape (seq_len,) with values from 0 to seq_len
  position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

  # create an array of shape (n_embd//2,) with values from 0 to n_embd//2
  div_term = torch.exp(torch.arange(0, n_embd, 2).float() * (-torch.log(torch.tensor(10000.0)) / n_embd))

  # fill the even indices with sin values and the odd indices with cos values
  pe[:, 0::2] = torch.sin(position * div_term)
  pe[:, 1::2] = torch.cos(position * div_term)

  # add a batch dimension to the positional encoding
  pe = pe.unsqueeze(0)

  return pe  #shape (1, seq_len, n_embd)