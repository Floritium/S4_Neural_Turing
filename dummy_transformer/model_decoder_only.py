import torch
import torch.nn as nn
import torch.nn.functional as F
from dummy.helper import positional_encoding
from dummy.tokenizer import decode

class Head(nn.Module):
  def __init__(self, head_size, n_embd, seq_len, dropout=0.1, inference=False):
    super().__init__()
    self.key   = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(seq_len, seq_len))) # register but not compute gradients (i.e. not in model.paramters())
    self.dropout = nn.Dropout(dropout) #dropout randomly prevents some tokens from communicating with each other
    self.inference = inference # choose state of computation (inference or training)

  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x) #shape (B,T, head_size)
    q = self.query(x) #shape (B,T, head_size)
    v = self.value(x) #shape (B,T, head_size)

    #compute self-attention scores
    w = q @ k.transpose(-2, -1) #shape (B,T, head_size) @ (B,head_size,T) --> (B,T,T)
    w *= C**-0.5 #scale by sqrt(d_k), so that variance of the w is 1

    # mask the upper triangular part of the matrix to prevent the model from cheating
    if self.inference == False:
      w = w.masked_fill(self.tril[:T,:T]==0, float('-inf')) # (B,T,T)

    w = F.softmax(w, dim=-1) # (B, T, T)
    w = self.dropout(w)

    #perform weighted aggregation of values
    out = w @ v # (B, T, T) @ (B, T, head_size) --> (B, T, head_size)
    return out
  
class Multi_head(nn.Module):
  def __init__(self, n_heads, head_size, n_embd, seq_len, dropout=0.1, inference=False):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size, n_embd, seq_len, dropout) for _ in range(n_heads)]) # apply the head module n_heads times each with different weights (independent heads)
    self.fc = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    x = torch.cat([h(x) for h in self.heads], dim=-1) #shape (B,T, n_heads*head_size = n_embd)
    x = self.fc(x) #shape (B, T, n_embd) (compute along the T dimension)
    return self.dropout(x)
  
class FeedForward(nn.Module):
  """ the feed forward network (FFN) in the paper"""

  def __init__(self, n_embd, dropout=0.1):
    super().__init__()
    # the paper (section 3.3) we have d_model=512 and d_ff=2048.
    # Therefore the inner layer is 4 times the size of the embedding layer
    self.net = nn.Sequential(
        nn.Linear(n_embd, n_embd*4),
        nn.ReLU(),
        nn.Linear(n_embd*4, n_embd),
        nn.Dropout(dropout)
      )

  def forward(self, x):
    return self.net(x)
  
class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_heads, head_size, n_embd, seq_len, dropout=0.1):
        super().__init__()
        head_size = n_embd // n_heads
        self.layernorm1 = nn.LayerNorm(n_embd)
        self.layernorm2 = nn.LayerNorm(n_embd)
        self.multi_head = Multi_head(n_heads, head_size, n_embd, seq_len, dropout)
        self.ff = FeedForward(n_embd, dropout)

    def forward(self, x):
        # x is of shape (B, T, C)
        x = x + self.multi_head(self.layernorm1(x)) # residoual connection + layer normalization (B, T, C)
        x = x + self.ff(self.layernorm2(x)) # residoual connection + layer normalization (B, T, C)
        return x # (B, T, C)

class DecoderOnly(nn.Module):
    def __init__(self, vocab_size, n_layers, n_heads, head_size, n_embd, seq_len, dropout=0.1, device='cpu'):
        super().__init__()
        self.device = device

        # vocab_size = len(stoi)
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.position_emb = positional_encoding(n_embd, seq_len)


        self.block_layers = nn.Sequential(*[TransformerDecoderBlock(n_heads, head_size, n_embd, seq_len, dropout) for _ in range(n_layers)])

        #layer normalization after decoder layers
        self.ln = nn.LayerNorm(n_embd)
        self.linear_last = nn.Linear(n_embd, vocab_size, bias=False) # linear layer to map the output to the vocabulary size

    def forward(self, x):
        B, T = x.shape
        
        #pos_embedding = self.position_emb(torch.arange(T, device=x.device))

        x = self.token_emb(x) + self.position_emb[:,:T].to(self.device) # broadcast the positional encoding to the batch size (B, T, C)
        x = self.block_layers(x) # (B, T, C)

        x = self.ln(x) # (B, T, C)
        x = self.linear_last(x) # here x is logists (B, T, C), C = vocab_size
        # x = torch.transpose(x, 1, 2) #shape (B,C,T)
        return x
    
    def generate(self, tokens, seq_len, max_len_generate=100):
        """ generate a sequence of tokens given a token """
        for _ in range(max_len_generate):
            print(tokens.shape)
            input = tokens[:, -seq_len:] # take the last token but hold the sequence length (B, T)
            logits = self(input)
            logits = logits[:, -1, :] # take last token. from shape (B, C, T) to (B, C) / (B, T, C) to (B, C)
            probs = F.softmax(logits, dim=-1) # shape (B, C)
            next_token = torch.multinomial(probs, num_samples=1) # shape (B, 1)
            #append next token ix to the solution sequence so far
            tokens = torch.cat([tokens, next_token], dim=-1) # shape (B, T+1)

        return tokens # decode the token to text

          

if __name__ == "__main__":
    # input
    B, T, C, = 4, 8, 16
    x = torch.randn(B,T,C) #shape (B,T,C)

    # decoder only
    vocab_size = 10
    n_layers = 2
    n_heads = 4
    n_embd = C
    head_size = 8
    seq_len = T # sequence len
    net = DecoderOnly(vocab_size=vocab_size, n_layers=n_layers, n_heads=n_heads, head_size=head_size, n_embd=n_embd, seq_len=seq_len)
    out = net(torch.randint(0, 10, (B, T)))

    # input = torch.zeros((1,1), dtype=torch.long) # shape (B, T) = (1, 1)
    # print(decode(net.generate(input, seq_len=seq_len, max_len_generate=200).squeeze(0)))

