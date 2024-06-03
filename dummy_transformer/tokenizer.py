import nltk 
import torch

# read the text file
with open('data/ep0.txt', 'r') as f:
    text = f.read()

# create a set of all unique characters in the text
chars = sorted(list(set(text)))
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# define encode and decode functions that convert strings to arrays of tokens and vice-versa
encode = lambda x: torch.tensor([stoi[ch] for ch in x], dtype=torch.long) #encode text to integers
decode = lambda x: ''.join([itos[i.item()] for i in x]) #decode integers to text
vocab_size = len(stoi)


def get_batch(source, device, sequence_len, batch_size):
  """ get batch of size sequence_len from source """
  
  # generate `batch_size` random offsets on the data 
  ix = torch.randint(len(source)-sequence_len, (batch_size,) )
  # collect `batch_size` subsequences of length `sequence_len` from source, as data and target
  x = torch.stack([source[i:i+sequence_len] for i in ix])
  # target is just x shifted right (ie the predicted token is the next in the sequence)
  y = torch.stack([source[i+1:i+1+sequence_len] for i in ix])
  return x.to(device), y.to(device)

if __name__ == "__main__":
    data = encode(text[:100]) # next 10 sequence of characters
    print(data)
    
    print()
    
    decoded_data = decode(data) # decode the data
    print(decoded_data)

    x, y = get_batch(data, torch.device('cpu'), 10, 32) # get batch of size 10 from data

    print(x, "\n", y) # print the batch