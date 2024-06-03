import torch
from dummy.model_decoder_only import DecoderOnly
from dummy.tokenizer import encode, decode, get_batch, text, itos, stoi
import os
from tqdm import tqdm

def training(model, criterion, optimizer, batch_size, data, device, seq_len=256, n_epochs=100):
    """ training loop for the model """
    train_data, test_data = data

    model.train()

    for _ in range(n_epochs):
        for steps in tqdm(range(0, len(train_data))):
            optimizer.zero_grad()
            x_batch, y_batch = get_batch(train_data, device, sequence_len=seq_len, batch_size=batch_size)

            logits = model(x_batch)
            print(x_batch.shape , y_batch.shape, logits.shape)
            loss = criterion(logits.view(-1, 75), y_batch.view(-1))
            loss.backward()
            optimizer.step()

            print(f"Steps: {steps}, Loss: {loss.item()}")
            if steps % 10 == 0:
                input = torch.zeros((1,1), dtype=torch.long, device=device) # shape (B, T) = (1, 1)
                print(decode(model.generate(input, seq_len=seq_len, max_len_generate=100).squeeze(0)))

                if not os.path.exists('model'):
                    os.makedirs('model')
                torch.save(model.state_dict(), 'model/model.pth')

    input = torch.zeros((1,1), dtype=torch.long, device=device) # shape (B, T) = (1, 1)
    print(decode(model.generate(input, seq_len=seq_len, max_len_generate=150).squeeze(0)))
    
    return model



def main():

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # data
    data_set = encode(text)
    # data_set = data_set[:1000]
    n = int(0.95*len(data_set))
    train_data = data_set[:n]
    test_data = data_set[n:]
    print(len(train_data))
    print(decode(train_data[:100]))
    
    #hyper paarameters
    vocab_size = len(stoi) # vocabulary size for output probabilities
    batch_size = 32
    seq_len = 256
    n_embd = 300
    n_layers = 3
    n_heads = 4
    head_size = n_embd//n_heads
    lr = 1e-3
    
    # model
    print(f"Backend: {device}")
    model = DecoderOnly(vocab_size, n_layers, n_heads, head_size, n_embd, seq_len, device=device)
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # # training
    training(model, criterion, optimizer, batch_size=batch_size, data=(train_data, test_data), device = device, seq_len=seq_len)

if __name__ == "__main__":
    main()