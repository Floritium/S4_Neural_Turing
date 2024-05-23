from dataset_mnist import get_data
from argparser import get_args
import torch
from tqdm import tqdm

trainset, valset, testset = get_data() # get sequential mnist dataset


# # # Training
# def train(args, trainloader, model, criterion, optimizer, device):
#     model.train()
#     train_loss = 0
#     correct = 0
#     total = 0
#     pbar = tqdm(enumerate(trainloader))
#     # for batch_idx, (inputs, targets) in pbar:
#     #     inputs, targets = inputs.to(device), targets.to(device)
#     #     optimizer.zero_grad()
#     #     outputs = model(inputs)
#     #     loss = criterion(outputs, targets)
#     #     loss.backward()
#     #     optimizer.step()

#     #     train_loss += loss.item()
#     #     _, predicted = outputs.max(1)
#     #     total += targets.size(0)
#     #     correct += predicted.eq(targets).sum().item()

#     #     pbar.set_description(
#     #         'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
#     #         (batch_idx, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total, correct, total)


def train(trainloader, device):
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(trainloader))
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        print(inputs.shape, targets.shape)


if __name__ == '__main__':
    
    # get args
    args = get_args()

    device = torch.device("mps" if torch.backends.mps .is_available() and not args.no_cuda else "cpu")

    # Dataloaders
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False)
    
    train(trainloader, device)
