import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torchvision


transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(1, 784).t())
    ])

def split_train_val(trainset, val_split=0.1):
    n_val = int(len(trainset) * val_split)
    n_train = len(trainset) - n_val
    trainset, valset = torch.utils.data.random_split(trainset, [n_train, n_val])
    return trainset, valset


transform_train = transform_test = transform

trainset = torchvision.datasets.MNIST(
    root='./data/sequential_mnist', train=True, download=True, transform=transform_train)
valset = torchvision.datasets.MNIST(
    root='./data/sequential_mnist', train=True, download=True, transform=transform_test)
testset = torchvision.datasets.MNIST(
    root='./data/sequential_mnist', train=False, download=True, transform=transform_test)


trainset, _ = split_train_val(trainset, val_split=0.1)
_, valset = split_train_val(valset, val_split=0.1)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    imshow(torchvision.utils.make_grid(images))
    print(' '.join(f'{labels[j]}' for j in range(4)))
    print(images.shape, len(trainset), len(valset), len(testset))

