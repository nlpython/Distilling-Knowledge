import torch
import torchvision.datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def evaluate(test_dataloader, model, device):
    """
    Evaluates the given expression and returns the result.
    """
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(test_dataloader, 'Evaluating'):
            x = x.to(device)
            y = y.to(device)
            output = model(x)

            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return correct / total

def load_data():
    # load MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root='./dataset',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./dataset',
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False
    )

    return train_loader, test_loader
