import torch

def accuracy(outputs, targets):
    preds = torch.argmax(outputs, dim=1)
    return (preds == targets).float().mean()
