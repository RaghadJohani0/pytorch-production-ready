import torch
from src.utils.metrics import accuracy

class Evaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def evaluate(self, loader):
        self.model.eval()
        acc = 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                acc += accuracy(outputs, labels).item()

        return acc / len(loader)
