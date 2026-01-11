import torch
from torch.cuda.amp import autocast, GradScaler

class Trainer:
    def __init__(self, model, optimizer, criterion, device, mixed_precision):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scaler = GradScaler() if mixed_precision else None

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            if self.scaler:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)
