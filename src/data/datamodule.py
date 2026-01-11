from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class DataModule:
    def __init__(self, data_dir, image_size, batch_size):
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def setup(self):
        self.train_ds = datasets.ImageFolder(
            f"{self.data_dir}/train", transform=self.transform
        )
        self.val_ds = datasets.ImageFolder(
            f"{self.data_dir}/val", transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)
