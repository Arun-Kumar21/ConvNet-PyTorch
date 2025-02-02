import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

class MNISTDataModule:
    def __init__(self, config):
        self.config = config
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def prepare_data(self):
        datasets.MNIST(self.config.DATA_DIR, train=True, download=True)
        datasets.MNIST(self.config.DATA_DIR, train=False, download=True)

    def setup(self):
        full_train = datasets.MNIST(
            self.config.DATA_DIR,
            train=True,
            transform=self.transform
        )

        train_size = int(len(full_train) * self.config.TRAIN_VAL_SPLIT)
        val_size = len(full_train) - train_size


        self.train_dataset, self.val_dataset = random_split(full_train, [train_size, val_size])

        self.test_dataset = datasets.MNIST(
            self.config.DATA_DIR,
            train=False,
            transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False
        )


