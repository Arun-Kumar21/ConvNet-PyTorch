import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class TrainModel:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr = config.LEARNING_RATE
        )

        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

    def train_epochs(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total = labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def evalute(self, val_loader):
        self.model.eval()
        running_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Evaluating"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc


    def train(self, train_loader, val_loader):
        print(f"Training on Device: {self.device}")

        for epoch in range(self.config.NUM_EPOCHS):
            train_loss, train_acc = self.train_epochs(train_loader)
            val_loss, val_acc = self.evalute(val_loader)

            self.train_loss.append(train_loss)
            self.train_acc.append(train_acc)
            self.val_loss.append(val_acc)
            self.val_acc.append(val_acc)

            print(f"\nEpoch [{epoch + 1}/{self.config.NUM_EPOCHS}]")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print("-" * 50)

