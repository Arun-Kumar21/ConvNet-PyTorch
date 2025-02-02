import torch

from server.config import Config
from server.scripts.data import MNISTDataModule
from server.scripts.model import ConvNet
from server.scripts.train_model import TrainModel


def main():
    torch.manual_seed(Config.RANDOM_SEED)
    torch.cuda.manual_seed(Config.RANDOM_SEED)


    data_module = MNISTDataModule(Config)
    data_module.prepare_data()
    data_module.setup()

    model = ConvNet(Config)

    trainer = TrainModel(model, Config)

    trainer.train(
        data_module.train_dataloader(),
        data_module.val_dataloader()
    )

    test_loss, test_acc = trainer.evalute(data_module.test_dataloader())
    print(f"\n Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()