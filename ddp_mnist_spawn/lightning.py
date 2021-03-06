import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.utilities.cli import LightningCLI
LightningCLI.fit = lambda x: None  # temporary hack
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning import LightningModule, LightningDataModule
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.parsing import save_hyperparameters


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class LiftModel(LightningModule):
    """
    Args:
        lr: learning rate
        gamma: Learning rate step gamma
    """

    def __init__(self, model: nn.Module = None, lr: float = 1.0, gamma: float = 0.7):
        super().__init__()
        self.save_hyperparameters()
        self.model = model or Net()

    def shared_step(self, batch, stage):
        data, target = batch
        output = self.model(data)
        loss = F.nll_loss(output, target, reduction='sum')
        self.log(f"{stage}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def test_step(self, batch, batch_idx):
        self.shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = optim.Adadelta(self.parameters(), lr=self.hparams.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]


class MnistDataModule(LightningDataModule):
    """
    Args:
        train_batch_size: input batch size for training
        test_batch_size: input batch size for testing
        num_workers: num workers to be used with the DataLoader
        pin_memory: Whether to pin the tensors when running on cuda
        shuffle: Whether to shuffle the training data
    """

    def __init__(self, train_batch_size: int = 64, test_batch_size: int = 1000, num_workers: int = 1, pin_memory: bool = True, shuffle: bool = True):
        super().__init__()
        save_hyperparameters(self)
        self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        datasets.MNIST('data', train=True, download=True)

    def train_dataloader(self):
        train_ds = datasets.MNIST('data', train=True, download=False, transform=self.transforms)
        return DataLoader(train_ds, batch_size=self.hparams.train_batch_size, shuffle=self.hparams.shuffle, pin_memory=self.hparams.pin_memory, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        test_ds = datasets.MNIST('data', train=False, download=False, transform=self.transforms)
        return DataLoader(test_ds, batch_size=self.hparams.test_batch_size, shuffle=False, pin_memory=self.hparams.pin_memory, num_workers=self.hparams.num_workers)


def main():
    cli = LightningCLI(LiftModel, MnistDataModule, trainer_defaults=dict(max_epochs=14, gpus=torch.cuda.device_count(), accelerator="ddp_spawn"), save_config_overwrite=True, save_config_callback=None)
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(cli.model, datamodule=cli.datamodule)
    if cli.trainer.is_global_zero:
        cli.trainer.save_checkpoint("mnist_cnn.pt")


if __name__ == '__main__':
    main()
