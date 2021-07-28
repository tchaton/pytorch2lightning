from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning import Trainer, LightningModule, seed_everything, LightningDataModule
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

    def __init__(self, model, lr: float, gamma: float):
        super().__init__()
        self.save_hyperparameters()
        self.model = model

    def shared_step(self, batch, stage):
        data, target = batch
        output = self.model(data)
        loss = F.nll_loss(output, target, reduction='sum')
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

    def __init__(self, *args, **kwargs):
        super().__init__()
        save_hyperparameters(self)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def train_dataloader(self):
        train_ds = datasets.MNIST('data', train=True, download=False, transform=self.transforms)
        return DataLoader(train_ds, batch_size=self.hparams.train_batch_size, shuffle=self.hparams.shuffle, pin_memory=self.hparams.pin_memory, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        test_ds = datasets.MNIST('data', train=False, download=False, transform=self.transforms)
        return DataLoader(test_ds, batch_size=self.hparams.test_batch_size, shuffle=False, pin_memory=self.hparams.pin_memory, num_workers=self.hparams.num_workers)


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    seed_everything(args.seed)

    dm = MnistDataModule(train_batch_size=args.batch_size, test_batch_size=args.test_batch_size, num_workers=1, pin_memory=use_cuda, shuffle=True)

    net =  Net()
    model = LiftModel(net, lr=args.lr, gamma=args.gamma)

    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=dm)
    trainer.test(datamodule=dm)

    if args.save_model:
        trainer.save_checkpoint("mnist_cnn.pt")


if __name__ == '__main__':
    main()