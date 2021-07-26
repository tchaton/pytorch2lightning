from __future__ import print_function
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DistributedSampler
from torch.profiler import profile, record_function, ProfilerActivity, schedule


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


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    ###### SETUP PROFILER
    my_schedule = schedule(
        skip_first=2,
        wait=1,
        warmup=1,
        active=1,
        repeat=1
    )

    def trace_handler(p):
        output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
        print(output)
        p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=False,
        schedule=my_schedule,
        on_trace_ready=trace_handler,

    ) as prof:
    ###### SETUP PROFILER
        for batch_idx, (data, target) in enumerate(train_loader):
            with record_function("transfer_data"):
                data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            with record_function("forward"):
                output = model(data)
                loss = F.nll_loss(output, target)
            with record_function("backward"):
                loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                if args.dry_run:
                    break
            
            ###### STEP PROFILER
            prof.step()
            ###### STEP PROFILER


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def setup_ddp():
    """Setup ddp enviroment"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8088"
    rank = int(os.getenv("LOCAL_RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))

    print(f"REGISTERING RANK {rank}")

    if torch.distributed.is_available() and sys.platform not in ("win32", "cygwin"):
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    return rank, world_size

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 14)')
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

    ###### SETUP PROGRESS GROUP
    rank, world_size = setup_ddp()
    ###### SETUP PROGRESS GROUP

    torch.cuda.set_device(f"cuda:{rank}")

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)

    sampler1 = DistributedSampler(dataset1, num_replicas=world_size, rank=rank, shuffle=False)
    sampler2 = DistributedSampler(dataset2, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs, sampler=sampler1)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs, sampler=sampler2)

    model = Net().to(device)

    model = DistributedDataParallel(model, device_ids=[rank])

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model and rank == 0:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    # python WORLD_SIZE=2 RANK=X ddp_mnist/pytorch.py
    main()