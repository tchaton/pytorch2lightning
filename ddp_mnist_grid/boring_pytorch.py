import os
import sys
import torch
from typing import Optional
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DistributedSampler


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


# Credit to the PyTorch Lightning Team
# https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/plugins/environments/torchelastic_environment.py
class TorchElasticEnvironment:
    """Environment for fault-tolerant and elastic training with `torchelastic <https://pytorch.org/elastic/>`_"""

    def __init__(self):
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "8088"

    @staticmethod
    def is_using_torchelastic() -> bool:
        """Returns ``True`` if the current process was launched using the torchelastic command."""
        required_env_vars = ("RANK", "GROUP_RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
        return all(v in os.environ for v in required_env_vars)

    def world_size(self) -> Optional[int]:
        world_size = os.environ.get("WORLD_SIZE")
        return int(world_size) if world_size is not None else world_size

    def global_rank(self) -> int:
        return int(os.environ["RANK"])


def init_ddp(global_rank, world_size):
    print(f"REGISTERING RANK {global_rank}")
    if torch.distributed.is_available() and sys.platform not in ("win32", "cygwin"):
        torch.distributed.init_process_group("nccl", rank=global_rank, world_size=world_size)

def create_dataloader(rank, world_size):
    dataset = RandomDataset(32, 64)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    return DataLoader(dataset, sampler=sampler, batch_size=2)

def create_model_and_optimizer(rank, world_size):
    device = torch.device(f"cuda:{rank}")
    model = torch.nn.Linear(32, 2).to(device)
    model = DistributedDataParallel(model, device_ids=[device])
    return model, torch.optim.SGD(model.parameters(), lr=0.1)

def train(model, optimizer, dataloader, device):
    for batch in dataloader:
        optimizer.zero_grad()
        batch = batch.to(device)
        loss = model(batch).sum()
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss}")

if __name__ == "__main__":

    # python -m torch.distributed.launch --nproc_per_node=2 ddp_mnist_grid/boring_pytorch.py

    # 1. capture env variables
    cluster = TorchElasticEnvironment()
    if not cluster.is_using_torchelastic():
        raise Exception("This script should be run on a Torch Elastic Cluster.")
    global_rank = cluster.global_rank()
    world_size = cluster.world_size()

    # 2. set cuda device
    device = torch.device(f"cuda:{global_rank}")
    torch.cuda.set_device(device)
    
    # 3. init ddp
    init_ddp(global_rank, world_size)

    # 4. create dataloader
    dataloader = create_dataloader(global_rank, world_size)
    
    # 5. create model / otpimizer and configure ddp
    model, optimizer = create_model_and_optimizer(global_rank, world_size)

    # 6. train
    train(model, optimizer, dataloader, device)

    # 7. clean distributed training
    torch.distributed.destroy_process_group()