import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.cuda.amp import autocast

from fairscale.optim.oss import OSS
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.optim.grad_scaler import ShardedGradScaler

CUBE_SIZE = 256
NUM_CHANNELS = 4
NUM_CLASSES = 10
BATCH_SIZE = 1

WORLD_SIZE = 1
EPOCHS = 2


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class DummyDataset(torch.utils.data.Dataset):

    def __init__(self, data_dims=(4, 128, 128, 128), num_classes=10, size=100):
        self.data_dims = data_dims
        self.num_classes = num_classes
        self.size = size
    
    def __getitem__(self, index):
        return torch.rand(*self.data_dims, dtype=torch.float32), torch.randint(0, self.num_classes, (1,))[0]
    
    def __len__(self):
        return self.size


class ThreeDCNN(torch.nn.Module):
    def __init__(self, width=128, height=128, depth=128, channels=1, num_classes=1):
        super(ThreeDCNN, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv3d(channels, 32, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(2),
            torch.nn.BatchNorm3d(32),
            torch.nn.Conv3d(32, 64, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(2),
            torch.nn.BatchNorm3d(64),
            torch.nn.Conv3d(64, 128, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(2),
            torch.nn.BatchNorm3d(128),
            torch.nn.Conv3d(128, 256, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(2),
            torch.nn.BatchNorm3d(256),
            torch.nn.AdaptiveAvgPool3d((1,1,1)),
            torch.nn.Flatten(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits
    
def train(
    rank: int,
    world_size: int,
    epochs: int):

    # process group init
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # Problem statement
    model = ThreeDCNN(width=CUBE_SIZE, height=CUBE_SIZE, depth=CUBE_SIZE, 
                  channels=NUM_CHANNELS, num_classes=NUM_CLASSES).to(rank)
    # Wrap the model into FSDP, which will reduce gradients to the proper ranks
    model = FSDP(model)

    train_ds = DummyDataset(data_dims=(NUM_CHANNELS, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE), 
                            num_classes=NUM_CLASSES, size=1000)
    dataloader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss()

    # optimizer specific arguments e.g. LR, momentum, etc...
    base_optimizer_arguments = { "lr": 1e-4}

    # Wrap a base optimizer into OSS
    base_optimizer = torch.optim.SGD  # any pytorch compliant optimizer
    optimizer = OSS(
        params=model.parameters(),
        optim=base_optimizer,
        **base_optimizer_arguments)

    # Any relevant training loop, nothing specific to OSS. For example:
    model.train()
    scaler = ShardedGradScaler()
    for e in range(epochs):
        for (data, target) in dataloader:
            data, target = data.to(rank), target.to(rank)
            # Train
            model.zero_grad()
            with autocast():
                outputs = model(data)
                loss = loss_fn(outputs, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
    cleanup()
            
def run_demo(demo_fn, world_size, epochs):
    mp.spawn(demo_fn,
             args=(world_size, epochs),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
    run_demo(train, WORLD_SIZE, EPOCHS)