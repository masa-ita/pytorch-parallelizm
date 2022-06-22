import os
import argparse

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from fairscale.optim.oss import OSS
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from fairscale.optim.grad_scaler import ShardedGradScaler


CUBE_SIZE = 320
NUM_CHANNELS = 4
NUM_CLASSES = 10
BATCH_SIZE = 1

WORLD_SIZE = 4
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
    args):

    # process group init
    print(f"Running ShardedDDP example on rank {rank}.")
    setup(rank, args.world_size)

    # Problem statement
    model = ThreeDCNN(width=args.cube_size, height=args.cube_size, depth=args.cube_size, 
                  channels=args.num_channels, num_classes=args.num_classes).to(rank)

    train_ds = DummyDataset(data_dims=(args.num_channels, args.cube_size, args.cube_size, args.cube_size), 
                            num_classes=args.num_classes, size=100)
    dataloader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss()

    # optimizer specific arguments e.g. LR, momentum, etc...
    base_optimizer_arguments = { "lr": 1e-4}

    # Wrap a base optimizer into OSS
    if args.optimizer == 'adam':
        base_optimizer = torch.optim.Adam
    else:
        base_optimizer = torch.optim.SGD  # any pytorch compliant optimizer
    optimizer = OSS(
        params=model.parameters(),
        optim=base_optimizer,
        **base_optimizer_arguments)


    # Wrap the model into ShardedDDP, which will reduce gradients to the proper ranks
    model = ShardedDDP(model, optimizer)

    # Any relevant training loop, nothing specific to OSS. For example:
    model.train()
    scaler = ShardedGradScaler()
    for e in range(args.epochs):
        for (data, target) in dataloader:
            data, target = data.to(rank), target.to(rank)
            # Train
            model.zero_grad()
            with autocast(enabled=args.mixed_precision):
                outputs = model(data)
                loss = loss_fn(outputs, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()           
    cleanup()
            
def run_demo(demo_fn, args):
    mp.spawn(demo_fn,
             args=args,
             nprocs=args.world_size,
             join=True)
    

def get_args():
    parser = argparse.ArgumentParser(description='3DCNN FairScale')
    parser.add_argument('--cube_size',
                        type=int,
                        default=128,
                        help='dummy data cube size')
    parser.add_argument('--num_channels',
                        type=int,
                        default=4,
                        help='dummy data channel size')
    parser.add_argument('--num_classes',
                        type=int,
                        default=10,
                        help='dummy data number of classes')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='batch size')
    parser.add_argument('--optimizer',
                        type=str,
                        default="sgd",
                        help='optimizer [sgd, adam]')
    parser.add_argument('--mixed_precision',
                        action="store_true",
                        default=False,
                        help="mixed precision mode.")
    parser.add_argument('--world_size',
                        type=int,
                        default=4,
                        help='multi processing world size')
    parser.add_argument('--epochs',
                        type=int,
                        default=1,
                        help='training epoch size')

    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    args = get_args()
    run_demo(train, args)
