#!/usr/bin/env python
# coding: utf-8

import os
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

import deepspeed
from deepspeed.pipe import PipelineModule
from deepspeed.utils import RepeatingLoader


CUBE_SIZE = 256
NUM_CHANNELS = 4
NUM_CLASSES = 10


categories = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
              'monitor', 'night_stand', 'sofa', 'table', 'toilet']


deepspeed.init_distributed()

class DummyDataset(torch.utils.data.Dataset):

    def __init__(self, dims=(4, 128, 128, 128), num_classes=10, size=1000):
        self.dims = dims
        self.num_classes = num_classes
        self.size = size
    
    def __getitem__(self, index):
        return np.random.rand(*self.dims).astype(np.float16), np.random.randint(0, self.num_classes)
    
    def __len__(self):
        return self.size


def dummy_trainset(local_rank):
    dist.barrier()
    if local_rank != 0:
        dist.barrier()
    
    trainset = DummyDataset(dims=(NUM_CHANNELS, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE), num_classes=10, size=500)

    if local_rank == 0:
        dist.barrier()
    return trainset

def get_args():
    parser = argparse.ArgumentParser(description='3DCNN')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('-s',
                        '--steps',
                        type=int,
                        default=100,
                        help='quit after this many steps')
    parser.add_argument('-p',
                        '--pipeline-parallel-size',
                        type=int,
                        default=2,
                        help='pipeline parallelism')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='distributed backend')
    parser.add_argument('--seed', type=int, default=1138, help='PRNG seed')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

def get_layers(width=128, height=128, depth=128, channels=1, num_classes=1):
    layers = torch.nn.Sequential(
        torch.nn.Conv3d(channels, 64, 3),
        torch.nn.ReLU(),
        torch.nn.MaxPool3d(2),
        torch.nn.BatchNorm3d(64),
        torch.nn.Conv3d(64, 64, 3),
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
    return layers


def train_base(args):
    torch.manual_seed(args.seed)

    net = get_layers(width=CUBE_SIZE, height=CUBE_SIZE, depth=CUBE_SIZE, 
                     channels=NUM_CHANNELS, num_classes=NUM_CLASSES)

    trainset = dummy_trainset(args.local_rank)

    engine, _, dataloader, __ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=[p for p in net.parameters() if p.requires_grad],
        training_data=trainset)

    dataloader = RepeatingLoader(dataloader)
    data_iter = iter(dataloader)

    rank = dist.get_rank()
    gas = engine.gradient_accumulation_steps()

    criterion = torch.nn.CrossEntropyLoss()

    total_steps = args.steps * engine.gradient_accumulation_steps()
    step = 0
    for micro_step in range(total_steps):
        batch = next(data_iter)
        inputs = batch[0].to(engine.device)
        labels = batch[1].to(engine.device)

        outputs = engine(inputs)
        loss = criterion(outputs, labels)
        engine.backward(loss)
        engine.step()

        if micro_step % engine.gradient_accumulation_steps() == 0:
            step += 1
            if rank == 0 and (step % 10 == 0):
                print(f'step: {step:3d} / {args.steps:3d} loss: {loss}')



def train_pipe(args, part='type:Conv3d'):
    torch.manual_seed(args.seed)
    deepspeed.runtime.utils.set_random_seed(args.seed)

    net = get_layers(width=CUBE_SIZE, height=CUBE_SIZE, depth=CUBE_SIZE, 
                     channels=NUM_CHANNELS, num_classes=NUM_CLASSES)
    net = PipelineModule(layers=net,
                         loss_fn=torch.nn.CrossEntropyLoss(),
                         num_stages=args.pipeline_parallel_size,
                         partition_method=part,
                         activation_checkpoint_interval=0)

    trainset = dummy_trainset(args.local_rank)

    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=[p for p in net.parameters() if p.requires_grad],
        training_data=trainset)

    for step in range(args.steps):
        loss = engine.train_batch()


if __name__ == '__main__':
    args = get_args()

    deepspeed.init_distributed(dist_backend=args.backend)
    args.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.local_rank)

    if args.pipeline_parallel_size == 0:
        train_base(args)
    else:
        train_pipe(args)