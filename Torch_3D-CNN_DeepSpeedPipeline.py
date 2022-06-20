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


deepspeed.init_distributed()

class DummyDataset(torch.utils.data.Dataset):

    def __init__(self, data_dims=(4, 128, 128, 128), num_classes=10, size=1000):
        self.data_dims = data_dims
        self.num_classes = num_classes
        self.size = size
    
    def __getitem__(self, index):
        return torch.rand(*self.data_dims, dtype=torch.float16), torch.randint(0, self.num_classes, (1,))[0]
    
    def __len__(self):
        return self.size


def dummy_trainset(local_rank, cube_size, num_channels, num_classes):
    dist.barrier()
    if local_rank != 0:
        dist.barrier()
    
    trainset = DummyDataset(data_dims=(num_channels, cube_size, cube_size, cube_size), 
                            num_classes=num_classes, size=500)

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
                        default=50,
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
    parser.add_argument('--pipeline_part',
                        type=str,
                        default='parameters',
                        help='pipeline partitioning')
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
    parser.add_argument('--seed', type=int, default=1138, help='PRNG seed')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

def get_layers(width=128, height=128, depth=128, channels=1, num_classes=1):
    layers = torch.nn.Sequential(
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
    return layers


def train_base(args):
    torch.manual_seed(args.seed)

    net = get_layers(width=args.cube_size, height=args.cube_size, depth=args.cube_size, 
                     channels=args.num_channels, num_classes=args.num_classes)

    trainset = dummy_trainset(args.local_rank, args.cube_size, args.num_channels, args.num_classes)

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



def train_pipe(args):
    torch.manual_seed(args.seed)
    deepspeed.runtime.utils.set_random_seed(args.seed)

    net = get_layers(width=args.cube_size, height=args.cube_size, depth=args.cube_size, 
                     channels=args.num_channels, num_classes=args.num_classes)
    net = PipelineModule(layers=net,
                         loss_fn=torch.nn.CrossEntropyLoss(),
                         num_stages=args.pipeline_parallel_size,
                         partition_method=args.pipeline_part,
                         activation_checkpoint_interval=0)

    trainset = dummy_trainset(args.local_rank, args.cube_size, args.num_channels, args.num_classes)

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
