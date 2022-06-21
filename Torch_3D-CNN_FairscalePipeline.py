#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from pickletools import optimize

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

import fairscale


CUBE_SIZE = 310
NUM_CHANNELS = 4
NUM_CLASSES = 10
BATCH_SIZE = 1


class DummyDataset(torch.utils.data.Dataset):

    def __init__(self, data_dims=(4, 128, 128, 128), num_classes=10, size=100):
        self.data_dims = data_dims
        self.num_classes = num_classes
        self.size = size
    
    def __getitem__(self, index):
        return torch.rand(*self.data_dims, dtype=torch.float32), torch.randint(0, self.num_classes, (1,))[0]
    
    def __len__(self):
        return self.size


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


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    scaler = GradScaler()
    for batch, (X, y) in enumerate(dataloader):
        device = model.devices[0]
        # Compute prediction error
        with autocast():
            pred = model(X.to(device))
            loss = loss_fn(pred.to(device), y.to(device))

        # Backpropagation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



def train_pipe(balance):
    net = get_layers(width=CUBE_SIZE, height=CUBE_SIZE, depth=CUBE_SIZE, 
                     channels=NUM_CHANNELS, num_classes=NUM_CLASSES)
    net = fairscale.nn.Pipe(net, balance=balance)

    train_dataset = DummyDataset(data_dims=(NUM_CHANNELS, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE), 
                            num_classes=NUM_CLASSES, size=100)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
    
    optimizer = optim.SGD(net.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    train(train_loader, net, loss_fn, optimizer)

if __name__ == '__main__':
    train_pipe([1,1,1,19])
