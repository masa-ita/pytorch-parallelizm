#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from glob import glob
import re
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam
import pytorch_lightning as pl
from torchmetrics import Accuracy

from deepspeed.ops.adam import FusedAdam



# In[ ]:


CUBE_SIZE = 512 
NUM_CHANNELS = 4
NUM_CLASSES = 10
BATCH_SIZE = 1
NUM_EPOCHES = 2

categories = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
              'monitor', 'night_stand', 'sofa', 'table', 'toilet']


# In[ ]:


class DummyDataset(Dataset):

    def __init__(self, dims=(4, 128, 128, 128), num_classes=10, size=1000):
        self.dims = dims
        self.num_classes = num_classes
        self.size = size
    
    def __getitem__(self, index):
        return np.random.rand(*self.dims).astype(np.float32), np.random.randint(0, self.num_classes)
    
    def __len__(self):
        return self.size


# In[ ]:


class DummyDataModule(pl.LightningDataModule):

    def __init__(self, dims=(4, 128, 128, 128), num_classes=10, batch_size=16):
        super().__init__()
        self.dims = dims
        self.num_classes = num_classes
        self.batch_size = batch_size

    def prepare_data(self):
        pass
        
    def setup(self, stage):
        self.train_ds = DummyDataset(self.dims, self.num_classes, size=1000)
        self.test_ds = DummyDataset(self.dims, self.num_classes, size=100)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
    
    def val_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)


# In[ ]:


class ThreeDCNN(pl.LightningModule):
    def __init__(self, width=128, height=128, depth=128, channels=1, num_classes=1):
        super(ThreeDCNN, self).__init__()
        self.layers = torch.nn.Sequential(
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
        self.metric = Accuracy()

    def forward(self, x):
        logits = self.layers(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return {"loss": loss, "preds": logits, "target": y}
    
    def training_step_end(self, outputs):
        self.metric(outputs["preds"], outputs["target"])
        self.log_dict({"loss": outputs["loss"], "metric": self.metric})

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return {"loss": loss, "preds": logits, "target": y}

    def validation_step_end(self, outputs):
        self.metric(outputs["preds"], outputs["target"])
        self.log_dict({"var_loss": outputs["loss"], "var_metric": self.metric})
        
    def configure_optimizers(self):
        return Adam(self.parameters())


# In[ ]:


dm = DummyDataModule(dims=(NUM_CHANNELS, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE), num_classes=NUM_CLASSES, batch_size=BATCH_SIZE)
model = ThreeDCNN(width=CUBE_SIZE, height=CUBE_SIZE, depth=CUBE_SIZE, channels=NUM_CHANNELS, num_classes=NUM_CLASSES)
trainer = pl.Trainer(devices=[0,1,2,3], accelerator="gpu", precision=16, max_epochs=NUM_EPOCHES, strategy="deepspeed_stage_3_offload")
trainer.fit(model, dm)


# In[ ]:




