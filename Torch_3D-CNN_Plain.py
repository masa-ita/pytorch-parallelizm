import torch

CUBE_SIZE = 128
NUM_CHANNELS = 4
NUM_CLASSES = 10
BATCH_SIZE = 1
NUM_EPOCHES = 5

class DummyDataset(torch.utils.data.Dataset):

    def __init__(self, data_dims=(4, 128, 128, 128), num_classes=10, size=1000):
        self.data_dims = data_dims
        self.num_classes = num_classes
        self.size = size
    
    def __getitem__(self, index):
        return torch.rand(*self.data_dims, dtype=torch.float32), torch.randint(0, self.num_classes, (1,))[0]
    
    def __len__(self):
        return self.size


train_ds = DummyDataset(data_dims=(NUM_CHANNELS, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE), 
                        num_classes=NUM_CLASSES, size=1000)

train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

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
    

model = ThreeDCNN(width=CUBE_SIZE, height=CUBE_SIZE, depth=CUBE_SIZE, 
                  channels=NUM_CHANNELS, num_classes=NUM_CLASSES)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

model.train()
for batch, (X, y) in enumerate(train_dataloader):

    # Compute prediction error
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
