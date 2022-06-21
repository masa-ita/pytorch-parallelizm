import argparse
import numpy as np
import logging
import time
import math

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam

from fairscale.nn import auto_wrap, default_auto_wrap_policy, enable_wrap
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.data_parallel import OffloadConfig

RPC_PORT = 29501


def init_random_seed(seed: int):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


class DummyDataset(torch.utils.data.Dataset):

    def __init__(self, data_dims=(4, 128, 128, 128), num_classes=10, size=100):
        self.data_dims = data_dims
        self.num_classes = num_classes
        self.size = size
    
    def __getitem__(self, index):
        return torch.rand(*self.data_dims, dtype=torch.float32), torch.randint(0, self.num_classes, (1,))[0]
    
    def __len__(self):
        return self.size
    
    
def get_dataloaders(args, device, fsdp_config, model_specs):
    """Returns dataloaders for real data."""

    train_ds = DummyDataset(data_dims=(NUM_CHANNELS, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE), 
                            num_classes=NUM_CLASSES, size=100)
    valid_ds = DummyDataset(data_dims=(NUM_CHANNELS, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE), 
                            num_classes=NUM_CLASSES, size=20)
    test_ds  = DummyDataset(data_dims=(NUM_CHANNELS, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE), 
                            num_classes=NUM_CLASSES, size=20)
    
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader  = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

    return train_dataloader, valid_dataloader, test_dataloader

class ThreeDCNN(torch.nn.Module):
    
    def __init__(self, width=128, height=128, depth=128, channels=1, num_classes=1, **kwargs):
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
        )
        self.layers = checkpoint_wrapper(self.layers, **kwargs)
        self.last_layer =  torch.nn.Linear(128, num_classes)


    def forward(self, x):
        x = self.layers(x)
        logits = self.last_layer(x)
        return logits
    

def get_model(args, device, config):

    if args.ssd_offload:
        return ThreeDCNN(width=CUBE_SIZE, height=CUBE_SIZE, depth=CUBE_SIZE, 
                         channels=NUM_CHANNELS, num_classes=NUM_CLASSES)
    else:
        return ThreeDCNN(width=CUBE_SIZE, height=CUBE_SIZE, depth=CUBE_SIZE, 
                         channels=NUM_CHANNELS, num_classes=NUM_CLASSES).to(device)


def get_model_and_optimizer(args, device, fsdp_config, model_config):
    """Return instantiated model and optimizer function."""

    model = get_model(args, device, model_config)

    lr = benchmark_config["lr"]

    def make_adam(params):
        return Adam(params, lr=lr)

    optimizer = make_adam
    return model, optimizer


def create_model_config(args, fsdp_config=None, model_specs=None):
    """Return a dict with the given model, dataset and optimizer."""

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dataloader_fn = get_dataloaders

    data = dataloader_fn(args, device, model_specs)
    model, optimizer = get_model_and_optimizer(args, device, model_specs)
    return {
        "model": model,
        "optimizer": optimizer,
        "data": data,
    }
    
    
def create_fsdp_config(model_name):
    """Return a dict with configurations required for `model_name` model."""

    return dict()
    
def get_model_specs():
    return dict()

    
def train(model_config, model, fsdp_config, model_specs, args):
    dataloader, _, _ = model_config["data"]
    criterion = fsdp_config["criterion"]
    optimizer = model_config["optimizer"]

    if not args.benchmark_eval:
        model.train()
    log_number_of_parameters(model)

    total_loss = 0.0

    optimizer = optimizer(model.parameters())

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    start_time = time.time()
    epoch_start_time = 0.0

    def get_batch(source):
        seq_len = len(source) - 1
        data = source[0:seq_len]
        target = source[1 : 1 + seq_len]
        return data, target

    for i, batch in enumerate(dataloader):
        if i == 1:
            epoch_start_time = time.time()

        source, target = get_batch(batch)
        if args.full_fp16:
            # source = source.half()
            target = target.half()
        if args.max_batch and i > args.max_batch:
            break

        if args.ssd_offload:
            input = source.cuda()
            target = target.cuda()
            output = model(input)
            print(f"output.dtype {output.dtype}, target.dtype {target.dtype}")
            loss = torch.nn.CrossEntropyLoss()(output.view(-1), target.view(-1))
        else:
            optimizer.zero_grad()
            input = source.cuda()
            target = target.cuda()
            output = model(input)

            loss = criterion(output.view(-1), target.view(-1))
            loss.backward()

            torch.nn.utils.clip_grad_value_(model.parameters(), model_specs["clip_value"])
            optimizer.step()

        total_loss += loss.item()

        log_interval = 1
        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            if dist.get_rank() == 0:
                print(
                    "| batch {:5d} | loss {:5.2f} | ppl {:8.2f}".format(
                        i, cur_loss, math.exp(cur_loss)
                    )
                )
            total_loss = 0
            start_time = time.time()

    if epoch_start_time != 0:
        torch.cuda.synchronize()
    else:
        raise RuntimeError(
            "Unable to benchmark on a single batch. Increase the size " " of the dataset and rerun the benchmark."
        )
    return loss.item()

    
def benchmark_model(model_config, model, fsdp_config, model_specs, args):

    epoch = fsdp_config["epochs"]
    start_time = time.time()
    if dist.get_rank() == 0:
        print("-" * 110)
        print("| start of epoch {:1d}".format(epoch))
        print("-" * 110)
    loss = train(model_config, model, fsdp_config, model_specs, args)
    elapsed_time = time.time() - start_time
    if dist.get_rank() == 0:
        print("-" * 110)
        print("| end of epoch {:1d} | time: {:5.2f}s | train loss {:5.2f} ".format(epoch, elapsed_time, loss))
        print("-" * 110)
    print(
        "Peak allocated bytes on cuda:{}: {:4f}GB".format(
            dist.get_rank(), torch.cuda.memory_stats(dist.get_rank())["allocated_bytes.all.peak"] / 2**30
        )
    )
        
    
def dummy_fsdp(rank, args, world_size):
    """Benchmark a given model using a single process and multiple devices."""

    init_method_pgroup = "tcp://localhost:{}".format(RPC_PORT)
    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, init_method=init_method_pgroup
    )

    torch.cuda.set_device(rank)
    init_random_seed(0)
    logging.basicConfig(level=logging.DEBUG)

    fsdp_config = create_fsdp_config()
    model_specs = get_model_specs()
    model_config = create_model_config(args, fsdp_config=fsdp_config, model_specs=model_specs)
    model = model_config["model"]
    config = {}
    if args.ssd_offload:
        config["offload_config"] = OffloadConfig(offload_type="ssd_offload")

    if args.full_fp16:
        config["compute_dtype"] = torch.float16
        config["mixed_precision"] = False

    if args.enable_auto_wrap:
        with enable_wrap(wrapper_cls=FSDP, **config):
            fsdp_model = auto_wrap(model, auto_wrap_policy=default_auto_wrap_policy)
            fsdp_model = FSDP(fsdp_model, **config)
    else:
        fsdp_model = FSDP(model, **config)

    if args.full_fp16:
        fsdp_model = fsdp_model.half()
    print(f"param dtype {[p.dtype for p in fsdp_model.parameters()]}")
    if args.dry_run:
        train(model_config, fsdp_model, fsdo_config, model_specs, args)
    else:
        benchmark_model(model_config, fsdp_model, fsdp_config, model_specs, args)


parser = argparse.ArgumentParser(description="FairScale")
parser.add_argument("--max_batch", type=int, default=4, help="Max number of batches")
parser.add_argument("--dry_run", action="store_true", help="Run a sample training run without regression testing.")
parser.add_argument("--debug", action="store_true", default=False, help="Display additional debug information")
parser.add_argument("--enable_auto_wrap", action="store_true", default=False, help="Use auto_wrap with FSDP")
parser.add_argument("--ssd_offload", action="store_true", default=False, help="Benchmark ssd_offload workflow.")
parser.add_argument("--full_fp16", action="store_true", default=False, help="Benchmark in full fp16 mode.")

if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)

    print(f"Running FSDP with args: {args}")
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    assert num_devices > 0

    mp.spawn(
        dummy_fsdp,
        args=(args, num_devices),
        nprocs=num_devices,
        join=True,
    )
