import os
import numpy as np
import torch
import argparse
import json
from pathlib import Path

# distributed training
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# SimCLR
from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet

from model import load_optimizer, save_model
from utils import yaml_config_hook

from datasets.t1_datasets import SimCLRT1Dataset
from monai.transforms import (
    Compose,
    RandSpatialCropd,
    Resized,
    RandFlipd,
    RandRotated,
    RandShiftIntensityd,
    RandAdjustContrastd,
)

def train(args, train_loader, model, criterion, optimizer):
    loss_epoch = 0
    for step, (x_i, x_j) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.to(args.device, non_blocking=True)
        x_j = x_j.to(args.device, non_blocking=True)

        h_i, h_j, z_i, z_j = model(x_i, x_j)

        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))

        if args.rank == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        if args.rank == 0:
            args.global_step += 1

        loss_epoch += loss.item()
    return loss_epoch


def replace_relu_inplace(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.ReLU) and child.inplace:
            setattr(module, name, torch.nn.ReLU(inplace=False))
        else:
            replace_relu_inplace(child)


def load_checkpoint(args, model, optimizer=None, scheduler=None, path=None):
    path = path or os.path.join(args.model_path, "latest.tar")
    ckpt = torch.load(path, map_location=args.device)

    # Remove 'module.' prefix if present (for DP/DDP models)
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)

    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    args.current_epoch = ckpt.get("epoch", 0)
    args.start_epoch = ckpt.get("epoch", 0)
    if args.rank == 0:
        print(f"Resumed from checkpoint at epoch {args.current_epoch}")


def main(local_rank: int, args):

    # Get rank and world size from torchrun env vars
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Set device for this process
    torch.cuda.set_device(local_rank)
    args.device = torch.device(f"cuda:{local_rank}")

    args.world_size = world_size
    args.rank = rank
    args.local_rank = local_rank

    dist.init_process_group(backend="nccl", init_method="env://")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    simclr_transform = Compose([
        RandSpatialCropd(keys=["MRI", "MASK"], roi_size=(30, 40, 40), random_center=True, random_size=True),
        Resized(keys=["MRI", "MASK"], spatial_size=(150, 192, 192), mode=["trilinear", "nearest"]),
        RandFlipd(keys=["MRI", "MASK"], prob=0.5, spatial_axis=[2]),
        RandRotated(keys=["MRI", "MASK"], range_x=0.785, prob=0.5, mode=["trilinear", "nearest"]),
        RandShiftIntensityd(keys=["MRI"], offsets=0.5, prob=0.8),
        RandAdjustContrastd(keys=["MRI"], gamma=(0.5, 1.5), prob=0.8),
        ])

    train_ds = SimCLRT1Dataset(args.data_file, transform=simclr_transform)

    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=train_ds,
            num_replicas=world_size,
            rank=rank, shuffle=True)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        sampler=train_sampler,
        batch_size=args.batch_size,
        drop_last=True,
        pin_memory=False,
        num_workers=args.num_workers)

    # initialize ResNet
    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # initialize model
    model = SimCLR(encoder, args.projection_dim, n_features)
    replace_relu_inplace(model)
    model = model.to(args.device)

    # optimizer / loss
    optimizer, scheduler = load_optimizer(args, model)
    criterion = NT_Xent(args.batch_size, args.temperature, args.world_size)

    # Load checkpoint if requested
    if args.reload:
        ckpt_path = os.path.join(args.model_path, "latest.tar")
        if os.path.exists(ckpt_path):
            load_checkpoint(args, model, optimizer, scheduler)
        else:
            if args.rank == 0:
                print("Reload requested, but latest.tar not found — starting fresh.")


    # DDP / DP
    if args.world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
        )


    args.global_step = 0
    args.current_epoch = args.start_epoch
    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(args, train_loader, model, criterion, optimizer)

        if args.rank == 0 and scheduler:
            scheduler.step()

        if args.rank == 0 and epoch % 10 == 0:
            save_model(args, model, optimizer, scheduler)

        if args.rank == 0:
            print(
                f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}"
            )
            args.current_epoch += 1

        # Save checkpoint every epoch by overwriting latest.pth
        if args.rank == 0:
            save_model(args, model, optimizer, scheduler, latest=True)


    ## end training
    save_model(args, model, optimizer)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "LOCAL_RANK not found in environment. "
            "This script should be launched with torchrun, e.g.\n"
            "  torchrun --nproc_per_node=4 --nnodes=2 ... main.py"
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    main(local_rank, args)