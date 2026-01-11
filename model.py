import os
import torch

from simclr import SimCLR
from simclr.modules import LARS


def load_optimizer(args, model):

    scheduler = None
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)  # TODO: LARS
    elif args.optimizer == "LARS":
        # optimized using LARS with linear learning rate scaling
        # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
        learning_rate = 0.3 * args.batch_size / 256
        optimizer = LARS(
            model.parameters(),
            lr=learning_rate,
            weight_decay=args.weight_decay,
            exclude_from_weight_decay=["batch_normalization", "bias"],
        )

        # "decay the learning rate with the cosine decay schedule without restarts"
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=0, last_epoch=-1
        )
    else:
        raise NotImplementedError

    return optimizer, scheduler


def save_model(args, model, optimizer, scheduler, latest=False):
    if latest:
        save_path = os.path.join(args.model_path, "latest.tar")
    else:
        save_path = os.path.join(args.model_path, f"checkpoint_{args.current_epoch}.tar")

    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    
    checkpoint = {
        'epoch': args.current_epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, save_path)
