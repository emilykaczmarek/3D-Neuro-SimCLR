import os
import csv
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from simclr.modules import get_resnet
from simclr.modules.identity import Identity
from utils import yaml_config_hook
from datasets.loaders import make_supervised_loaders

from distutils.util import strtobool

def log_metrics_to_csv(csv_path, epoch, train_loss, val_loss, test_loss, val_metrics, test_metrics, task_type):
    metric_names = ["acc", "auc"] if task_type == "classification" else ["mae", "mse"]
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", "train_loss", "val_loss", "test_loss",
            *[f"val_{k}" for k in metric_names],
            *[f"test_{k}" for k in metric_names],
        ])
        if not file_exists:
            writer.writeheader()
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
        }
        for m in metric_names:
            row[f"val_{m}"] = val_metrics.get(m, float("nan"))
            row[f"test_{m}"] = test_metrics.get(m, float("nan"))
        writer.writerow(row)


def save_ckpt(path, model, optimizer, epoch, best_metric=None):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
        },
        path,
    )

def init_model_and_optimizer(
    model,
    device,
    lr, 
    finetune,
    resume_ckpt: str | None = None,
    pretrained_ckpt: str | None = None,
):
    """
    Resumes fine-tuning/linear probing if resume_ckpt exists; otherwise, loads pre-trained encoder
    """
    start_epoch = 0

    if resume_ckpt and os.path.exists(resume_ckpt):
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        best_metric = ckpt.get("best_metric", None)
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed fine-tuning/linear probing from checkpoint: {resume_ckpt} (epoch {start_epoch})")

    else:
        ckpt = torch.load(pretrained_ckpt, map_location=device)
        # Remove DDP prefix if present
        state_dict = {k.replace("module.", ""): v for k, v in ckpt['model_state_dict'].items()}
        model.load_state_dict(state_dict, strict=False)
        start_epoch = 0
        best_metric = None
        print(f"Loaded encoder pretrained weights from: {pretrained_ckpt}")

    if not finetune:
        for p in model.encoder.parameters():
            p.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    if resume_ckpt and os.path.exists(resume_ckpt):
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    return model, optimizer, start_epoch, best_metric

class LinearHeadModel(nn.Module):
    def __init__(self, encoder, n_features, num_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(n_features, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        return self.head(x)


def train(model, loader, criterion, optimizer, device, args):
    model.train()
    total_loss = 0
    for step, batch in enumerate(loader):
        x, y = batch['MRI'].to(device), batch['task_label'].to(device)
        y = y.float().unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, args):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            x, y = batch['MRI'].to(device), batch['task_label'].to(device)
            y = y.float().unsqueeze(1)

            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item()

            all_preds.append(output.detach().cpu())
            all_labels.append(y.detach().cpu())

    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)

    metrics = {}

    if args.task_type == "classification":
        probs = torch.sigmoid(preds).squeeze(1).numpy()
        y_true = labels.squeeze(1).numpy()
        y_hat = (probs > 0.5).astype(np.int64)
        metrics["acc"] = float((y_hat == y_true).mean())
        try:
            metrics["auc"] = float(roc_auc_score(y_true, probs))
        except Exception:
            metrics["auc"] = float("nan")
        
    elif args.task_type == "regression":
        err = preds - labels
        metrics["mae"] = float(err.abs().mean().item())
        metrics["mse"] = float((err ** 2).mean().item())

    return total_loss / max(1,len(loader)), metrics


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.trial)
    np.random.seed(args.trial)

    train_loader, val_loader, test_loader = make_supervised_loaders(
        train_csv=args.train_file,
        val_csv=args.val_file,
        test_csv=args.test_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    encoder = get_resnet("resnet18", pretrained=False)
    n_features = encoder.fc.in_features 
    encoder.fc = Identity()
    model = LinearHeadModel(encoder, n_features, num_classes=1).to(device)

    # Create paths and csv for checkpointing latest and best fine-tuned/linear probed models
    eval_dir = os.path.join(args.save_path, "evaluation")
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    ckpt_path = os.path.join(eval_dir, f"{args.trial}_{args.tag}_ft_{args.finetune}_latest.pth.tar")
    csv_path =  os.path.join(eval_dir, f"{args.trial}_{args.tag}_ft_{args.finetune}_metrics.csv")
    best_path = os.path.join(eval_dir, f"{args.trial}_{args.tag}_ft_{args.finetune}_best.pth.tar")
    pretrained_path = args.simclr_ckpt

    model, optimizer, start_epoch, best_metric_ckpt = init_model_and_optimizer(model=model, device=device, lr=args.lr, finetune=args.finetune, resume_ckpt=ckpt_path, pretrained_ckpt=pretrained_path)
    
    if args.task_type == "regression":
        criterion = nn.MSELoss()
    elif args.task_type == "classification":
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError("task_type must be 'classification' or 'regression")

    if best_metric_ckpt is not None:
        best_metric = best_metric_ckpt
    else:
        best_metric = float("-inf") if args.task_type == "classification" else float("inf")

    for epoch in range(start_epoch, args.epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device, args)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, args)
        test_loss, test_metrics = evaluate(model, test_loader, criterion, device, args)

        # Log metrics per epoch
        log_metrics_to_csv(csv_path, epoch, train_loss, val_loss, test_loss, val_metrics, test_metrics, args.task_type)
        print(f"Epoch {epoch+1}/{args.epochs} | train {train_loss:.4f} | val {val_loss:.4f} {val_metrics} | test {test_loss:.4f} {test_metrics}")
        
        if args.task_type == "classification":
            score = val_metrics.get("auc")
            is_best = score > best_metric
        else: 
            score = val_loss
            is_best = score < best_metric

        if is_best:
            best_metric = score
            save_ckpt(best_path, model, optimizer, epoch + 1, best_metric=best_metric)
        
        # Save latest ckpt
        save_ckpt(ckpt_path, model, optimizer, epoch + 1, best_metric=best_metric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("./config/config_eval.yaml")
    for k, v in config.items():
        if isinstance(v, bool):
            parser.add_argument(f"--{k}", action="store_true" if not v else "store_false")
        else:
            parser.add_argument(f"--{k}", default=v)
    args = parser.parse_args()
    main(args)
