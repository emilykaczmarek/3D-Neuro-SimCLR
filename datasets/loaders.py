import torch
from datasets.t1_datasets import SupervisedT1Dataset


def make_supervised_loaders(
    *,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    batch_size: int,
    num_workers: int = 8,
    pin_mem: bool = True,
    drop_last: bool = True,
    label_dtype: torch.dtype = torch.float32,
    label_map=None,
    root_dir=None,
    label_col="label",
    image_col="image_path",
    mask_col="mask_path",
):
    train_ds = SupervisedT1Dataset(
        train_csv,
        label_col=label_col,
        label_dtype=label_dtype,
        root_dir=root_dir,
        image_col=image_col,
        mask_col=mask_col,
    )

    val_ds = SupervisedT1Dataset(
        val_csv,
        label_col=label_col,
        label_dtype=label_dtype,
        root_dir=root_dir,
        image_col=image_col,
        mask_col=mask_col,
    )

    test_ds = SupervisedT1Dataset(
        test_csv,
        label_col=label_col,
        label_dtype=label_dtype,
        root_dir=root_dir,
        image_col=image_col,
        mask_col=mask_col,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=False,
    )

    return train_loader, val_loader, test_loader
