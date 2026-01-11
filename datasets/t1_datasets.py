import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from torch import Tensor


class CSVPairBase(Dataset):
    """
    Shared loader for (image, mask) pairs.
    - CSV must include image_col + mask_col.
    - Applies your transpose+crop.
    - Provides masked standardization helper.
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        *,
        image_col: str = "image_path",
        mask_col: str = "mask_path",
        root_dir: Optional[Union[str, Path]] = None,  # allow relative paths in CSV
    ):
        self.csv_path = str(csv_path)
        self.image_col = image_col
        self.mask_col = mask_col
        self.root_dir = Path(root_dir) if root_dir is not None else None

        df = pd.read_csv(self.csv_path)
        for col in (self.image_col, self.mask_col):
            if col not in df.columns:
                raise ValueError(f"Required column {col!r} not found in CSV columns: {list(df.columns)}")

        self.rows = df.to_dict(orient="records")

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, p: Any) -> str:
        if p is None or (isinstance(p, float) and np.isnan(p)):
            raise ValueError("Found missing path in CSV; image_path and mask_path are required.")
        p = str(p)
        if self.root_dir is not None and not Path(p).is_absolute():
            return str(self.root_dir / p)
        return p

    @staticmethod
    def _load_nifti(path: str) -> np.ndarray:
        return nib.load(path).get_fdata().astype(np.float32)

    @staticmethod
    def _transpose_crop(arr: np.ndarray) -> np.ndarray:
        return np.transpose(arr, (0, 3, 2, 1))[:, 10:160, 19:211, 1:]

    @staticmethod
    def standardize(images: Tensor, masks: Tensor, eps: float = 1e-12) -> Tensor:
        images = images.mul(masks)
        N = masks.sum(dim=(2, 3, 4), keepdims=True).clamp_min(1.0)
        means = images.sum(dim=(2, 3, 4), keepdims=True) / N
        stds = torch.sqrt(images.pow(2).sum(dim=(2, 3, 4), keepdims=True) / N - means.pow(2) + eps)
        images = images.sub(means)
        images = images.div(stds)
        images = images.mul(masks)
        return images

    def _load_pair(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        img_path = self._resolve(row[self.image_col])
        msk_path = self._resolve(row[self.mask_col])

        img = self._load_nifti(img_path)
        msk = self._load_nifti(msk_path)

        if img.shape != msk.shape:
            raise ValueError(f"Shape mismatch: {img_path} vs {msk_path}: {img.shape} vs {msk.shape}")

        # add channel dim -> (1, Z, Y, X)
        img = img[None, ...]
        msk = msk[None, ...]

        # transpose + crop
        img = self._transpose_crop(img)
        msk = self._transpose_crop(msk)

        return {
            "MRI": img,
            "MASK": msk,
            "image_path": img_path,
            "mask_path": msk_path,
            "row": row, 
        }


class SimCLRT1Dataset(CSVPairBase):
    def __init__(
        self,
        csv_path: Union[str, Path],
        transform,
        *,
        image_col: str = "image_path",
        mask_col: str = "mask_path",
        root_dir: Optional[Union[str, Path]] = None,
        eps: float = 1e-12,
    ):
        super().__init__(csv_path, image_col=image_col, mask_col=mask_col, root_dir=root_dir)
        self.transform = transform
        self.eps = eps

    def __getitem__(self, idx: int):
        sample = self._load_pair(idx)

        def process_view(raw_dict):
            while True:
                d = self.transform({"MRI": raw_dict["MRI"], "MASK": raw_dict["MASK"]})
                img = d["MRI"].float().unsqueeze(0)  # [1, C, Z, Y, X]
                msk = d["MASK"].float().unsqueeze(0)

                img = self.standardize(img, msk, eps=self.eps).squeeze(0)  # [C, Z, Y, X]

                if torch.isnan(msk).any() or torch.isinf(msk).any() or torch.isnan(img).any() or torch.isinf(img).any():
                    continue
                if msk.abs().max() == 0 or img.abs().max() == 0:
                    continue
                break

            return img

        x1 = process_view(sample)
        x2 = process_view(sample)
        return x1, x2


class SupervisedT1Dataset(CSVPairBase):
    def __init__(
        self,
        csv_path: Union[str, Path],
        *,
        label_col: str,
        image_col: str = "image_path",
        mask_col: str = "mask_path",
        root_dir: Optional[Union[str, Path]] = None,
        label_dtype: torch.dtype = torch.float32,
        eps: float = 1e-12,
    ):
        super().__init__(csv_path, image_col=image_col, mask_col=mask_col, root_dir=root_dir)
        self.label_col = label_col
        self.label_dtype = label_dtype
        self.eps = eps

        # validate label_col exists in CSV
        if len(self.rows) > 0 and self.label_col not in self.rows[0]:
            raise ValueError(f"label_col={label_col!r} not found in CSV. Available keys: {list(self.rows[0].keys())}")

    def __getitem__(self, idx: int):
        sample = self._load_pair(idx)

        img = torch.from_numpy(sample["MRI"]).float().unsqueeze(0)  # [1, C, Z, Y, X]
        msk = torch.from_numpy(sample["MASK"]).float().unsqueeze(0)

        img = self.standardize(img, msk, eps=self.eps).squeeze(0)  # [C, Z, Y, X]

        raw_y = sample["row"][self.label_col]
        y = torch.tensor(raw_y, dtype=self.label_dtype)

        return {"MRI":img, "task_label":y}
