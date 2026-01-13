# 3D Brain MRI Foundation Model

PyTorch code for training and evaluating **foundation models on 3D T1-weighted brain MRI**, using:

- **Self-supervised learning (SimCLR-style pretraining)**
- **Supervised fine-tuning or linear probing for downstream tasks**

---

## Overview

**This repository supports two training modes:**

### 1. Self-Supervised Learning (SimCLR)

Learns general-purpose image representations from unlabeled MRI volumes by maximizing agreement between two augmented views of the same scan.

- No labels required
- Suitable for large-scale pretraining

### 2. Supervised Learning

Fine-tunes pretrained models on labeled datasets.

- Supports classification or regression
- Uses explicit train / validation / test splits
- Shares the same dataset interface as SSL

---

## Dataset Preprocessing and Formatting

Preprocessing is performed using the publicly available GitHub repository TurboPrep (https://github.com/LemuelPuglisi/turboprep). To reproduce preprocessing used to train our SimCLR 3D Brain Foundation model, please use the following preprocessing step: 

```
turboprep $T1_FILE $OUTPUT_DIR $T1_TEMPLATE -m t1 -r r
```

`$T1_FILE` should be the path to a T1 MRI scan in a `.nii.gz` format.

`$OUTPUT_DIR` is a path to a directory where turboprep will output a `normalized.nii.gz` for the preprocessed MRI and a `mask.nii.gz` for the brain mask. Note that since the file names that turboprep outputs is always the same, you should make a unique output directory for each input file.

`$T1_TEMPLATE` is the path to the MNI 152 ICBM non-linear symmetrical 2009c template, which can be found [here](https://nist.mni.mcgill.ca/icbm-152-nonlinear-atlases-2009/). The path should be similar to `mni_icbm152_nlin_sym_09c/mni_icbm152_t1_tal_nlin_sym_09c.nii`.

`-m t1` is a turboprep option to specify using T1 modality for intensity normalization.

`-r r` is a turboprep option to specify using rigid registration to the template instead of the default affine registration.


## Dataset Files for Training

**After preprocessing, all datasets should be defined using **CSV files** for model training/evaluation. The absolute paths to the image and the brain mask are required (and are both output from the preprocessing steps above).**

### Self-Supervised Pretraining

Requires a **single CSV file** (e.g., `pretrain.csv`) containing:

- `image_path`
- `mask_path`

Each row corresponds to a single MRI volume and its associated brain mask.

### Supervised Training

Requires **three CSV files**:

- `train.csv`
- `val.csv`
- `test.csv`

Each containing:

- `image_path`
- `mask_path`
- `label`

The label may represent:
- binary classification targets
- continuous values (regression)

The interpretation of the label is controlled by the evaluation config file (i.e., `task_type`).

---

## Running Self-Supervised Pretraining

### Command
The code automatically opens `config/config.yaml` for pre-training. 

For pre-training, distributed computing is configured. To run with a single node, use:

```bash
torchrun main.py
```

For a single node with multi-gpu (e.g., 4 gpus), use 

```bash
torchrun --nproc_per_node=4 main.py
```

And for multi-node use (e.g., 4 gpus, 4 nodes):

```bash
torchrun --nproc_per_node=4 --nnodes=4 main.py
```

## Running Supervised Training

### Command
The code automatically opens `config/config_eval.yaml` for downstream fine-tuning/evaluation. 


```bash
python main_eval.py 
```

## Pretrained Model Weights

We provide a pretrained SimCLR 3D Brain MRI foundation model.

### Download

Download the checkpoint from GitHub Releases:

https://github.com/emilykaczmarek/3D-Neuro-SimCLR/releases/download/v1.0.0/simclr_3d_brain_foundation.tar


### Extract

After downloading, extract the file:

```bash
tar -xf simclr_3d_brain_foundation.tar
```

## Acknowledgements

This codebase builds upon and adapts components from the following open-source projects:

- **TurboPrep** — MRI preprocessing and normalization pipeline  
  https://github.com/LemuelPuglisi/turboprep

- **SimCLR** — A Simple Framework for Contrastive Learning of Visual Representations  
  https://github.com/Spijkervet/SimCLR

- **MONAI** — Medical Open Network for AI  
  https://monai.io/

