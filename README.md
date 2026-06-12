# ConHop
This repository contains the implementation of **ConHop: Consistency-Aware Hop-Token Distillation for Low-Latency Heterophilic Graph Inference**.

The manuscript has been submitted to **IEEE ICDE 2027**. The code is provided for reproducibility and review purposes.

## Overview

ConHop is a GNN-to-MLP distillation framework for low-latency graph inference under heterophily. It transfers graph-aware knowledge from a pre-trained GNN teacher to an efficient MLP student, so that the student can perform graph-free online prediction after training.

The framework contains two main components:

- **Hop-Token Structural Representation (HSR)**: constructs multi-hop propagated features and adaptively fuses them through hop-token routing and cross-hop modeling.
- **Consistency-Aware Knowledge Distillation (CKD)**: estimates teacher reliability through prediction consistency under graph/feature perturbations and downweights unstable teacher supervision during distillation.

## File Structure

```text
.
├── dataloader.py        # Dataset loading and train/validation/test split construction
├── models.py            # GNN teachers, MLP student, and hop-token fusion module
├── train.py             # Main training script
├── train_and_eval.py    # Training, evaluation, HSR construction, and CKD weighting
├── utils.py             # Utility functions for seeds, splits, conversion, and evaluation
├── configs/             # Optional YAML hyperparameter files
│   ├── tran.conf.yaml   # Optional configuration for the transductive setting
│   └── ind.conf.yaml    # Optional configuration for the inductive setting
├── data/                # Automatically downloaded datasets
├── outputs/             # Saved teacher logits and checkpoints
└── README.md
```

## Requirements

The experiments were implemented with DGL and PyTorch. The following packages are required:

```text
python >= 3.8
torch == 1.12.0
dgl
torch-geometric
ogb
numpy
scikit-learn
pyyaml
```

A typical installation can be done as follows:

```bash
conda create -n conhop python=3.8
conda activate conhop

pip install torch==1.12.0
pip install dgl
pip install torch-geometric
pip install ogb numpy scikit-learn pyyaml
```

Please install the PyTorch, DGL, and PyG versions that match your CUDA environment.

## Clean Path Layout

The code uses project-local paths by default and does not rely on machine-specific directories.

```text
./data/       # downloaded datasets
./outputs/    # teacher logits and checkpoints
./configs/    # optional YAML configuration files
```

The dataset directory is further organized as:

```text
./data/dgl/   # DGL datasets
./data/pyg/   # PyTorch Geometric datasets, e.g., Penn94
./data/ogb/   # OGB datasets, e.g., ogbn-arxiv
```

The output directory is organized as:

```text
./outputs/{dataset}/{teacher}/{setting}/{split_or_seed}/
```

For example:

```text
./outputs/chameleon/SAGE/tran/0/
```

Each teacher output directory contains:

```text
out.npz             # saved teacher logits
teacher_model.pt    # saved teacher checkpoint
```

You can change these directories using command-line arguments:

```bash
python train.py --data_dir ./data --output_dir ./outputs --config_dir ./configs
```

## Datasets

The code supports both homophilic and heterophilic graph datasets.

Homophilic datasets include:

```text
cora, citeseer, pubmed, cs, phy
```

Heterophilic datasets include:

```text
chameleon, romanempire, arating, penn94, arxiv-year
```

Most datasets are downloaded automatically through DGL, PyTorch Geometric, or OGB. For `penn94`, the code loads the dataset through `torch_geometric.datasets.LINKXDataset` and converts it to a DGL graph. For `arxiv-year`, the code loads `ogbn-arxiv` from OGB and constructs five classes by binning publication years.

## Configuration Files

YAML configuration files are optional. By default, the script looks for:

```text
./configs/tran.conf.yaml
./configs/ind.conf.yaml
```

If the corresponding file is found, it is loaded automatically. If it is not found, the script uses the command-line/default hyperparameters.

You can also provide an explicit config file:

```bash
python train.py --config_path ./configs/tran.conf.yaml
```

## Running Experiments

The training pipeline has two stages:

1. Train a GNN teacher and save its logits/checkpoint.
2. Train the MLP student with HSR and CKD using the saved teacher outputs.

### 1. Train the Teacher

Example: train a GraphSAGE teacher on Chameleon under the transductive setting.

```bash
python train.py \
  --dataset chameleon \
  --exp_setting tran \
  --distill_mode 0 \
  --teacher SAGE \
  --student MLP \
  --data_dir ./data \
  --output_dir ./outputs \
  --device cuda:0
```

This saves teacher outputs to:

```text
./outputs/chameleon/SAGE/tran/{split_id}/
```

### 2. Train the Student

After the teacher has been trained, run:

```bash
python train.py \
  --dataset chameleon \
  --exp_setting tran \
  --distill_mode 1 \
  --teacher SAGE \
  --student MLP \
  --data_dir ./data \
  --output_dir ./outputs \
  --device cuda:0
```

The student training stage loads the saved teacher logits and checkpoint, constructs multi-hop structural features, computes consistency-aware distillation weights, and trains the MLP-based student.

### Inductive Setting

To run the inductive setting, replace `tran` with `ind`.

Teacher training:

```bash
python train.py \
  --dataset chameleon \
  --exp_setting ind \
  --distill_mode 0 \
  --teacher SAGE \
  --student MLP \
  --data_dir ./data \
  --output_dir ./outputs \
  --device cuda:0
```

Student training:

```bash
python train.py \
  --dataset chameleon \
  --exp_setting ind \
  --distill_mode 1 \
  --teacher SAGE \
  --student MLP \
  --data_dir ./data \
  --output_dir ./outputs \
  --device cuda:0
```

## Important Hyperparameters

```text
--feat_hops              Maximum number of propagation hops. Default: 4
--use_adaptive_hop       Whether to enable hop-token fusion. Default: 1
--hop_bottleneck_dim     Bottleneck dimension of hop tokens. Default: 128
--hop_transformer_heads  Number of Transformer heads. Default: 4
--hop_transformer_layers Number of Transformer layers. Default: 1
--edge_drop_rate         Edge dropout rate for CKD perturbation. Default: 0.2
--feat_drop_rate         Feature dropout rate for CKD perturbation. Default: 0.1
--consistency_lambda     Sharpness parameter for CKD weighting. Default: 5.0
--weight_floor           Minimum reliability weight. Default: 0.2
--lamb                   Trade-off between supervised loss and KD loss
--tau                    Temperature for knowledge distillation
```

## Reproducibility Notes

- Results are averaged over multiple dataset splits or random seeds.
- For heterophilic datasets with built-in splits, the code uses the provided `train_mask`, `val_mask`, and `test_mask`.
- For homophilic datasets, the code runs multiple random seeds.
- For `penn94`, five splits are used by default.
- For `arxiv-year`, a fixed random split is generated from the OGB Arxiv graph.
- Minor numerical differences may occur due to GPU hardware, CUDA versions, and library versions.

## Citation

If you find this repository useful, please cite the submitted manuscript:

```bibtex
@misc{conhop2027,
  title  = {ConHop: Consistency-Aware Hop-Token Distillation for Low-Latency Heterophilic Graph Inference},
  author = {Yaogang Geng and Hong Yu and Runxi Zhang and Xiaoling Wang and Ye Wang},
  note   = {Manuscript submitted to IEEE ICDE 2027},
  year   = {2026}
}
```

## Contact


