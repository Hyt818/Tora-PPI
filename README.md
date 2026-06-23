# Tora-PPI

Tora-PPI is a topology-aware framework for robust multilabel PPI prediction. By combining PPR-enhanced residue topology, ego-subgraph interaction context, and representation alignment across sequence, structure, and network views, Tora-PPI improves prediction stability under unseen proteins, dataset shifts, species transfer, structural noise, and mutation-induced perturbations.

## Repository Structure

```text
Tora_PPI/
├── train.py                    # Training entry point
├── configs/                    # YAML training configurations
├── data/SHS27k/                # SHS27k input data and precomputed features
├── split_data/                 # Train/test split JSON files
├── model/
│   ├── data/                   # PPI data loading and graph construction
│   ├── features/               # Sequence and molecular/structure preprocessing
│   ├── graph/                  # PPR and subgraph utilities
│   ├── layers/                 # Reusable neural network layers
│   ├── models/                 # Tora-PPI model architecture
│   ├── training/               # Losses, trainer, profiling, helper functions
│   └── utils/                  # Metrics, logging, seed control, split utilities
├── environment.yml             # Conda environment file
└── requirements.txt            # Pip package snapshot
```

## Model Overview

The main model is implemented in:

```text
model/models/gnn_models.py
```

The training pipeline uses:

- residue-level graph features from `FSP_residue_Embedding.pt`
- protein-level sequence features from `FSP_sequence_Embedding.pt`
- PPR-enhanced residue graph edges from `edge_index_ppr.npy` and `edge_attr_ppr.npy`
- PPI train/test split files from `split_data/`
- a multi-label BCE loss with auxiliary InfoNCE and VICReg losses

The default prediction target contains seven interaction types:

```text
reaction, binding, ptmod, activation, inhibition, catalysis, expression
```

## Data

The SHS27k data used by the provided configs is stored under:

```text
data/SHS27k/
```

Expected files:

```text
protein.actions.SHS27k.txt
protein.SHS27k.sequences.dictionary.tsv
vec5_CTC.txt
FSP_residue_Embedding.pt
FSP_sequence_Embedding.pt
edge_index_ppr.npy
edge_attr_ppr.npy
```

Large `.pt` and `.npy` files are tracked with Git LFS. After cloning the
repository, run:

```bash
git lfs install
git lfs pull
```

## Environment

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate Tora-PPI
```

If your environment name is different, activate the corresponding environment
before running the training script.

## Training

Run training from the repository root:

```bash
cd /path/to/Tora_PPI
python train.py --config configs/shs27k_random_seed9.yaml
```

Available config files in the current repository:

```text
configs/shs27k_random_seed9.yaml
```

Example:

```bash
python train.py --config configs/shs27k_random_seed9.yaml
```

Command-line arguments after `--config` override YAML values. For example:

```bash
python train.py \
  --config configs/shs27k_random_seed9.yaml \
  --cuda 0 \
  --epoch_num 500
```

## Outputs

Training outputs are written to:

```text
results/SHS27k/<split>/seed_<seed>_run_id_<run_id>_<timestamp>/
```

Typical files:

```text
config.txt
valid_results.txt
gnn_model_train.ckpt
gnn_model_valid_best.ckpt
gnn_model_final.ckpt
```


## Notes

- Run commands from the repository root so that relative paths in YAML configs
  resolve correctly.
- If a config uses a split file whose seed does not match `seed_num`,
  `train.py` raises an error to prevent accidental mixed-seed experiments.
- Large data files require Git LFS.
