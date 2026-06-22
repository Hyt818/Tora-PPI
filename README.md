# Tora-PPI full model

This folder contains the organized full-model training code copied from
`/home/huangyuting/PPI`.

The original files in `/home/huangyuting/PPI` were not modified.

## Main entry

```bash
python train.py
```

The current `train.py` keeps the original training behavior and default paths.
For non-SHS27k datasets, update the command-line paths and split handling.

## File groups

- `model/data/`: PPI data loading and graph construction
- `model/models/`: Tora-PPI model components
- `model/features/`: sequence and structure feature preprocessing
- `model/graph/`: PPR and subgraph utilities
- `model/layers/`: reusable neural network layers
- `model/training/losses.py`: contrastive and regularization losses
- `model/training/helpers.py`: feature-graph batching helpers and parameter counting
- `model/training/trainer.py`: training and validation loop
- `model/training/profiler.py`: inference efficiency profiling
- `model/utils/`: metrics and general utilities

## Config-based training

```bash
python train.py --config configs/shs27k_random_seed1.yaml
```

Command-line arguments provided after `--config` override values loaded from the YAML file.
