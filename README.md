# QGAN–ViT–SFO Reproducibility Package

This package provides the implementation framework corresponding to the
methodology and implementation settings reported in the manuscript.

## Reported configuration

- PyTorch 2.0.1
- TorchVision 0.15.2
- Python 3.10
- Ubuntu 22.04 LTS
- NVIDIA RTX 3090, 24 GB VRAM
- Intel Core i9-12900K
- 128 GB RAM
- AdamW
- Learning rate: 0.0001
- Batch size: 32
- Weight decay: 0.01
- Dropout: 0.25
- QFA scale: 1.75
- Transformer depth: 6
- Attention heads: 8
- Patch size: 16 × 16
- Segmentation loss weight: 0.6
- Classification loss weight: 0.4
- SFO population: 30
- SFO iterations: 50
- Training budget: 200 epochs
- FP16 mixed precision

## Dataset manifest

Create a CSV such as:

image,mask,label
/path/image001.png,/path/mask001.png,0
/path/image002.png,/path/mask002.png,1
...

Use integer class labels matching the manuscript's class definition.

## Training

python scripts/train.py --manifest data/busi_manifest.csv

## Evaluation

python scripts/evaluate.py \
    --manifest data/busi_test_manifest.csv \
    --checkpoint checkpoints/best_model.pth

## Complexity

python profiling/complexity.py

## Figures

python visualization/plots.py

## Important reproducibility note

The manuscript specifies the use of ADF and SFO, but the exact historical
numerical ADF diffusion coefficient and complete SFO update equations are
not sufficiently specified in the manuscript text alone. The supplied
ADF/SFO modules therefore expose these components explicitly and must be
replaced with the authors' original implementation if exact reproduction
of the historical reported run is required.

The package does not fabricate trained weights. The actual checkpoint
generated from the authors' training run should be placed in:

checkpoints/best_model.pth
