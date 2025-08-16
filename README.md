# HCAL
Our paper `"Hierarchy-Consistent Learning and Adaptive Loss Balancing for Hierarchical Multi-Label Classification"` has been accepted by CIKM 2025.

## Environment
GPU: NVIDIA GeForce RTX 3090
> matplotlib==3.7.5
> 
> numpy==1.24.1
> 
> Pillow==11.2.1
> 
> scikit_learn==1.3.2
> 
> torch==2.4.1+cu121
> 
> torchvision==0.19.1+cu121
> 
> tqdm==4.66.6

## Dataset
Please change in `main.py` or `multimain.py`:
```
from dataloader."dataset name" import train_loader,test_loader
```
