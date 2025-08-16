# HCAL
Our paper `"Hierarchy-Consistent Learning and Adaptive Loss Balancing for Hierarchical Multi-Label Classification"` has been accepted by CIKM 2025.

## Environment
GPU: NVIDIA GeForce RTX 3090
> python==3.8.2
> 
> torch==2.4.1+cu121
> 
> torchvision==0.19.1+cu121


## Dataset
Please change in `main.py` or `multimain.py`:
```
from dataloader."dataset name" import train_loader,test_loader
```
