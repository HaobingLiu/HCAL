import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

FEATURE_EMBED=256
class ProtoResNet(nn.Module):
    def __init__(self, num_classes=100, coarse_classes=20):
        super().__init__()
        self.backbone = resnet18(pretrained=True)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) #cifar100
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Linear(512, FEATURE_EMBED)
        self.fine_prototypes = nn.Parameter(torch.randn(num_classes, FEATURE_EMBED))
        self.coarse_prototypes = nn.Parameter(torch.randn(coarse_classes, FEATURE_EMBED))
        nn.init.normal_(self.fine_prototypes, mean=0, std=0.01)
        nn.init.normal_(self.coarse_prototypes, mean=0, std=0.01)
        self.register_buffer("perturbed_fine", torch.zeros_like(self.fine_prototypes))
        self.register_buffer("perturbed_coarse", torch.zeros_like(self.coarse_prototypes))

    def forward(self, x):
        features = self.backbone(x)
        return F.normalize(features, dim=1)