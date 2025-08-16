import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

FEATURE_EMBED = 256
class ProtoResNet(nn.Module):
    def __init__(self, classes_1, classes_2, classes_3):
        super().__init__()
        # ResNet18
        self.backbone = resnet18(pretrained=True)
        # 修改输入层适配32x32输入
        # self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        # 修改全连接层输出特征
        self.backbone.fc = nn.Linear(512, FEATURE_EMBED)
        # 可学习原型矩阵 [num_classes, 256]
        self.prototypes_1 = nn.Parameter(torch.randn(classes_1, FEATURE_EMBED))
        self.prototypes_2 = nn.Parameter(torch.randn(classes_2, FEATURE_EMBED))
        self.prototypes_3 = nn.Parameter(torch.randn(classes_3, FEATURE_EMBED))
        nn.init.normal_(self.prototypes_1, mean=0, std=0.01)
        nn.init.normal_(self.prototypes_2, mean=0, std=0.01)
        nn.init.normal_(self.prototypes_3, mean=0, std=0.01)
    def forward(self, x):
        features = self.backbone(x)
        return F.normalize(features, dim=1)