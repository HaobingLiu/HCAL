import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
#from cifar100 import train_loader, test_loader
from dataloader.aircraft import train_loader_3, test_loader_3
from tqdm import tqdm
import random
import numpy as np

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练模型
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # 修改第一层卷积适配32x32输入
        # resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()  # 移除原始maxpool层

        # 移除最后的全连接层
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 特征维度适配
        self.feature_dim = 512  # ResNet-18最终特征维度

    def forward(self, x):
        x = self.backbone(x)  # [batch, 512, 4, 4]
        x = self.avgpool(x)  # [batch, 512, 1, 1]
        x = x.view(x.size(0), -1)  # [batch, 512]
        return x


class ImageMMOE(nn.Module):
    def __init__(self, n_expert=3, mmoe_hidden_dim=128, num_tasks=3, num_1 = 100, num_2 = 20, num_3 = 10):
        super(ImageMMOE, self).__init__()
        self.resnet = CNNBackbone()
        hidden_size = 512  # CNN输出维度

        # 冻结底层参数 (可选)
        for param in self.resnet.parameters():
            param.requires_grad = False  # 冻结所有预训练层
        # 解冻最后两层
        for param in self.resnet.backbone[-2:].parameters():
            param.requires_grad = True

        # 专家网络（每个专家为全连接层）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, mmoe_hidden_dim),
                nn.ReLU()
            ) for _ in range(n_expert)
        ])

        # 门控网络（每个任务独立）
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, n_expert),
                nn.Softmax(dim=1)
            ) for _ in range(num_tasks)
        ])

        # 任务塔
        self.task_towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(mmoe_hidden_dim, 256),
                nn.ReLU(),
                nn.Linear(256, num_1 if i == 0 else (num_2 if i == 1 else num_3))
            ) for i in range(num_tasks)
        ])

    def forward(self, x):
        features = self.resnet(x)  # [batch, 512]

        # 专家输出
        expert_outputs = torch.stack([expert(features) for expert in self.experts],
                                     dim=1)  # [batch, mmoe_hidden, n_expert]

        # 门控权重
        gate_outputs = [gate(features) for gate in self.gates]  # 两个任务的门控权重列表

        # 加权专家输出
        task_inputs = [
            torch.einsum('be,bec->bc', gw, expert_outputs)  # [batch, mmoe_hidden]
            for gw in gate_outputs
        ]

        # 任务预测
        task_outputs = [tower(task_input) for tower, task_input in zip(self.task_towers, task_inputs)]
        return task_outputs


def hcvr(pre_1, pre_2, pre_3, h1to2, h2to3):
    pre_1 = pre_1.tolist()
    pre_2 = pre_2.tolist()
    pre_3 = pre_3.tolist()
    hcvr = 0
    for i in range(len(pre_1)):
        if h1to2[pre_1[i]] == pre_2[i] and h2to3[pre_2[i]] == pre_3[i]:
            hcvr += 1
    return hcvr

def train_model(model, train_loader, val_loader, optimizer, criterion_1, criterion_2, criterion_3, device, epochs=50):
    # 初始化最佳准确率记录
    best_acc_1 = 0.0
    best_acc_2 = 0.0
    best_acc_3 = 0.0
    best_hcvr_acc = 0.0
    # fine2coarse = train_loader.dataset.fine2coarse()

    # 主训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        train_correct_1 = 0
        train_correct_2 = 0
        train_correct_3 = 0
        train_hcvr_correct = 0

        # 训练阶段（带进度条）
        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch + 1}/{epochs} [Train]",
                  colour='green') as train_pbar:
            for imgs, labels_1, labels_2, labels_3 in train_pbar:
                # 数据迁移到设备
                imgs = imgs.to(device)
                labels_1 = labels_1.to(device)
                labels_2 = labels_2.to(device)
                labels_3 = labels_3.to(device)

                # 前向传播
                optimizer.zero_grad()
                pred_1, pred_2, pred_3 = model(imgs)
                #print(fine_pred,coarse_pred)
                # 计算损失
                loss_1 = criterion_1(pred_1, labels_1)
                loss_2 = criterion_2(pred_2, labels_2)
                loss_3 = criterion_3(pred_3, labels_3)
                loss = loss_1 + loss_2 + loss_3

                # 反向传播
                loss.backward()
                optimizer.step()

                # 统计指标
                total_loss += loss.item()
                train_correct_1 += (pred_1.argmax(1) == labels_1).sum().item()
                train_correct_2 += (pred_2.argmax(1) == labels_2).sum().item()
                train_correct_3 += (pred_3.argmax(1) == labels_3).sum().item()
                train_hcvr_correct += hcvr(pred_1.argmax(1), pred_2.argmax(1),pred_3.argmax(1), train_loader.dataset.h1to2(),train_loader.dataset.h2to3())

                # 实时更新进度条信息
                train_pbar.set_postfix({
                    'Loss': f"{loss.item():.3f}",
                    'Acc 1': f"{(pred_1.argmax(1) == labels_1).float().mean().item():.3f}",
                    'Acc 2': f"{(pred_2.argmax(1) == labels_2).float().mean().item():.3f}",
                    'Acc 3': f"{(pred_3.argmax(1) == labels_3).float().mean().item():.3f}",
                })

        # 计算训练指标
        avg_train_loss = total_loss / len(train_loader)
        train_acc_1 = train_correct_1 / len(train_loader.dataset)
        train_acc_2 = train_correct_2 / len(train_loader.dataset)
        train_acc_3 = train_correct_3 / len(train_loader.dataset)
        train_hcvr = 1 - train_hcvr_correct / len(train_loader.dataset)

        model.eval()
        val_correct_1 = 0
        val_correct_2 = 0
        val_correct_3 = 0
        val_hcvr_correct = 0
        with torch.no_grad(), tqdm(val_loader, unit="batch", desc=f"Epoch {epoch + 1}/{epochs} [Val]  ",
                                   colour='yellow') as val_pbar:
            for imgs, labels_1, labels_2, labels_3 in val_pbar:
                imgs = imgs.to(device)
                labels_1 = labels_1.to(device)
                labels_2 = labels_2.to(device)
                labels_3 = labels_3.to(device)

                # 前向推理
                pred_1, pred_2, pred_3 = model(imgs)

                # 统计正确数
                val_correct_1 += (pred_1.argmax(1) == labels_1).sum().item()
                val_correct_2 += (pred_2.argmax(1) == labels_2).sum().item()
                val_correct_3 += (pred_3.argmax(1) == labels_3).sum().item()
                val_hcvr_correct += hcvr(pred_1.argmax(1), pred_2.argmax(1),pred_3.argmax(1), train_loader.dataset.h1to2(),train_loader.dataset.h2to3())

                # 计算当前已处理样本数
                current_samples = val_pbar.n * val_loader.batch_size
                if current_samples == 0:
                    current_samples = 1  # 防止第一次迭代除零

                # 更新进度条
                val_pbar.set_postfix({
                    'Acc 1': f"{val_correct_1 / current_samples:.3f}",
                    'Acc 2': f"{val_correct_2 / current_samples:.3f}",
                    'Acc 3': f"{val_correct_3 / current_samples:.3f}"
                })

        # 计算验证指标
        val_acc_1 = val_correct_1 / len(val_loader.dataset)
        val_acc_2 = val_correct_2 / len(val_loader.dataset)
        val_acc_3 = val_correct_3 / len(val_loader.dataset)
        val_hcvr = 1 - val_hcvr_correct / len(val_loader.dataset)

        # 保存最佳模型
        if val_acc_1 >= best_acc_1 and val_acc_2 >= best_acc_2 and val_acc_3 >=best_acc_3:
            best_acc_1 = val_acc_1
            best_acc_2 = val_acc_2
            best_acc_3 = val_acc_3
            best_hcvr_acc = val_hcvr
            # torch.save(model.state_dict(), f"./mmoe/air/best_model_epoch{epoch + 1}.pth")

        # 打印epoch总结
        print(f"\nEpoch {epoch + 1} Summary:")
        print(
            f"Train Loss: {avg_train_loss:.4f} | Acc 1: {train_acc_1:.4f} | Acc 2: {train_acc_2:.4f} | Acc 3: {train_acc_3:.4f} | HCVR: {train_hcvr: .4f}")
        print(
            f"Val   Loss: {'N/A':<7} | Acc 1: {val_acc_1:.4f} | Acc 2: {val_acc_2:.4f} | Acc 3: {val_acc_3:.4f} | HCVR: {val_hcvr: .4f}")
        print(
            f"Best  Val  | Acc 1: {best_acc_1:.4f} | Acc 2: {best_acc_2:.4f} | Acc 3: {best_acc_3:.4f} | HCVR: {best_hcvr_acc:.4f}\n")

    return model

if __name__ == "__main__":
    seed_everything(9)
    model = ImageMMOE(num_1=train_loader_3.dataset.len_1(), num_2=train_loader_3.dataset.len_2(),num_3=train_loader_3.dataset.len_3())
    # print(train_loader_3.dataset.len_1(), train_loader_3.dataset.len_2(),train_loader_3.dataset.len_3())
    # lr=1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_model(model, train_loader_3, test_loader_3, optimizer, criterion, criterion, criterion, device, epochs=100)
    # pth_test(model,"cifar10_best_model_epoch48.pth",test_loader,device,criterion_fine, criterion_coarse)
