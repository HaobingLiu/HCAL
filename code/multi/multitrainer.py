import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from multimodel import ProtoResNet
from multiloss import AWA, ProtoLoss, DynamicMTLLoss

class ProtoTrainer:
    def __init__(self, classes1=100, classes2=20, classes3=10, device='cuda:2'):
        self.device = torch.device(device)
        self.model = DataParallel(
            ProtoResNet(classes1, classes2, classes3),
            device_ids=[2]
        ).to(self.device)
        self.awa = AWA().to(device)
        self.dy = DynamicMTLLoss().to(device)
        self.criterion = ProtoLoss()
        self.optimizer = torch.optim.SGD([
            {'params': self.model.module.backbone.parameters(), 'lr': 0.01},
            {'params': self.model.module.prototypes_1, 'lr': 0.05},
            {'params': self.model.module.prototypes_2, 'lr': 0.1},
            {'params': self.model.module.prototypes_3, 'lr': 0.2}
        ], momentum=0.9, weight_decay=1e-4)
        self.acc1 = 0.0
        self.acc2 = 0.0
        self.acc3 = 0.0
        self.best_hcvr = 0.0

    def hcvr(self,pre_1, pre_2, pre_3, h1to2,h2to3):
        pre_1 = pre_1.tolist()
        pre_2 = pre_2.tolist()
        pre_3 = pre_3.tolist()
        hcvr = 0
        for i in range(len(pre_1)):
            if h1to2[pre_1[i]] == pre_2[i] and h2to3[pre_2[i]] == pre_3[i]:
                hcvr += 1
        return hcvr

    def compute_accuracy(self, features, labels, type = "1"):
        """计算分类准确率"""
        with torch.no_grad():
            if type == "1":
                prototypes = F.normalize(self.model.module.prototypes_1)
            elif type == "2":
                prototypes = F.normalize(self.model.module.prototypes_2)
            elif type == "3":
                prototypes = F.normalize(self.model.module.prototypes_3)
            sim = torch.mm(features, prototypes.T)
            preds = sim.argmax(dim=1)
            acc = (preds == labels).float().mean()
        return acc.item(), preds

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss, total_acc = 0.0, 0.0
        acc_h1 = 0
        acc_h2 = 0
        acc_h3 = 0
        num_hcvr = 0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]", unit="batch", colour='green') as pbar:
            for inputs, labels_1, labels_2, labels_3 in pbar:
                inputs = inputs.to(self.device)
                labels_1 = labels_1.to(self.device)
                labels_2 = labels_2.to(self.device)
                labels_3 = labels_3.to(self.device)

                # 前向传播
                features = self.model(inputs)
                pro_1 = F.normalize(self.model.module.prototypes_1)
                pro_2 = F.normalize(self.model.module.prototypes_2)
                pro_3 = F.normalize(self.model.module.prototypes_3)

                # 计算原始损失
                loss, loss_1, loss_2, loss_3 = self.criterion(
                    features, pro_1, labels_1, pro_2, labels_2, pro_3, labels_3
                )

                task_losses = [loss_1, loss_2, loss_3]
                self.awa.adversarial_step(task_losses)
                # dy = self.dy(loss_1, loss_2, loss_3)
                awa_loss = sum(w * l for w, l in zip(self.awa.lambdas, task_losses))

                sim_1 = torch.mm(features, pro_1.T)
                sim_2 = torch.mm(features, pro_2.T)
                sim_3 = torch.mm(features, pro_3.T)
                # 反向传播
                self.optimizer.zero_grad()
                awa_loss.backward()
                #loss.backward()
                # dy.backward()
                self.optimizer.step()

                preds_1 = sim_1.argmax(dim=1)
                acc_1 = (preds_1 == labels_1).float().mean()

                preds_2 = sim_2.argmax(dim=1)
                acc_2 = (preds_2 == labels_2).float().mean()

                preds_3 = sim_3.argmax(dim=1)
                acc_3 = (preds_3 == labels_3).float().mean()

                num_hcvr += self.hcvr(preds_1, preds_2, preds_3, train_loader.dataset.h1to2(), train_loader.dataset.h2to3())

                total_loss += loss.item()
                acc_h1 += acc_1
                acc_h2 += acc_2
                acc_h3 += acc_3

                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'loss_1': f"{loss_1.item():.4f}",
                    'loss_2': f"{loss_2.item():.4f}",
                    'acc_1': f"{acc_1:.4f}",
                    'acc_2': f"{acc_2:.4f}",
                    'acc_3': f"{acc_3:.4f}",
                })

        return total_loss / len(train_loader), acc_h1 / len(train_loader), acc_h2 / len(
            train_loader),acc_h3 / len(train_loader), 1 - num_hcvr / len(train_loader)

    @torch.no_grad()
    def test(self, test_loader, epoch=0):
        self.model.eval()
        all_features, all_h1, all_h2, all_h3 = [], [], [], []
        acc_h1, acc_h2, acc_h3, total_samples = 0.0, 0.0, 0.0, 0
        num_hcvr = 0

        # 收集特征和标签
        for inputs, labels_1, labels_2, labels_3 in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(self.device)
            labels_1 = labels_1.to(self.device)
            labels_2 = labels_2.to(self.device)
            labels_3 = labels_3.to(self.device)

            features = self.model(inputs)
            acc_1, pre_1 = self.compute_accuracy(features, labels_1, "1")
            acc_2, pre_2 = self.compute_accuracy(features, labels_2, "2")
            acc_3, pre_3 = self.compute_accuracy(features, labels_3, "3")
            num_hcvr += self.hcvr(pre_1, pre_2, pre_3, test_loader.dataset.h1to2(), test_loader.dataset.h2to3())

            all_features.append(features.cpu())
            all_h1.append(labels_1.cpu())
            all_h2.append(labels_2.cpu())
            all_h3.append(labels_3.cpu())
            acc_h1 += acc_1 * inputs.size(0)
            acc_h2 += acc_2 * inputs.size(0)
            acc_h3 += acc_3 * inputs.size(0)
            total_samples += inputs.size(0)

        # 计算整体准确率
        final_acc_1 = acc_h1 / total_samples
        final_acc_2 = acc_h2 / total_samples
        final_acc_3 = acc_h3 / total_samples
        hcvr_acc = 1 - num_hcvr / total_samples

        # 保存最佳模型
        if final_acc_1 > self.acc1 and final_acc_2 > self.acc2 and final_acc_3 > self.acc3:
            print("Best Update.")
            self.acc1 = final_acc_1
            self.acc2 = final_acc_2
            self.acc3 = final_acc_3
            self.hcvr_acc = hcvr_acc
            # checkpoint = {
            #     'model': self.model.state_dict(),
            #     'optimizer': self.optimizer.state_dict(),
            #     'fine_best_acc': self.fine_acc,
            #     'coarse_best_acc': self.coarse_acc,
            #     'epoch': epoch,
            # }
            # torch.save(checkpoint, f'./air/1/best_model_epoch{epoch}.pth')

            # self.visualize(features=torch.cat(all_features),fine_labels=torch.cat(all_fine),
            #                epoch=epoch,num_fine_classes=test_loader.dataset.len_fine())

        return final_acc_1, final_acc_2, final_acc_3, hcvr_acc
