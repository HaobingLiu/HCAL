import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from model import ProtoResNet
from loss import AWA, ProtoLoss

class ProtoTrainer:
    def __init__(self, num_classes=100, coarse_classes=20, device='cuda:2'):
        self.device = torch.device(device)
        self.model = DataParallel(
            ProtoResNet(num_classes, coarse_classes),
            device_ids=[2]
        ).to(self.device)
        self.awa = AWA().to(device)
        self.criterion = ProtoLoss(coarse_weight=0.6)
        self.optimizer = torch.optim.SGD([
            {'params': self.model.module.backbone.parameters(), 'lr': 0.01},
            {'params': self.model.module.fine_prototypes, 'lr': 0.05},
            {'params': self.model.module.coarse_prototypes, 'lr': 0.1}
        ], momentum=0.9, weight_decay=1e-4)
        self.fine_acc = 0.0
        self.coarse_acc = 0.0
        self.hcvr_acc = 0.0

    def hcvr(self,fine_pre, coarse_pre, fine2coarse):
        fine_pre = fine_pre.tolist()
        coarse_pre = coarse_pre.tolist()
        hcvr = 0
        for i in range(len(fine_pre)):
            if fine2coarse[fine_pre[i]] == coarse_pre[i]:
                hcvr += 1
        return hcvr

    def compute_accuracy(self, features, labels, type="fine"):
        with torch.no_grad():
            if type == "fine":
                prototypes = F.normalize(self.model.module.fine_prototypes)
            elif type == "coarse":
                prototypes = F.normalize(self.model.module.coarse_prototypes)
            sim = torch.mm(features, prototypes.T)
            preds = sim.argmax(dim=1)
            # print(preds)
            acc = (preds == labels).float().mean()
        return acc.item(), preds

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss, total_acc = 0.0, 0.0
        fine_acc = 0
        coarse_acc = 0
        num_hcvr = 0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]", unit="batch", colour='green') as pbar:
            for inputs, fine_labels, coarse_labels in pbar:
                inputs = inputs.to(self.device)
                fine_labels = fine_labels.to(self.device)
                coarse_labels = coarse_labels.to(self.device)

                features = self.model(inputs)
                fine_pro = F.normalize(self.model.module.fine_prototypes)
                coarse_pro = F.normalize(self.model.module.coarse_prototypes)

                loss, fine_loss, coarse_loss = self.criterion(
                    features, fine_pro, fine_labels, coarse_pro, coarse_labels
                )

                fine_prototypes = F.normalize(self.model.module.fine_prototypes)
                fine_sim = torch.mm(features, fine_prototypes.T)
                coarse_prototypes = F.normalize(self.model.module.coarse_prototypes)
                coarse_sim = torch.mm(features, coarse_prototypes.T)

                # 反向传播
                task_losses = [fine_loss, coarse_loss]
                self.awa.adversarial_step(task_losses)
                awa_loss = sum(w * l for w, l in zip(self.awa.lambdas, task_losses))
                self.optimizer.zero_grad()

                awa_loss.backward()
                self.optimizer.step()

                fine_preds = fine_sim.argmax(dim=1)
                acc_f = (fine_preds == fine_labels).float().mean()

                coarse_preds = coarse_sim.argmax(dim=1)
                acc_c = (coarse_preds == coarse_labels).float().mean()
                num_hcvr += self.hcvr(fine_preds, coarse_preds, train_loader.dataset.fine2coarse())

                total_loss += loss.item()
                fine_acc += acc_f
                coarse_acc += acc_c

                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'fine_loss': f"{fine_loss.item():.4f}",
                    'coarse_loss': f"{coarse_loss.item():.4f}",
                    'fine_acc': f"{acc_f:.4f}",
                    'coarse_acc': f"{acc_c:.4f}",
                })

        return total_loss / len(train_loader), fine_acc / len(train_loader), coarse_acc / len(
            train_loader), 1 - num_hcvr / len(train_loader)

    @torch.no_grad()
    def test(self, test_loader):
        self.model.eval()
        all_features, all_fine, all_coarse = [], [], []
        fine_acc, coarse_acc, total_samples = 0.0, 0.0, 0
        num_hcvr = 0

        for inputs, fine_labels, coarse_labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(self.device)
            fine_labels = fine_labels.to(self.device)
            coarse_labels = coarse_labels.to(self.device)

            features = self.model(inputs)
            acc_f, finepre = self.compute_accuracy(features, fine_labels)
            acc_c, coarsepre = self.compute_accuracy(features, coarse_labels, "coarse")
            num_hcvr += self.hcvr(finepre, coarsepre, test_loader.dataset.fine2coarse())

            all_features.append(features.cpu())
            all_fine.append(fine_labels.cpu())
            all_coarse.append(coarse_labels.cpu())
            fine_acc += acc_f * inputs.size(0)
            coarse_acc += acc_c * inputs.size(0)
            total_samples += inputs.size(0)

        final_fine_acc = fine_acc / total_samples
        final_coarse_acc = coarse_acc / total_samples
        hcvr_acc = 1 - num_hcvr / total_samples

        if final_fine_acc > self.fine_acc and final_coarse_acc > self.coarse_acc:
            print("Best Update.")
            self.fine_acc = final_fine_acc
            self.coarse_acc = final_coarse_acc
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

        return final_fine_acc, final_coarse_acc, hcvr_acc

    def visualize(self, features, fine_labels, epoch, num_fine_classes):
        combined = torch.cat([features.cpu(),
                              self.model.module.fine_prototypes.cpu(),
                              self.model.module.coarse_prototypes.cpu()])

        tsne = TSNE(n_components=2, random_state=42)
        embed = tsne.fit_transform(combined.numpy())

        num_samples = len(features)
        num_fine = len(self.model.module.fine_prototypes)
        num_coarse = len(self.model.module.coarse_prototypes)

        feat_2d = embed[:num_samples]
        fine_proto_2d = embed[num_samples: num_samples + num_fine]
        coarse_proto_2d = embed[num_samples + num_fine:]

        cmap = plt.cm.get_cmap('gist_ncar', num_fine_classes)  # 或'nipy_spectral'
        fine_colors = cmap(fine_labels.cpu().numpy() % num_fine_classes)

        plt.figure(figsize=(20, 8), dpi=150)

        plt.subplot(121)
        plt.scatter(feat_2d[:, 0], feat_2d[:, 1], c=fine_colors, alpha=0.7, s=20,
                    edgecolors='w', linewidth=0.3)
        for i in range(num_fine):
            plt.scatter(fine_proto_2d[i, 0], fine_proto_2d[i, 1],
                        marker='*', s=200, color=cmap(i % num_fine_classes),
                        edgecolors='black', linewidth=1, zorder=3)
        plt.title(f'Lower-Level Prototypes and Features')

        plt.subplot(122)
        plt.scatter(feat_2d[:, 0], feat_2d[:, 1], c=fine_colors, alpha=0.7, s=20,
                    edgecolors='w', linewidth=0.3)

        coarse_cmap = plt.cm.get_cmap('tab20', len(coarse_proto_2d))
        for i in range(len(coarse_proto_2d)):
            plt.scatter(coarse_proto_2d[i, 0], coarse_proto_2d[i, 1],
                        marker='o', s=200, color=coarse_cmap(i),
                        edgecolors='black', linewidth=1, zorder=3)
        plt.title(f'Higher-Level Prototypes and Features')

        plt.tight_layout()
        plt.savefig(f'high_color_epoch{epoch}.pdf', bbox_inches='tight')
        plt.close()
