import torch
import torch.nn as nn
import torch.nn.functional as F
class AWA(nn.Module):
    def __init__(self, num_tasks=3, gamma=0.5):
        super().__init__()
        self.lambdas = nn.Parameter(torch.ones(num_tasks))
        self.gamma = gamma

    def adversarial_step(self, losses):
        with torch.no_grad():
            grad_lambda = [loss.detach() for loss in losses]
            new_lambda = (torch.stack(grad_lambda) / self.gamma).softmax(-1)
        self.lambdas.data = new_lambda

class ProtoLoss(nn.Module):
    def __init__(self, temp=0.1, weight1=0.5,weight2=0.5, alpha=8,phi=0.3,epsilon=0.01):
        super().__init__()
        self.temp = temp
        self.weight1 = weight1
        self.weight2 = weight2
        self.weight3 = 1-weight2-weight1
        self.alpha = alpha
        self.phi = phi
        self.epsilon = epsilon

    # def perturb_prototype(self, proto):
    #     """生成带均匀扰动的原型（核心PSCL类增强）"""
    #     noise = (torch.rand_like(proto) - 0.5) * 2 * self.epsilon
    #     return F.normalize(proto + noise, dim=1)

    def cross_layer_orth_loss(self, fine_p, coarse_p):
        """层次间原型正交约束（防止层级耦合）"""
        sim_matrix = torch.mm(F.normalize(fine_p), F.normalize(coarse_p).T)
        return torch.mean(torch.abs(sim_matrix))

    # def forward(self, features, prototypes_1, labels_1, prototypes_2, labels_2, prototypes_3, labels_3):
    #     # === 第一层对比损失 ===
    #     # 使用更稳定的log-sum-exp形式计算对比损失
    #     logits1 = torch.mm(features, prototypes_1.T) / self.temp
    #     pos_mask1 = F.one_hot(labels_1, num_classes=prototypes_1.size(0)).bool()
    #     fine_loss = -torch.mean(
    #         torch.logsumexp(logits1 + torch.where(pos_mask1, 0.0, -torch.inf), dim=1) -
    #         torch.logsumexp(logits1, dim=1)
    #     )
    #
    #     # === 第二层对比损失 ===
    #     # 向量化分组池化
    #     unique_labels2, inverse_indices2, counts2 = torch.unique(
    #         labels_2, return_inverse=True, return_counts=True, sorted=True
    #     )
    #
    #     # 预分配内存并执行并行池化
    #     group_features2 = torch.zeros(
    #         unique_labels2.size(0),
    #         features.size(1),
    #         device=features.device,
    #         dtype=features.dtype
    #     )
    #     group_features2.scatter_add_(
    #         0,
    #         inverse_indices2.view(-1, 1).expand(-1, features.size(1)),
    #         features
    #     )
    #     group_features2.div_(counts2.view(-1, 1))
    #
    #     # 计算第二层对比损失
    #     logits2 = torch.mm(group_features2, prototypes_2.T) / self.temp
    #     pos_mask2 = F.one_hot(unique_labels2, num_classes=prototypes_2.size(0)).bool()
    #     coarse_loss = -torch.mean(
    #         torch.logsumexp(logits2 + torch.where(pos_mask2, 0.0, -torch.inf), dim=1) -
    #         torch.logsumexp(logits2, dim=1)
    #     )
    #
    #     # === 第三层对比损失 ===
    #     # 直接从第二层池化结果计算第三层池化
    #     unique_labels3, inverse_indices3, counts3 = torch.unique(
    #         labels_3, return_inverse=True, return_counts=True, sorted=True
    #     )
    #
    #     # 将第三层标签映射到第二层分组上
    #     # 找到第二层分组对应的第三层标签
    #     _, group2_to_label3 = torch.unique(labels_2, return_inverse=True)
    #     group_labels3 = torch.gather(labels_3, 0, inverse_indices2)
    #
    #     # 再次执行分组池化
    #     group_features3 = torch.zeros(
    #         unique_labels3.size(0),
    #         group_features2.size(1),
    #         device=features.device,
    #         dtype=features.dtype
    #     )
    #     group_features3.scatter_add_(
    #         0,
    #         inverse_indices3.view(-1, 1).expand(-1, group_features2.size(1)),
    #         group_features2
    #     )
    #     group_features3.div_(counts3.view(-1, 1))
    #
    #     # 计算第三层对比损失
    #     logits3 = torch.mm(group_features3, prototypes_3.T) / self.temp
    #     pos_mask3 = F.one_hot(unique_labels3, num_classes=prototypes_3.size(0)).bool()
    #     con_coarse_loss = -torch.mean(
    #         torch.logsumexp(logits3 + torch.where(pos_mask3, 0.0, -torch.inf), dim=1) -
    #         torch.logsumexp(logits3, dim=1)
    #     )
    #
    #     # === 组合总损失 ===
    #     loss = (
    #             self.weight1 * fine_loss +
    #             self.weight2 * coarse_loss +
    #             self.weight3 * con_coarse_loss
    #     )
    #
    #     return loss, fine_loss, coarse_loss, con_coarse_loss

    def forward(self, features, prototypes_1, labels_1, prototypes_2, labels_2, prototypes_3, labels_3):
        # === 生成扰动原型（双层次） ===
        # perturbed_fine = self.perturb_prototype(fine_prototypes)
        # perturbed_coarse = self.perturb_prototype(coarse_prototypes)

        fine_pos_mask = F.one_hot(labels_1, num_classes=prototypes_1.size(0)).float()
        fine_sim = torch.mm(features, prototypes_1.T) / self.temp
        fine_pos_sim = (fine_sim * fine_pos_mask).sum(dim=1)
        fine_neg_sim = torch.logsumexp(fine_sim, dim=1)
        #fine_loss = -torch.mean(fine_pos_sim - fine_neg_sim)
        fine_loss = -torch.mean(
            (fine_sim * fine_pos_mask).sum(1) -
            torch.logsumexp(fine_sim, dim=1)
        )

        unique_coarse = torch.unique(labels_2)
        coarse_features = []
        for c in unique_coarse:
            mask = (labels_2 == c)
            mean_feature = features[mask].mean(dim=0)  # 均值池化
            coarse_features.append(mean_feature)
        coarse_features = torch.stack(coarse_features)  # [num_unique_coarse, 512]

        coarse_sim = torch.mm(coarse_features, prototypes_2.T) / self.temp
        coarse_pos_mask = F.one_hot(unique_coarse, num_classes=prototypes_2.size(0)).float()
        coarse_pos_sim = (coarse_sim * coarse_pos_mask).sum(dim=1)
        coarse_neg_sim = torch.logsumexp(coarse_sim, dim=1)
        coarse_loss = -torch.mean(coarse_pos_sim - coarse_neg_sim)
        # coarse_sim = torch.mm(coarse_features, perturbed_coarse.T) / self.temp
        # coarse_pos = F.one_hot(unique_coarse, num_classes=coarse_prototypes.size(0))
        # coarse_loss = -torch.mean(
        #     (coarse_sim * coarse_pos).sum(1) -
        #     torch.logsumexp(coarse_sim, dim=1)
        # )

        # orth_loss = self.cross_layer_orth_loss(fine_prototypes, coarse_prototypes)

        mean_coarse = torch.unique(labels_3)
        con_coarse_features = []
        for c in mean_coarse:
            mask = (labels_3 == c)
            #mean_feature = coarse_features[mask].mean(dim=0)  # 均值池化
            mean_feature = features[mask].mean(dim=0)
            con_coarse_features.append(mean_feature)
        con_coarse_features = torch.stack(con_coarse_features)

        con_coarse_sim = torch.mm(con_coarse_features, prototypes_3.T) / self.temp
        con_coarse_pos_mask = F.one_hot(mean_coarse, num_classes=prototypes_3.size(0)).float()
        con_coarse_pos_sim = (con_coarse_sim * con_coarse_pos_mask).sum(dim=1)
        con_coarse_neg_sim = torch.logsumexp(con_coarse_sim, dim=1)
        con_coarse_loss = -torch.mean(con_coarse_pos_sim - con_coarse_neg_sim)

        loss = self.weight1 * fine_loss + self.weight2 * coarse_loss + self.weight3 * con_coarse_loss

        return loss, fine_loss, coarse_loss, con_coarse_loss

class DynamicMTLLoss(nn.Module):
    def __init__(self):
        super(DynamicMTLLoss, self).__init__()
        # 初始化三个可学习参数 θ1, θ2, θ3（初始为0）
        self.theta1 = nn.Parameter(torch.tensor(0.0, requires_grad=True))
        self.theta2 = nn.Parameter(torch.tensor(0.0, requires_grad=True))
        self.theta3 = nn.Parameter(torch.tensor(0.0, requires_grad=True))

    def forward(self, loss_1, loss_2, loss_3):
        # 每个任务对应的动态权重：exp(-θ)
        weight1 = torch.exp(-self.theta1)
        weight2 = torch.exp(-self.theta2)
        weight3 = torch.exp(-self.theta3)

        # 多任务联合损失
        loss = (1.0 / 3.0) * (
            weight1 * loss_1 +
            weight2 * loss_2 +
            weight3 * loss_3
        ) + (self.theta1 + self.theta2 + self.theta3)

        return loss