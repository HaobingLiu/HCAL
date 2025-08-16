import torch
import torch.nn as nn
import torch.nn.functional as F

class ProtoLoss(nn.Module):
    def __init__(self, temp=0.1, coarse_weight=0.5, alpha=8,phi=0.3,epsilon=0.01):
        super().__init__()
        self.temp = temp
        self.coarse_weight = coarse_weight
        self.alpha = alpha
        self.phi = phi
        self.epsilon = epsilon

    def perturb_prototype(self, proto):
        noise = (torch.rand_like(proto) - 0.5) * 2 * self.epsilon
        return F.normalize(proto + noise, dim=1)

    def cross_layer_orth_loss(self, fine_p, coarse_p):
        sim_matrix = torch.mm(F.normalize(fine_p), F.normalize(coarse_p).T)
        return torch.mean(torch.abs(sim_matrix))

    def forward(self, features, fine_prototypes, fine_labels, coarse_prototypes, coarse_labels):
        # === 生成扰动原型（双层次） ===
        perturbed_fine = self.perturb_prototype(fine_prototypes)
        perturbed_coarse = self.perturb_prototype(coarse_prototypes)


        fine_sim = torch.mm(features, perturbed_fine.T) / self.temp
        fine_pos = F.one_hot(fine_labels, num_classes=fine_prototypes.size(0))
        fine_loss = -torch.mean(
            (fine_sim * fine_pos).sum(1) -
            torch.logsumexp(fine_sim, dim=1)
        )

        unique_coarse = torch.unique(coarse_labels)
        coarse_features = []
        for c in unique_coarse:
            mask = (coarse_labels == c)
            mean_feature = features[mask].mean(dim=0)
            coarse_features.append(mean_feature)
        coarse_features = torch.stack(coarse_features)

        coarse_sim = torch.mm(coarse_features, perturbed_coarse.T) / self.temp
        coarse_pos = F.one_hot(unique_coarse, num_classes=coarse_prototypes.size(0))
        coarse_loss = -torch.mean(
            (coarse_sim * coarse_pos).sum(1) -
            torch.logsumexp(coarse_sim, dim=1)
        )

        loss = (1 - self.coarse_weight) * fine_loss + self.coarse_weight * coarse_loss

        return loss, fine_loss, coarse_loss

class AWA(nn.Module):
    def __init__(self, num_tasks=2, gamma=0.5):
        super().__init__()
        self.lambdas = nn.Parameter(torch.ones(num_tasks))
        self.gamma = gamma

    def adversarial_step(self, losses):
        with torch.no_grad():
            grad_lambda = [loss.detach() for loss in losses]
            new_lambda = (torch.stack(grad_lambda) / self.gamma).softmax(-1)
        self.lambdas.data = new_lambda



# class ProtoLoss(nn.Module):
#     def __init__(self, temp=0.1, coarse_weight=0.5, alpha=8, phi=0.3, epsilon=0):
#         super().__init__()
#         self.temp = temp
#         self.coarse_weight = coarse_weight
#         self.alpha = alpha
#         self.phi = phi
#         self.epsilon = epsilon
#
#     def perturb_prototype(self, proto):
#         noise = (torch.rand_like(proto) - 0.5) * 2 * self.epsilon
#         return F.normalize(proto + noise, dim=1)
#
#     def cross_layer_orth_loss(self, fine_p, coarse_p):
#         sim_matrix = torch.mm(F.normalize(fine_p), F.normalize(coarse_p).T)
#         return torch.mean(torch.abs(sim_matrix))
#
#     def forward(self, features, fine_prototypes, fine_labels, coarse_prototypes, coarse_labels):
#         perturbed_fine = self.perturb_prototype(fine_prototypes)
#         perturbed_coarse = self.perturb_prototype(coarse_prototypes)
#
#         fine_sim = torch.mm(features, perturbed_fine.T) / self.temp
#         fine_pos = F.one_hot(fine_labels, num_classes=fine_prototypes.size(0))
#         fine_loss = -torch.mean(
#             (fine_sim * fine_pos).sum(1) -
#             torch.logsumexp(fine_sim, dim=1)
#         )
#
#         coarse_sim = torch.mm(features, perturbed_coarse.T) / self.temp
#
#         coarse_pos = F.one_hot(coarse_labels, num_classes=coarse_prototypes.size(0))
#
#         coarse_loss = -torch.mean(
#             (coarse_sim * coarse_pos).sum(1) -
#             torch.logsumexp(coarse_sim, dim=1)
#         )
#
#         # === 组合损失 ===
#         loss = (1 - self.coarse_weight) * fine_loss + self.coarse_weight * coarse_loss
#
#         return loss, fine_loss, coarse_loss
