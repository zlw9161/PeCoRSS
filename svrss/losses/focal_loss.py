import torch
import torch.nn as nn


class FocalLoss(nn.Module):

    def __init__(self, weight=None, global_weight = 1., reduction='mean', gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.global_weight = global_weight
        self.gamma = gamma
        self.eps = eps
        self.reduction = reduction
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        if self.reduction == 'mean':
            return self.global_weight * loss.mean()
        else:
            return self.global_weight * loss.sum()


if __name__ == '__main__':
    # 预测为均匀分布
    pred = torch.ones(2, 2, 3, 3)
    # gt为第1类仅有1个，即其他均为背景
    target = torch.zeros((2, 3, 3), dtype=torch.long)
    target[:, 1, 1] = 1
    weight = torch.tensor([0.25, 0.75])
    focal_loss = FocalLoss(weight=weight)
    loss = focal_loss(input=pred, target=target)
    print(loss)