import torch
import torch.nn as nn


class PairwiseMarginLoss(nn.Module):
    def __init__(self, a: float, m: float, Lp: float):
        super(PairwiseMarginLoss, self).__init__()
        self.a = a
        self.m = m
        self.distance = nn.PairwiseDistance(p=Lp, keepdim=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, y: int):
        loss = y * (self.distance(x1, x2) - self.m) + self.a
        return self.relu(loss)

