import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedCrossEntropyLoss(nn.Module):
    def forward(self, input, target, mask):
        nll_loss = F.cross_entropy(input, target, reduce='none')
        avg_nll_loss = (nll_loss*mask).mean()
        return avg_nll_loss