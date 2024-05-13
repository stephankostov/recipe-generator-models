import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class MaskedMSELoss(nn.MSELoss):
    
    def __init__(self) -> None:
        super().__init__(size_average=None, reduce=None, reduction='none')

    def forward(self, input: Tensor, target: Tensor, mask: Tensor, activation: bool = True) -> Tensor:
        activations = F.sigmoid(input) if activation else input
        loss = F.mse_loss(activations, target, reduction=self.reduction)
        return (loss*mask).sum()/(mask != 0).sum()