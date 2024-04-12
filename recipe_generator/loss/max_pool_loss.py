import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxPoolCELoss(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, input, target, masked_weights):
        nll = - F.log_softmax(input, dim=2)
        max_nll, _ = torch.max(nll, dim=2)
        nll_loss = torch.gather(max_nll, dim=1, index=target)
        mean_loss = torch.mean(nll_loss*masked_weights)
        return mean_loss