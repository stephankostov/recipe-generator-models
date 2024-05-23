import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch import Tensor

class MaskedMSELoss(nn.MSELoss):
    
    def __init__(self, activation: bool=True) -> None:
        super().__init__(size_average=None, reduce=None, reduction='none')
        self.activation=activation

    def forward(self, input: Tensor, target: Tensor, mask: Tensor, activation: bool=True) -> Tensor:
        activations = F.sigmoid(input) if activation else input
        loss = F.mse_loss(activations, target, reduction=self.reduction)
        return (loss*mask).sum()/(mask != 0).sum()

class MaxPoolCELoss(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, input, target, masked_weights):
        nll = - F.log_softmax(input, dim=2)
        max_nll, _ = torch.max(nll, dim=2)
        nll_loss = torch.gather(max_nll, dim=1, index=target)
        mean_loss = torch.mean(nll_loss*masked_weights)
        return mean_loss

class SeqBCELoss(nn.BCELoss):

    # Binary Cross Entropy applied on the sequence as a whole (position independent)
    # Each sequence cannot contain a duplicate and is of size n_predictions. This results in a single vocab_size vector, with n_prediction 1 values for each food_id in the vocab_size vector. 
    # input: batch_size x n_predictions x vocab_size
    # evaluation: batch_size x vocab_size
    # output: scalar

    def __init__(self, vocab_size, weight=None, size_average=None, reduce=None, reduction='mean'):
        super().__init__(weight, size_average, reduce, reduction)
        self.vocab_size = vocab_size
        self.vocab_weights = torch.ones([vocab_size], device='cuda')
        self.vocab_weights[0] = 0
        self.vocab_weights[2] = 0

    def multi_label_one_hot(self, sequence):
        sequence = F.one_hot(sequence, self.vocab_size)
        sequence = torch.sum(sequence, axis=1).type(torch.float)
        return sequence
    
    def logit_transform(self, logits, target):
        logits = torch.sum(logits, axis=2) # combine all token predictions into a single one
        probs = F.sigmoid(logits) # activation function on logits
        # probs = probs*self.vocab_weights # nulling special tokens
        selected_probs = torch.gather(probs, 1, target) # calculate loss only on the target tokens
        return selected_probs

    def forward(self, input, target, masked_weights):
        probs = self.logit_transform(input, target)
        loss = F.binary_cross_entropy(probs, torch.ones((probs.shape), device='cuda'), weight=self.weight, reduction='none')
        loss = (loss*masked_weights).mean()
        return loss
    
class MaskedCrossEntropyLoss(nn.Module):
    def forward(self, input, target, mask):
        input = input.transpose(1,2)
        nll_loss = F.cross_entropy(input, target, reduce='none')
        avg_nll_loss = (nll_loss*mask).mean()
        return avg_nll_loss
    
# edit invariant sequence loss 
# paper: https://arxiv.org/abs/2106.15078 
# implementation: https://github.com/guangyliu/EISL

class EISL(_WeightedLoss):

    def __init__(self, eisl_weight, ngram):
        super().__init__()
        self.ngram_factor = eisl_weight
        self.ce_factor = 1 - eisl_weight
        self.ngram = ngram

    def forward(self, output, target, mask=torch.tensor([])):
        
        ce_loss = F.cross_entropy(output, target, reduce=None)
        if mask.numel(): ce_loss = (ce_loss*mask).sum() / (mask != 0).sum()
        else: ce_loss = ce_loss.mean()

        # temp reshaping of vars to suit usual loss TODO
        output = output.transpose(1,2)

        log_probs = F.log_softmax(output, dim=-1)*(mask.unsqueeze(2))

        ngram_loss = self.compute_EISL_cnn(log_probs, target, self.ngram, mask=mask)

        eisl_loss = ngram_loss * self.ngram_factor + ce_loss * self.ce_factor

        return eisl_loss
    
    def compute_EISL_cnn(self, output, target, ngram, pad=0, weight_list=None, mask=torch.tensor([])):

        """
        output: [batch_size, output_len, vocab_size]
            - matrix with probabilities  -- log probs
        target: [batch_size, target_len]
            - reference batch
        ngram_list: int or List[int]
            - n-gram to consider
        pad: int
            the idx of "pad" token
        weight_list : List
            corresponding weight of ngram

        NOTE: output_len == target_len
        """

        batch_size, output_len, vocab_size = output.size()
        _, target_len = target.size()

        output = torch.relu(output + 10) - 10 # unsure origin of this - but its limiting the low probabilities I've changed this to be log(1/vocab_size)

        # [batch_size, output_len, target_len]
        index = target.unsqueeze(1).expand(-1, output_len, target_len)

        # [batch_size, 1, output_len, target_len]
        cost_nll = output.gather(dim=2, index=index).unsqueeze(1)

        # out: [batch, 1, output_len, target_len]
        # eye_filter: [1, 1, ngram, ngram]
        eye_filter = torch.eye(ngram, device=cost_nll.device).view([1, 1, ngram, ngram])

        # term: [batch, output_len - ngram + 1, target_len - ngram + 1]
        conv = F.conv2d(cost_nll, eye_filter).squeeze(1) / ngram

        # maybe dim should be 2, but sometime 1 is better
        gum_softmax = F.gumbel_softmax(conv, tau=1, dim=2)

        loss = - (conv).mean() / (mask.sum() / mask.numel())

        return loss