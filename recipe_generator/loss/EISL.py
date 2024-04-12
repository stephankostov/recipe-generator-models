# edit invariant sequence loss 
# paper: https://arxiv.org/abs/2106.15078 
# implementation: https://github.com/guangyliu/EISL

import torch
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

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
