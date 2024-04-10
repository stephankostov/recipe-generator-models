# Binary Cross Entropy applied on the sequence as a whole (position independent)
# Each sequence cannot contain a duplicate and is of size n_predictions. This results in a single vocab_size vector, with n_prediction 1 values for each food_id in the vocab_size vector. 
# input: batch_size x n_predictions x vocab_size
# evaluation: batch_size x vocab_size
# output: scalar

import torch
import torch.nn as nn
import torch.nn.functional as F

class SeqBCELoss(nn.BCELoss):

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