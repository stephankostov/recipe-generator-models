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

    def multi_label_one_hot(self, sequence):
        sequence = F.one_hot(sequence, self.vocab_size)
        sequence = torch.sum(sequence, axis=1).type(torch.float)
        return sequence

    def forward(self, input, target, masked_weights):
        input_probs = F.sigmoid(torch.sum(input, axis=2))
        encoded_target = self.multi_label_one_hot(target)
        loss = F.binary_cross_entropy(input_probs, encoded_target, weight=self.weight, reduction='none')
        loss = (loss*self.vocab_weights).mean() # only account for target variables (target) and hide padding (vocab_weights)
        return loss