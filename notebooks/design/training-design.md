# Training Objective

How do we train these models? What makes sense is doing this in a self-supervised manner. Looking into two very popular approaches, GPT and BERT. The former uses a next-token generation technique of using a sequence $x_{0:n}$ as input and $x_{1:(n+1)}$ as target, trained with a causal attention mask to hide the future tokens at each position. BERT on the other hand uses token masking, masking a proportion of the tokens of a sequence at random, and training the model to predict these tokens.

## Research

- "It has been argued that the pre-training objective used by BERT is not well suited for tasks that require decoding texts, e.g., conditional text generation in machine translation and summarization ([Yang et al., 2019](https://arxiv.org/abs/1906.08237))"

# Loss Function

## Permutation Invariance 

The position of ingredients in a given recipe does not have any significant meaning, and therefor we would like our model to be able to ignore them. These probems in the literature are known as permutation invariance. We would like to account for that in the training process, as it plays an important role in learning. The standard loss function of a token generation is the cross entropy loss of the probabilities of each token at each position. If for a position $i$, a model predicts a certain ingredient $y_i$ in the incorrect position, it will be penalised as if it had not been able to predict it whatsoever. 

### Solutions

- Loss Function: Position Invariant Loss Function
- Data Augmentation: Shuffling recipes
- Attention Mechanism: Transformers have an inherent permutation invariant property - does this apply to our case of training however?
- Architecture: Set Transformer [https://arxiv.org/abs/1810.00825], [https://arxiv.org/abs/1703.06114]

The loss function is the clever architectural form of doing this, whereas the data augmentation is the brute force method of just training it on more combinations of data.

As it so happens, transformers have been shown to have the permutation invariance property naturally emergent.

### Loss Function

Changing the loss function, whilst most ideal, comes with a number of complexities. Fundamenetally, we would ideally like our function to be convex, and also continuous and differentiable. One simple idea that comes to mind is performing a max pooling: for each token in the target sequence, we would get the cross entropy loss against the highest probability value in the output sequence. However, doing cuts like this I'd imagine would lead to a non-differentiable loss function #todo investigate this.

There are other loss functions available in the literature that we can try, one such is [https://arxiv.org/abs/2106.15078].