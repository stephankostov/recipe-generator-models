# Food Tokens

## Permutation Invariance 

The position of ingredients in a given recipe does not have any significant meaning. These probems in the literature are known as permutation invariance. We would like to account for that in the training process, as it plays an important role in learning. The standard loss function of a token generation is the cross entropy loss of the probabilities of each token at each position. If for a position $i$, a model predicts a certain ingredient $y_i$ in the incorrect position, it will be penalised as if it had not been able to predict it whatsoever. 

(wrong: The model does not predict off of its own generated context. At each position it makes prediction given the (masked) label that has been inputted to it.) On top of this, whatever the label ingredient at that position, $\hat{y}_i$ is, since the model is using context of its own predictions (wrong), when it does predict this value at a different position the value will not be given any reward either.

Solutions:

- Loss Function: Position invariant
    - Max pooling - compare the cross entropy of the maximum probabilities in the sequence.
    - Other loss invariant loss functions.
- Data Augmentation: Shuffling recipes
- Attention Mechanism

The loss function is the clever architectural form of doing this, whereas the data augmentation is the brute force method of just training it on more combinations of data.

As it so happens, transformers have been shown to have the permutation invariance property naturally emergent.

### Positional Embeddings

On this note, do we remove the positional embeddings? Yes, as all these do is encode information about the position of the tokens, which in our case is irrelevant. 

Point of confusion: doesn't the prediction require the positions to be known in order to make predictions at the correct position? No, this is not how this works. At each position, the decoder makes predictions depending on what it has seen before it, through masking.