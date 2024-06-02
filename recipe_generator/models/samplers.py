# https://github.com/jarobyte91/pytorch_beam_search/blob/master/src/pytorch_beam_search
import torch
from tqdm import tqdm
import torch.utils.data as tud

def beam_search(
    model, 
    X, 
    predictions = 20,
    beam_width = 5,
    batch_size = 128, 
    progress_bar = 0
):
    """
    Implements Beam Search to extend the sequences given in X. The method can compute 
    several outputs in parallel with the first dimension of X.

    Parameters
    ----------    
    X: LongTensor of shape (examples, length)
        The sequences to start the decoding process.

    predictions: int
        The number of tokens to append to X.

    beam_width: int
        The number of candidates to keep in the search.

    batch_size: int
        The batch size of the inner loop of the method, which relies on the beam width. 

    progress_bar: bool
        Shows a tqdm progress bar, useful for tracking progress with large tensors.

    Returns
    -------
    X: LongTensor of shape (examples, length + predictions)
        The sequences extended with the decoding process.

    probabilities: FloatTensor of length examples
        The estimated log-probabilities for the output sequences. They are computed by iteratively adding the 
        probability of the next token at every step.
    """
    with torch.no_grad():
        # The next command can be a memory bottleneck, but can be controlled with the batch 
        # size of the predict method.
        next_probabilities = model.forward(X)[:, -1, :]
        vocabulary_size = next_probabilities.shape[-1]
        probabilities, idx = next_probabilities.squeeze().log_softmax(-1)\
            .topk(k = beam_width, axis = -1)
        X = X.repeat((beam_width, 1, 1)).transpose(0, 1)\
            .flatten(end_dim = -2)
        next_chars = idx.reshape(-1, 1)
        X = torch.cat((X, next_chars), axis = -1)
        # This has to be minus one because we already produced a round
        # of predictions before the for loop.
        predictions_iterator = range(predictions - 1)
        if progress_bar > 0:
            predictions_iterator = tqdm(predictions_iterator)
        for i in predictions_iterator:
            dataset = tud.TensorDataset(X)
            loader = tud.DataLoader(dataset, batch_size = batch_size)
            next_probabilities = []
            iterator = iter(loader)
            if progress_bar > 1:
                iterator = tqdm(iterator)
            for (x,) in iterator:
                next_probabilities.append(
                    model.forward(x)[:, -1, :].log_softmax(-1)
                )
            next_probabilities = torch.cat(next_probabilities, axis = 0)
            next_probabilities = next_probabilities.reshape(
                (-1, beam_width, next_probabilities.shape[-1])
            )
            probabilities = probabilities.unsqueeze(-1) + next_probabilities
            probabilities = probabilities.flatten(start_dim = 1)
            probabilities, idx = probabilities.topk(
                k = beam_width, 
                axis = -1
            )
            next_chars = torch.remainder(idx, vocabulary_size).flatten()\
                .unsqueeze(-1)
            best_candidates = (idx / vocabulary_size).long()
            best_candidates += torch.arange(
                X.shape[0] // beam_width, 
                device = X.device
            ).unsqueeze(-1) * beam_width
            X = X[best_candidates].flatten(end_dim = -2)
            X = torch.cat((X, next_chars), axis = 1)
        return X.reshape(-1, beam_width, X.shape[-1]), probabilities