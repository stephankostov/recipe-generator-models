import os
import json
from typing import NamedTuple
import pickle
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import model
import data
import optimiser
from utils import set_seeds

class Config(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 3431 # random seed
    batch_size: int = 32
    lr: int = 5e-5 # learning rate
    n_epochs: int = 10 # the number of epoch
    # `warm up` period = warmup(0.1)*total_steps
    # linearly increasing learning rate from zero to the specified value(5e-5)
    warmup: float = 0.1
    save_steps: int = 100 # interval for saving model
    total_steps: int = 100000 # total number of steps to train

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))

def main(train_cfg='../config/train.json',
          model_cfg='../config/bert_base.json',
          recipes_file='../../data/local/final/partial/recipe_food_ids/0.npy',
          food_vectors_file='../../data/local/final/full/food_compounds/0.npy'):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_cfg = Config.from_json(train_cfg)
    model_cfg = model.Config.from_json(model_cfg)

    set_seeds(train_cfg.seed)

    food_vectors = np.load(food_vectors_file)
    recipes = np.load(recipes_file)

    # set special token indexes
    special_token_ids = {
      'pad': food_vectors.shape[0]-3, 'unknown': food_vectors.shape[0]-2, 'mask': food_vectors.shape[0]-1
    }
    
    food_vectors = torch.tensor(food_vectors, dtype=torch.float)
    vocab_size = len(food_vectors)
    n_dim = food_vectors.shape[1]

    preprocess_pipeline = [data.MLMDataMasker(
        max_pred=2,
        mask_prob=0.15,
        max_len=15,
        indexer=data.convert_tokens_to_ids,
        token_vocab=range(0,len(food_vectors)-1),
        special_token_ids=special_token_ids
    )]

    dataset = data.MaskedRecipeDataset(recipes, preprocess_pipeline)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=2)

    bert_model = model.BertModel4Pretrain(model_cfg, food_vectors)
    bert_model.to(device)

    loss_func = nn.CrossEntropyLoss(reduction='none')
    adam_optimiser = optimiser.optim4GPU(train_cfg, bert_model).to(device)

    training_metrics = []

    for epoch in range(train_cfg.n_epochs):
        
        for i, batch in enumerate(tqdm(dataloader)):

            batch = next(iter(dataloader))
            batch = [x.to(device) for x in batch]
            input_ids, input_mask, masked_ids, masked_pos, masked_weights = batch

            # print('inputs:', *batch, sep='\n')
            output = bert_model(input_ids, input_mask, masked_pos)

            adam_optimiser.zero_grad()
            loss = loss_func(output.transpose(1, 2), masked_ids)
            loss = (loss*masked_weights.float()).mean()
            loss.backward()
            adam_optimiser.step()
            
            if i % train_cfg.save_steps == 0:
                training_metrics.append({
                    'epoch': epoch, 'step': i, 'loss': loss.item()
                })
    
    with open('train_metrics.pickle', 'wb') as f:
        pickle.dump(training_metrics, f)



if __name__ == '__main__':
    main()