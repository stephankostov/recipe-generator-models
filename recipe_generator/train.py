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

def main(train_cfg='./config/train.json',
          model_cfg='./config/bert_base.json',
          recipes_file='../data/local/final/full/recipe_food_ids/0.npy',
          food_vectors_file='../data/local/final/full/food_compounds/0.npy'):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_cfg = Config.from_json(train_cfg)
    model_cfg = model.Config.from_json(model_cfg)

    set_seeds(train_cfg.seed)

    food_vectors = np.load(food_vectors_file)
    recipes = np.load(recipes_file)

    print("Loaded data", food_vectors.shape, recipes.shape)

    # set special token indexes
    special_token_ids = ['pad','mask','unknown']
    
    food_vectors = torch.tensor(food_vectors, dtype=torch.float)

    preprocess_pipeline = [data.MLMDataMasker(
        max_pred=1,
        mask_prob=0.15,
        max_len=model_cfg.max_len,
        indexer=data.convert_tokens_to_ids,
        token_vocab=range(0,len(food_vectors)-1),
        special_token_ids=special_token_ids
    )]

    cv_ratio = 0.8
    np.random.shuffle(recipes)
    data_train, data_validation = recipes[:int(cv_ratio*len(recipes))], recipes[int(cv_ratio*len(recipes)):]
    train_ds, validation_dl = data.MaskedRecipeDataset(data_train, preprocess_pipeline), data.MaskedRecipeDataset(data_validation, preprocess_pipeline), 
    train_dl, validation_dl = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=2), DataLoader(validation_dl, batch_size=train_cfg.batch_size, shuffle=False, num_workers=2)

    print(len(train_dl))

    bert_model = model.BertModel4Pretrain(model_cfg, food_vectors)
    bert_model.to(device)

    print(sum(p.numel() for p in bert_model.parameters())/1e6, 'M parameters')

    loss_func = nn.CrossEntropyLoss(reduction='none')
    adam_optimiser = optimiser.optim4GPU(train_cfg, bert_model)

    training_metrics = []
    total_step = 0

    for epoch in range(train_cfg.n_epochs):
        
        for i, batch in enumerate(tqdm(train_dl)):

            # loading data onto device
            batch = [x.to(device) for x in batch]
            input_ids, input_mask, masked_ids, masked_pos, masked_weights = batch
            # print('inputs:', *[(b, b.shape) for b in batch], sep='\n')

            # training
            bert_model.train()
            output = bert_model(input_ids, input_mask, masked_pos)

            adam_optimiser.zero_grad()
            loss = loss_func(output.transpose(1, 2), masked_ids)
            loss = (loss*masked_weights.float()).mean()
            loss.backward()
            adam_optimiser.step()
            
            total_step += 1

            # evaluation
            if i % train_cfg.save_steps == 0 or total_step >= train_cfg.total_steps:

                with torch.no_grad():

                    bert_model.eval()
                    batch = next(iter(validation_dl))
                    batch = [x.to(device) for x in batch]

                    bert_model.eval()
                    input_ids, input_mask, masked_ids, masked_pos, masked_weights = batch

                    output = bert_model(input_ids, input_mask, masked_pos)
                    validation_loss = loss_func(output.transpose(1, 2), masked_ids)
                    validation_loss = (loss*masked_weights.float()).mean()
                    training_metrics.append({
                        'epoch': epoch, 'total_step': total_step, 'train_loss': loss.item(), 'validation_loss': validation_loss.item()
                    })
    
                if total_step >= train_cfg.total_steps: return

    
    with open('./outputs/train_metrics.pickle', 'wb') as f:
        pickle.dump(training_metrics, f)



if __name__ == '__main__':
    main()