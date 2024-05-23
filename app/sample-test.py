from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

import streamlit as st
import pandas as pd
import numpy as np
import torch

import wandb
wandb_api = wandb.Api()

from recipe_generator.models.gpt import GPTLanguageModel
from recipe_generator.config.gpt import GPTConfig

from recipe_generator.models.transformer import IngredientWeightPredictor
from recipe_generator.config.quantity import IngredientWeightsPredictorCFG

def load_model(model, cfg, embedding_weights, model_weights_artifact):
    save_file = Path(f"artifacts/{model.__name__}/model.pt")
    if not save_file.exists(): 
        wandb_api.artifact(model_weights_artifact).download(save_file.parent)
    cfg = cfg()
    model = model(cfg, embedding_weights)
    model.load_state_dict(torch.load(save_file))
    return model

def generate(ingredient_model, quantity_model, initial_recipe=None):
    if not initial_recipe: initial_recipe = torch.ones((5,1), dtype=torch.int, device=device)*3
    ingredient_samples = ingredient_model.generate(initial_recipe, max_new_tokens=14).detach().cpu()
    quantity_samples = quantity_model.generate(ingredient_samples.to('cuda')).detach().cpu()
    print(ingredient_samples.shape, quantity_samples.shape)
    return ingredient_samples, quantity_samples

def create_output_df(ingredient_samples, quantity_samples):
    indices = [(i, j) for i in range(5) for j in range(15)]
    index = pd.MultiIndex.from_tuples(indices)
    results = pd.DataFrame({'ingredient': foods[ingredient_samples.flatten()], 'quantity': (quantity_samples.flatten()*1000).int()}, index=index)
    return results

def generate_and_output():
    samples = generate(ingredient_model, quantity_model)
    results = create_output_df(*samples)
    st.session_state.recipe = results


# args
device = 'cuda' if torch.cuda.is_available() else 'cpu'
food_embeddings_file='../data/local/final/full/food_embeddings/0.npy'
special_token_embeddings_file='../data/local/final/full/special_token_embeddings/0.npy'
foods_file='../data/local/final/full/food_names/0.npy'
ingredient_model_weights = "stephankostov/recipe-generator-ingredient/model:v23"
quantity_model_weights = "stephankostov/recipe-generator-quantity-test/model:v89"

# loading arrays
embedding_weights = {
    'ingredients': torch.tensor(np.load(food_embeddings_file), dtype=torch.float), 
    'special_tokens': torch.tensor(np.load(special_token_embeddings_file), dtype=torch.float)
}
foods = np.load(foods_file)

# loading models
ingredient_model = load_model(GPTLanguageModel, GPTConfig, embedding_weights, ingredient_model_weights).to(device).eval()
quantity_model = load_model(IngredientWeightPredictor, IngredientWeightsPredictorCFG, embedding_weights, quantity_model_weights).to(device).eval()

samples = generate(ingredient_model, quantity_model)
results = create_output_df(*samples)
print(results.to_markdown())
