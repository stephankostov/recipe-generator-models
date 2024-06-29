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

from recipe_generator.models.ingredient import IngredientModel
from recipe_generator.models.quantity import QuantityModel
from recipe_generator.models.samplers import beam_search
from recipe_generator.utils import load_config

def load_model(model, embedding_weights, model_weights_artifact, model_config_artifact):
    save_dir = Path(f"artifacts/{model.__name__}")
    if not save_dir.exists():
        wandb_api.artifact(model_weights_artifact).download(save_dir)
        wandb_api.artifact(model_config_artifact).download(save_dir)
    cfg = load_config(next(save_dir.glob('*.yaml')))
    model = model(cfg, embedding_weights)
    model.load_state_dict(torch.load(save_dir/'model.pt'))
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
food_embeddings_file='artifacts/food_embeddings.npy'
special_token_embeddings_file='artifacts/special_token_embeddings.npy'
foods_file='artifacts/food_names.npy'
ingredient_model_weights = "stephankostov/recipe-generator-ingredient-v2/model:v28"
ingredient_model_config = "stephankostov/recipe-generator-ingredient-v2/model_cfg:v0"
quantity_model_weights = "stephankostov/recipe-generator-quantity-v1/model:v9"
quantity_model_config = "stephankostov/recipe-generator-quantity-v1/model_cfg:v0"


# loading arrays
embedding_weights = {
    'ingredients': torch.tensor(np.load(food_embeddings_file), dtype=torch.float), 
    'special_tokens': torch.tensor(np.load(special_token_embeddings_file), dtype=torch.float)
}
foods = np.load(foods_file)

# loading models
ingredient_model = load_model(IngredientModel, embedding_weights, ingredient_model_weights, ingredient_model_config).to(device).eval()
quantity_model = load_model(QuantityModel, embedding_weights, quantity_model_weights, quantity_model_config).to(device).eval()

initial_recipe = torch.ones((5,1), dtype=torch.int, device=device)*3
samples, _ = beam_search(ingredient_model, initial_recipe, 14)
samples = samples.detach().cpu()
print(samples)
print(foods[samples])
# ingredient_samples = ingredient_model.generate(initial_recipe, max_new_tokens=14).detach().cpu()

# samples = generate(ingredient_model, quantity_model)
# results = create_output_df(*samples)
# print(results.to_markdown())
