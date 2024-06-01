print("Running app")

from pathlib import Path
import sys
import os
from dotenv import load_dotenv

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
load_dotenv()

import streamlit as st
import pandas as pd
import numpy as np
import torch

import wandb

from recipe_generator.models.gpt import GPTLanguageModel
from recipe_generator.config.gpt import GPTConfig

from recipe_generator.models.transformer import IngredientWeightPredictor
from recipe_generator.config.quantity import IngredientWeightsPredictorCFG

from app.gdrive_api import download_gdrive_folder

print("Imported modules")

# args
device = 'cuda' if torch.cuda.is_available() else 'cpu'
food_embeddings_file='artifacts/food_embeddings.npy'
special_token_embeddings_file='artifacts/special_token_embeddings.npy'
foods_file='artifacts/food_names.npy'
ingredient_model_weights = "stephankostov/recipe-generator-ingredient-v2/model:v9"
quantity_model_weights = "stephankostov/recipe-generator-quantity-test/model:v109"


@st.cache_resource
def wandb_login():
    wandb.login(key=os.environ['WANDB_API_KEY'])
    wandb_api = wandb.Api()
    return wandb_api

@st.cache_resource
def load_model(model, cfg, embedding_weights, model_weights_artifact):
    save_file = Path(f"artifacts/{model.__name__}/model.pt")
    if not save_file.exists(): 
        wandb_api.artifact(model_weights_artifact).download(save_file.parent)
    cfg = cfg()
    model = model(cfg, embedding_weights)
    model.load_state_dict(torch.load(save_file))
    model.to(device).eval()
    return model

@st.cache_data
def load_files():
    embedding_weights = {
        'ingredients': np.load(food_embeddings_file),
        'special_tokens': np.load(special_token_embeddings_file)
    }
    foods = np.load(foods_file)
    foods = np.char.replace(foods, 'cook', 'cooked')
    return foods, embedding_weights

def generate(ingredient_model, quantity_model, token_context, quantity_context):

    ingredient_samples = ingredient_model.generate(token_context, max_new_tokens=14-token_context.size(1))
    quantity_samples = quantity_model.generate(ingredient_samples, quantity_context)

    ingredient_samples = ingredient_samples.detach().cpu()
    quantity_samples = quantity_samples.detach().cpu()

    return ingredient_samples, quantity_samples

def create_output_df(ingredient_samples, quantity_samples):
    results = pd.DataFrame({'ingredient': foods[ingredient_samples.flatten()], 'quantity': (quantity_samples.flatten()*1000).int()})
    return results

def generate_and_output():

    ingredients = st.session_state.ingredients

    ingredient_ids = [3] + ([np.where(foods==ingredient)[0][0] for ingredient in ingredients['ingredient']] if not ingredients.empty else [])
    token_context = torch.tensor(ingredient_ids, dtype=torch.int, device=device).unsqueeze(0)
    quantities = [0.] + (list(ingredients['quantity']/1000) if not ingredients.empty else [])
    quantity_context = torch.tensor(quantities, dtype=torch.float, device=device).unsqueeze(0)
    
    samples = generate(ingredient_model, quantity_model, token_context, quantity_context)
    results = create_output_df(*samples)

    st.session_state.recipe = results

def add_ingredient(ingredient, quantity):
    # ingredient_id = np.where(foods==ingredient)[0][0]
    st.session_state.ingredients.loc[len(st.session_state.ingredients)] = [ingredient, quantity]

def refresh_ingredients():
    st.session_state.ingredients = pd.DataFrame(columns=['ingredient', 'quantity'])

@st.cache_resource
def download_gdrive():
    download_gdrive_folder('recipe-generator', Path('artifacts'))

print("Downloading files")
# downloading data
download_gdrive()
wandb_api = wandb_login()

print("Loading files")
foods, embedding_weights = load_files()
ingredient_model = load_model(GPTLanguageModel, GPTConfig, embedding_weights, ingredient_model_weights)
quantity_model = load_model(IngredientWeightPredictor, IngredientWeightsPredictorCFG, embedding_weights, quantity_model_weights)


print("Initialising streamlit page")
# st.set_page_config(page_title="Recipe Generator", page_icon="🍳")

if "recipe" not in st.session_state:
    st.session_state.recipe = pd.DataFrame()
if "generate" not in st.session_state:
    st.session_state.generate = False
if "add" not in st.session_state:
    st.session_state.add = False
if "ingredients" not in st.session_state:
    st.session_state.ingredients = pd.DataFrame(columns=['ingredient', 'quantity'])
if "add_ingredient" not in st.session_state:
    st.session_state.add_ingredient = False

st.title("Recipe Generator")
st.markdown(
    "A rough demo of the models in action."
)

st.markdown("---")
st.markdown(
    "Enter in the ingredients to include: (optional) "
)

cols = st.columns(2)
with cols[0]: ingredient = st.selectbox("Ingredient", foods[5:])
with cols[1]: quantity = st.number_input("Weight (grams)", 0, 1000)
st.button("Add", on_click=add_ingredient, args=(ingredient, quantity))
st.markdown("")
if not st.session_state.ingredients.empty: 
    st.table(st.session_state.ingredients)
    st.button("Refresh", on_click=refresh_ingredients)

st.markdown("---")
st.button(
    label='Generate',
    type="primary",
    on_click=generate_and_output
)

text_spinner_placeholder = st.empty()

if not st.session_state.recipe.empty:
    st.markdown("""---""")
    st.table(st.session_state.recipe)