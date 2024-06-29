Repository for the [recipe-generator](https://github.com/stephankostov/recipe-generator) models and training.

# Development Process

Notebooks used for initial exploration and design planning. Code written as standard Python modules using PyTorch.

# Design

See [./notebooks/design/](./notebooks/design/)

# Key Features

- Implementation of model architectures [./recipe_generator/models/](./recipe_generator/models/)
- Model training [./recipe_generator/train/](./recipe_generator/train/)
    - Experiment tracking with wandb ([ingredients](https://wandb.ai/stephankostov/recipe-generator-quantity-public/workspace), [quantities](https://wandb.ai/stephankostov/recipe-generator-quantity-public/workspace))
- Frontend application ([./app/](./app/))([see demo](https://molecular-recipe-generator.streamlit.app/))

# ToDo

- Model Interpretability
    - The advantage of using this molecular composition data as our vectorised tokens is having the ability to interpret what is happening in the models. We could disect the attention mechanism to visualise what is giving high key-query weights, and what the values are choosing as important. Here we could try 3blue1brown's mathematical visualisation python library [manim](https://github.com/3b1b/manim).
- Ablation tests 
    - Performance test using standard embedding model.
- Integrated model
    - Design model that can incorporate two inputs/outputs ie. token id's and quantites.
- Mathematical formulation to investigate theoretical limits of model.
    - Do Pareto distributed embeddings violate the model's assumptions?
    - Does pooled loss function effect convexity & differentiability?
    - ...
- Intricate Hyperparameter Tuning
    - Find optimal hyperparameters using Bayesian/Grid Search.
    - Find optimal model size using pruning techniques.
