from typing import NamedTuple
from recipe_generator.utils.utils import debugger_is_active

class TrainConfig(NamedTuple):
    
    seed: int = 1337
    batch_size: int = 1000
    lr: int = 1e-3
    n_epochs: int = 10
    save_steps: int = 50
    max_steps: int = 1e9
    device: str = 'cuda'
    wandb: bool = True and not debugger_is_active()
