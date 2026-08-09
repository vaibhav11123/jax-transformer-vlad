from dataclasses import dataclass
from typing import Literal

import data

@dataclass
class Config:
    seed: int=1331
    # --task: digit addition--
    n_digits: int=3
    vocab_size: int=len(data.vocab)
    # --model--
    n_embd: int=384
    n_head: int=6
    n_layer: int=6
    dropout: float=0.1
    #--optimization--
    batch_size: int=256
    learning_rate: float=3e-4
    max_iters: int=5000
    eval_interval: int=200
    eval_iters: int=50
    #--logging--
    wandb_entity: str="vaibhav23-iit-delhi"
    wandb_project: str="jax-transformer-vlad"
    wandb_run_name: str="digit-addition-10m"
    wandb_mode: Literal["online", "offline", "disabled", "shared"]="online"
    @property 
    def block_size(self) -> int:
        return 3*self.n_digits+2
