from dataclasses import dataclass
from typing import List, Literal

from util.hparams import HyperParams


@dataclass
class MEMITHyperParams(HyperParams):
    # Method
    layers: List[int]
    layer_selection: Literal["all", "random"]
    fact_token: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last"
    ]
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    mom2_update_weight: float

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str

    model: str = None
    ds_name: str = None

    learning_rate: float = None
    noise_scale: float = None
    scheduler: dict = None
    total_iters: int = 200

    prune_requests: int = 0
    beta: float = None
    alpha: float = None
    betas: dict = None
    alphas: dict = None
    beta_ratio: float = 1.0
    compute_initialization_at_once: bool = False
    lr_after_hitting_boundary: float = None
    decay_factor: float = 0.8
    skip_search: bool = False
    keep_original_requests: bool = False
    random_initialize: bool = False
    simply_combine_objective: bool = False
    
