from copy import deepcopy
from typing import Dict, List, Tuple

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

from util import nethook
from util.generate import generate_fast
from scipy.linalg import svd

from .compute_existing_u import compute_u, get_inv_cov
from .compute_existing_v import compute_v
from .rome_hparams import ROMEHyperParams
from .rome_main import upd_matrix_match_shape, get_context_templates

CONTEXT_TEMPLATES_CACHE = None

def apply_unlearn_rome_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: ROMEHyperParams,
    copy=False,
    return_orig_weights=False,
    C=None,
    C_1=None,
    return_kstar=False,
) -> Tuple[AutoModelForCausalLM, List[str]]:
    """
    Returns a model with the desired changes.

    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.

    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    if copy:
        model = deepcopy(model)

    weights_copy = {}

    if return_kstar:
        assert len(requests) == 1

    for i, request in enumerate(requests):

        if return_kstar:
            deltas, k_stars, v_stars = execute_rome(model, tok, request, hparams, C, C_1, return_kstar=return_kstar)
        else:
            deltas = execute_rome(model, tok, request, hparams, C, C_1)

        with torch.no_grad():
            for w_name, (delta) in deltas.items():
                # upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
                w = nethook.get_parameter(model, w_name)
                # upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

                if return_orig_weights and w_name not in weights_copy:
                    assert i == 0
                    weights_copy[w_name] = w.detach().clone()

                w[...] += delta

        print(f"New weights successfully inserted into {list(deltas.keys())}")

    if return_kstar:
        return model, weights_copy, k_stars, v_stars
    else:
        return model, weights_copy

def execute_rome(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    Cs: torch.Tensor=None,
    C_1s: torch.Tensor=None,
    return_kstar: bool=False
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the ROME update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    request = deepcopy(request)
    if request["target_new"]["str"][0] != " ":
        # Space required for correct tokenization
        request["target_new"]["str"] = " " + request["target_new"]["str"]
    print(
        f"Executing ROME algorithm for the update: "
        f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
    )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Update loop: sequentially intervene at each specified layer
    deltas = {}
    if return_kstar:
        k_stars = {}
        v_stars = {}

    for layer in sorted(hparams.layers):
        # Compute rank-1 update matrix
        k_star: torch.Tensor = compute_u(
            model,
            tok,
            request,
            hparams,
            layer,
            get_context_templates(model, tok, hparams.context_template_length_params),
        )
        print("k_star shape:", k_star.shape)

        v_star: torch.Tensor = compute_v(
            model,
            tok,
            request,
            hparams,
            layer,
            get_context_templates(model, tok, hparams.context_template_length_params),
        )
        print("v_star shape:", v_star.shape)

        # C_1 = C_1
        if C_1s is None:
            C_1 = get_inv_cov(
                model,
                tok,
                hparams.rewrite_module_tmp.format(layer),
                hparams.mom2_dataset,
                hparams.mom2_n_samples,
                hparams.mom2_dtype,
            )
            C = torch.inverse(C_1) # TODO: directly return C instead of C_1
        elif Cs is None:
            raise NotImplementedError
        else:
            assert Cs is not None
            C = Cs[layer]
            C_1 = torch.inverse(C)

        with torch.no_grad():
            # # Determine correct transposition of delta matrix
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            # upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
            # upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            # # Update model weights and record desired changes in `delta` variable
            # weights[weight_name][...] += upd_matrix
            # deltas[weight_name] = (
            #     left_vector.detach(),
            #     right_vector.detach(),
            # )

            # Determine correct transposition of delta memory
            # v_star * k_star^T * C_1

            k_star = k_star / k_star.norm()

            if return_kstar:
                k_stars[layer] = k_star
                v_stars[layer] = v_star

            # Compressed Implementation:

            new_weights = (weights[weight_name] @ C - v_star.unsqueeze(1) @ k_star.unsqueeze(0)) @ torch.inverse(
                C - k_star.unsqueeze(1) @ k_star.unsqueeze(0)
            )
            delta = new_weights - weights[weight_name]

            # # Expanded Implementation:
            # kkTC_1 = k_star.unsqueeze(1) @ k_star.unsqueeze(0) @ C_1
            # vkT = v_star.unsqueeze(1) @ k_star.unsqueeze(0)
            # kTC_1k = k_star.unsqueeze(0) @ C_1 @ k_star.unsqueeze(1)

            # delta = - vkT @ C_1
            # delta += (
            #     (weights[weight_name].detach() @ kkTC_1 - vkT @ C_1 @ kkTC_1) / (1 - kTC_1k)
            # )

            # Update model weights and record desired changes in `delta` variable
            weights[weight_name][...] += delta
            deltas[weight_name] = delta

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")
    
    if return_kstar:
        return deltas, k_stars, v_stars
    else:
        return deltas
