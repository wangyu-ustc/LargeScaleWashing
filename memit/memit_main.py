import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPTJForCausalLM

from rome.layer_stats import layer_stats
from util import nethook, get_predictions
from util.generate import generate_fast
from util.globals import *

from .compute_ks import compute_ks, compute_zs
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .memit_hparams import MEMITHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}


def apply_memit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas = execute_memit(model, tok, requests, hparams, cache_template=cache_template)

    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items():
            key_mat, val_mat = key_mat.to("cuda"), val_mat.to("cuda")
            upd_matrix = key_mat @ val_mat.T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix.float()

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def apply_law_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    ds_name=None,
    copy=False,
    edit_to=None,
    alg_name=None,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas = execute_law(model, tok, requests, hparams, ds_name=ds_name, cache_template=cache_template, edit_to=edit_to, alg_name=alg_name)

    with torch.no_grad():
        if alg_name == 'LAW': 
            for w_name, (upd_matrix) in deltas.items():
                w = nethook.get_parameter(model, w_name)
                upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

                if return_orig_weights and w_name not in weights_copy:
                    weights_copy[w_name] = w.detach().clone()

                w[...] += upd_matrix.float().to(w.device)
            
        else:
            for w_name, (key_mat, val_mat) in deltas.items():
                key_mat, val_mat = key_mat.to("cuda"), val_mat.to("cuda")
                upd_matrix = key_mat @ val_mat.T
                w = nethook.get_parameter(model, w_name)
                upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

                if return_orig_weights and w_name not in weights_copy:
                    weights_copy[w_name] = w.detach().clone()

                w[...] += upd_matrix.float()

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_memit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests[:10]:
        print(
            f"MEMIT request sample: "
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

    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:

            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )

            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    
    zs = torch.stack(z_list, dim=1)

    # Insert
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        # Compute residual error
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T

        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)

        # Load covariance matrix
        force_recompute = False
        # force_recompute = layer != hparams.layers[0]
        cov = get_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples
            if not force_recompute
            else hparams.mom2_n_samples // 10,
            hparams.mom2_dtype,
            force_recompute=force_recompute,
        )

        # Compute update in double precision
        layer_ks, targets = (
            layer_ks.double(),
            targets.double(),
        )

        adj_k = torch.linalg.solve(
            hparams.mom2_update_weight * cov.double() + layer_ks @ layer_ks.T,
            layer_ks,
        )

        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
        upd_matrix = resid @ adj_k.T

        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))

        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
            deltas[weight_name] = (
                adj_k.detach().cpu(),
                resid.detach().cpu(),
            )

        # Clear GPU memory
        cov.cpu()
        for x in [layer_ks, cur_zs, targets]:
            x.cpu()
            del x
        torch.cuda.empty_cache()

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas

def cosine_lr_scheduler(start_lr, end_lr, current_step, total_steps):
    """
    A simple cosine learning rate scheduler.
    
    :param start_lr: Initial learning rate
    :param end_lr: Final learning rate
    :param current_step: Current step during the training process
    :param total_steps: Total number of steps expected during training
    :return: Adjusted learning rate for the current step
    """
    cosine_decay = 0.5 * (1 + math.cos(math.pi * current_step / total_steps))
    lr = end_lr + (start_lr - end_lr) * cosine_decay
    return lr


def proj(delta, udv, beta, device="cuda", n_iters=1000, lr=0.01, verbose=True, zero_init=False):
    #delta 1000 x 4000 (tensor)
    #cov (K^TK) 4000 x 4000 (tensor)
    #beta (float)
    #OUTPUT: project delta to delta', s.t. ||delta'K||_2^2\leq \beta
    delta = delta.to(device)
    U_t, D, V_t = udv
    U_t, D, V_t = U_t.to(device), D.to(device), V_t.to(device)

    D = torch.diag(D)
    D_inv_sqrt_vec = torch.sqrt(1 / D.diag())
#     print((U_t - V_t.t()).norm(), D, D_inv)
#     assert((U_t - V_t.t()).norm() <1e-5)
    
    if zero_init:
        opt_delta_u_t_sqrt_d = torch.zeros(delta.shape).to(device)
    else:
        opt_delta_u_t_sqrt_d = delta.matmul(U_t).matmul(torch.sqrt(D))
    norm = (opt_delta_u_t_sqrt_d ** 2).sum()
    if norm > beta:
        opt_delta_u_t_sqrt_d.data = opt_delta_u_t_sqrt_d.data / math.sqrt(norm.item()) * math.sqrt(beta)
    
    opt_delta_u_t_sqrt_d.requires_grad = True
    
    delta_mat_u_t = delta.matmul(U_t)
    
    optimizer = torch.optim.SGD([opt_delta_u_t_sqrt_d], lr=lr)
    for i in range(n_iters):
        optimizer.zero_grad()
        loss = ((opt_delta_u_t_sqrt_d * D_inv_sqrt_vec- delta_mat_u_t) ** 2).sum()
        loss.backward()
        optimizer.step()
        norm = (opt_delta_u_t_sqrt_d ** 2).sum()
        if norm > beta:
            opt_delta_u_t_sqrt_d.data = opt_delta_u_t_sqrt_d.data / math.sqrt(norm.item()) * math.sqrt(beta)
            norm = (opt_delta_u_t_sqrt_d ** 2).sum()
#             print(norm, cur_norm, beta)
#             break
        if verbose:
            if i%100 == 0:
                print(i, loss.item(), norm.item())
#     print(opt_delta_u_t_sqrt_d.shape, D_inv.shape, U_t.t().shape)
    proj_delta = (opt_delta_u_t_sqrt_d * D_inv_sqrt_vec).matmul(U_t.t())
    return proj_delta

def update_upd_matrix(
    mom2_update_weight,
    cov,
    layer_ks,
    learning_rate,
    upd_matrix,
    threshold=None,
    scheduler=None,
    total_iters=200,
    beta=None,
    alpha=None,
    projection=False,
    verbose=False,
    lr_after_hitting_boundary=None,
    interval=100,
    loss_type='l2',
    sub_sample_rate=1,
):
    cov = cov.cuda()
    upd_matrix = upd_matrix.cuda()
    layer_ks = layer_ks.cuda()

    # def loss_function(upd_matrix):
    #     return mom2_update_weight * torch.sqrt(torch.trace(upd_matrix.T @ cov @ upd_matrix)) - torch.linalg.norm(upd_matrix.T @ layer_ks)

    # gradient = mom2_update_weight * cov - layer_ks @ layer_ks.T
    # hessian = 

    # if scheduler is not None:
    #     gradient = mom2_update_weight * cov - layer_ks @ layer_ks.T
    #     for idx in range(200):
    #         # lr = cosine_lr_scheduler(scheduler['start_lr'], scheduler['end_lr'], idx, scheduler['total_steps'])
    #         # upd_matrix -= lr * ((gradient).T @ upd_matrix)
    #         # if threshold is not None and torch.linalg.norm(upd_matrix) > threshold:
    #         #     upd_matrix = threshold * upd_matrix / torch.linalg.norm(upd_matrix)
            
    #         pre_loss = loss_function(upd_matrix)
    #         for lr in range(1, 1001, 10):
    #             lr = lr * 1e-9
    #             new_upd_matrix = upd_matrix - lr * ((gradient).T @ upd_matrix)
    #             loss = loss_function(new_upd_matrix)
    #             if loss > pre_loss:
    #                 break
    #             else:
    #                 pre_loss = loss

    #         upd_matrix = new_upd_matrix

    #         if idx % 10 == 0:
    #             loss = loss_function(upd_matrix)
    #             print(f"Iteration {idx}, upd_matrix norm: {torch.linalg.norm(upd_matrix)}, loss: {loss.item()}")

    #             import ipdb; ipdb.set_trace()

    # else:
    upd_matrix.requires_grad = True
    optimizer = torch.optim.Adam([upd_matrix], lr=learning_rate)

    if loss_type == 'l1':
        original_K_star_loss = torch.sum(torch.abs(upd_matrix.T @ layer_ks))
    else:
        original_K_star_loss = torch.linalg.norm(upd_matrix.T @ layer_ks) ** 2
    orginal_K_loss = torch.trace(upd_matrix.T @ cov @ upd_matrix) * mom2_update_weight
    # original_loss = orginal_K_loss - original_K_star_loss

    print("Original K* loss:", original_K_star_loss)

    end_training = False
    learning_rate_updated = False

    if projection:
        U_t, D, V_t = torch.linalg.svd(cov)

    best_upd_matrix = None
    best_loss = 0

    for idx in range(total_iters):
        optimizer.zero_grad()
        # loss = mom2_update_weight * torch.trace(upd_matrix.T @ cov @ upd_matrix) - torch.linalg.norm(upd_matrix.T @ layer_ks) ** 2
        # loss = - torch.linalg.norm(upd_matrix.T @ layer_ks) ** 2
        # loss = torch.sum((upd_matrix.T @ layer_ks) ** 2)

        if sub_sample_rate < 1:
            randperm = torch.randperm(layer_ks.shape[1])
            indices = randperm[:int(len(randperm) * sub_sample_rate)]
            cur_layer_ks = layer_ks[:, indices]

        else:
            cur_layer_ks = layer_ks

        if loss_type == 'l1':
            loss = - torch.sum(torch.abs(upd_matrix.T @ cur_layer_ks))
        else:
            loss = - torch.sum((upd_matrix.T @ cur_layer_ks) ** 2)
            # loss = - torch.linalg.norm(upd_matrix.T @ layer_ks) ** 2
        loss.backward()
        optimizer.step()

        # if threshold is not None and torch.linalg.norm(upd_matrix) > threshold:
        #     upd_matrix = threshold * upd_matrix / torch.linalg.norm(upd_matrix)
        #     break
        if beta is not None:
            ratio = torch.trace(upd_matrix.T @ cov @ upd_matrix) / torch.trace(cov)
            if ratio > beta:

                if projection:
                    upd_matrix.data = proj(delta=upd_matrix.T.detach(), udv=(U_t, D, V_t), beta=beta * torch.trace(cov), lr=1e-6, verbose=verbose, n_iters=3000).detach().T
                else:
                    upd_matrix.data = (upd_matrix / torch.sqrt(ratio / beta)).detach()

                if lr_after_hitting_boundary is not None and not learning_rate_updated:
                    for g in optimizer.param_groups:
                        g['lr'] = lr_after_hitting_boundary
                    learning_rate_updated = True
            
            # torch.trace(x1.T @ cov @ x1) / torch.trace(cov)
            # torch.linalg.norm(x1.T @ layer_ks) ** 2

        if alpha is not None:
            ratio = torch.linalg.norm(upd_matrix)
            if ratio > alpha:
                upd_matrix.data = (upd_matrix / (ratio / alpha)).detach()

        if - loss.item() > best_loss:
            best_loss = - loss.item()
            best_upd_matrix = upd_matrix.detach()

        if idx % interval == 0:
            print(f"Iteration {idx}, loss: {loss.item()}, upd_matrix norm: {torch.linalg.norm(upd_matrix)}, ")
            # new_k_star_loss = (torch.linalg.norm(upd_matrix.T @ layer_ks) ** 2).item()
            if loss_type == 'l1':
                new_k_star_loss = torch.sum(torch.abs(upd_matrix.T @ layer_ks)).item()
            else:
                new_k_star_loss = torch.linalg.norm(upd_matrix.T @ layer_ks) ** 2

            # if new_k_star_loss > best_loss:
            #     best_loss = new_k_star_loss
            #     best_upd_matrix = upd_matrix.detach()

            if sub_sample_rate < 1:
                if loss_type == 'l1':
                    cur_new_k_star_loss = torch.sum(torch.abs(upd_matrix.T @ cur_layer_ks)).item()
                else:
                    cur_new_k_star_loss = torch.linalg.norm(upd_matrix.T @ cur_layer_ks) ** 2
                print("Sub sampled Loss K*", cur_new_k_star_loss)
            
            print("Loss for K*", new_k_star_loss)
            print("Loss for K", torch.trace(upd_matrix.T @ cov @ upd_matrix).item() * mom2_update_weight)
            print("Current Beta: ", (torch.trace(upd_matrix.T @ cov @ upd_matrix) / torch.trace(cov)).item())

        # DEBUG
        #     if new_k_star_loss < original_K_star_loss:
        #         end_training = True
        #         break 

        # if end_training:
        #     break

    # gradient = mom2_update_weight * cov - layer_ks @ layer_ks.T
    # for _ in range(200):
    #     upd_matrix -= (gradient * learning_rate).T @ upd_matrix
    #     if threshold is not None and torch.linalg.norm(upd_matrix) > threshold:
    #         upd_matrix = threshold * upd_matrix / torch.linalg.norm(upd_matrix)
    #         # break

    new_K_star_loss = torch.linalg.norm(upd_matrix.T @ layer_ks) ** 2
    new_K_loss = torch.trace(upd_matrix.T @ cov @ upd_matrix) * mom2_update_weight

    # return best_upd_matrix.cpu() if best_upd_matrix is not None else upd_matrix, not end_training, original_K_star_loss.item(), orginal_K_loss.item(), new_K_star_loss.item(), new_K_loss.item()
    return best_upd_matrix.cpu() if best_upd_matrix is not None else upd_matrix

def update_upd_matrix_with_lambda(
    mom2_update_weight,
    cov,
    layer_ks,
    learning_rate,
    upd_matrix,
    threshold=None,
    scheduler=None,
    total_iters=200,
    beta=None,
    alpha=None,
    projection=False,
    verbose=False,
    lr_after_hitting_boundary=None,
    interval=1,
    loss_type='l2',
    sub_sample_rate=1,
):
    cov = cov.cuda()
    upd_matrix = upd_matrix.cuda()
    layer_ks = layer_ks.cuda()

    upd_matrix.requires_grad = True
    optimizer = torch.optim.Adam([upd_matrix], lr=learning_rate)

    original_K_star_loss = torch.linalg.norm(upd_matrix.T @ layer_ks)

    print("Original K* loss:", original_K_star_loss)
    print("Original K loss:", torch.trace(upd_matrix.T @ cov @ upd_matrix))
    print("Original loss:", mom2_update_weight * torch.trace(upd_matrix.T @ cov @ upd_matrix) - torch.linalg.norm(upd_matrix.T @ layer_ks) ** 2)

    end_training = False

    for idx in range(total_iters):
        
        optimizer.zero_grad()

        loss = mom2_update_weight * torch.trace(upd_matrix.T @ cov @ upd_matrix) - torch.sum((upd_matrix.T @ layer_ks) ** 2)

        loss.backward()

        optimizer.step()

        new_k_star_loss = torch.linalg.norm(upd_matrix.T @ layer_ks)

        if new_k_star_loss < original_K_star_loss:
            end_training = True
            break

        print(f"Iteration {idx}, loss: {loss.item()}, upd_matrix norm: {torch.linalg.norm(upd_matrix)}")
    
    return upd_matrix.cpu(), not end_training
        


def get_zs(requests, hparams, cache_template, model, tok, context_templates):
    # Compute z for final layer
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:

            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )   

            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    
    zs = torch.stack(z_list, dim=1)
    return zs


def execute_law(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    ds_name: str,
    cache_template: Optional[str] = None,
    edit_to: Optional[str] = None,
    alg_name: str = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        # TODO: This doesn't look right for llama-based models
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests[:10]:
        if 'subject' in request:
            print(
                f"MEMIT request sample: "
                f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
            )
        else:
            print(
                f"MEMIT request sample: "
                f"[{request['prompt']}] -> [{request['target_new']['str']}]"
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

    context_templates = get_context_templates(model, tok)

    # # just for debug:
    # context_templates = context_templates[:1]

    if not alg_name == 'LAW':

        # Compute z for final layer
        z_layer = hparams.layers[-1]
        z_list = []

        for request in requests:
            # Retrieve k/v pair if already stored in cache
            cache_fname = (
                Path(
                    str(cache_template).format(
                        z_layer, hparams.clamp_norm_factor, request["case_id"]
                    )
                )
                if cache_template is not None
                else None
            )
            data_loaded = False
            if (
                cache_fname is not None  # Require cache template
                and cache_fname.exists()  # Cache file must exist
            ):
                try:
                    data = np.load(cache_fname)
                    z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                    data_loaded = True
                except Exception as e:
                    print(f"Error reading cache file due to {e}. Recomputing...")

            # Compute k/v pair if not loaded from cache
            if not data_loaded:

                cur_z = compute_z(
                    model,
                    tok,
                    request,
                    hparams,
                    z_layer,
                    context_templates,
                )   

                z_list.append(cur_z)

                if cache_fname is not None:
                    cache_fname.parent.mkdir(exist_ok=True, parents=True)
                    np.savez(
                        cache_fname,
                        **{
                            "v_star": cur_z.detach().cpu().numpy(),
                        },
                    )
                    print(f"Cached k/v pair at {cache_fname}")
        
        zs = torch.stack(z_list, dim=1)

        # zs is of shape (1600, 3680)
        # Insert
        for i, layer in enumerate(hparams.layers):
            
            print(f"\n\nLAYER {layer}\n")

            

            if isinstance(model, GPTJForCausalLM):
                
                ks_file = f"./data/gpt-j-{ds_name}-layer-{layer}-ks.pt"
                cur_zs_file = f"./data/gpt-j-{ds_name}-layer-{layer}-cur_zs.pt"
                if not os.path.exists(ks_file):
                    layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
                    torch.save(layer_ks, ks_file)
                    
                else:
                    layer_ks = torch.load(ks_file)
                
                if not os.path.exists(cur_zs_file):
                    cur_zs = get_module_input_output_at_words(
                        model,
                        tok,
                        z_layer,
                        context_templates=[request["prompt"] for request in requests],
                        words=[request["subject"] for request in requests],
                        module_template=hparams.layer_module_tmp,
                        fact_token_strategy=hparams.fact_token,
                        track='out'
                    ).T
                    torch.save(cur_zs, cur_zs_file)
                
                else:
                    cur_zs = torch.load(f"./data/gpt-j-{ds_name}-layer-{layer}-cur_zs.pt")

                torch.save(layer_ks, f"./data/gpt-j-layer-{layer}-ks.pt")
                torch.save(cur_zs, f"./data/gpt-j-layer-{layer}-cur_zs.pt")

            else:
                # Compute residual error
                cur_zs = get_module_input_output_at_words(
                    model,
                    tok,
                    z_layer,
                    context_templates=[request["prompt"] for request in requests],
                    words=[request["subject"] for request in requests],
                    module_template=hparams.layer_module_tmp,
                    fact_token_strategy=hparams.fact_token,
                    track='out'
                ).T
                
                # Get current model activations
                layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T

            # DEBUG
            # if not os.path.exists(f"./data/{ds_name}_layer{layer}_ks.pt"):
            #     layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
            #     torch.save(layer_ks, f"./data/{ds_name}_layer{layer}_ks.pt")
            # else:
            #     layer_ks = torch.load(f"./data/{ds_name}_layer{layer}_ks.pt")
            
            print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

            # the following was wrong
            # targets = zs - cur_zs

            if edit_to is not None:
                targets = zs - cur_zs
            else:
                targets = cur_zs - zs

            print("z error", torch.linalg.norm(targets, dim=0).mean())

            repeat_factor = (layer_ks.size(1) // targets.size(1))
            targets = targets.repeat_interleave(repeat_factor, dim=1)

            # Load covariance matrix
            force_recompute = False
            # force_recompute = layer != hparams.layers[0]
            cov = get_cov(
                model,
                tok,
                hparams.rewrite_module_tmp.format(layer),
                hparams.mom2_dataset,
                hparams.mom2_n_samples
                if not force_recompute
                else hparams.mom2_n_samples // 10,
                hparams.mom2_dtype,
                force_recompute=force_recompute,
            )

            layer_ks, targets = layer_ks.cpu(), targets.cpu()
            cov = cov.cpu()

            # Compute update in double precision
            layer_ks, targets = (
                layer_ks.double(),
                targets.double(),
            )

            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            orig_norm = torch.linalg.norm(weights[weight_name])
            print("orig norm", orig_norm)

            mom2_update_weight = hparams.mom2_update_weight

            if edit_to is not None:
                adj_k = torch.linalg.solve(
                    mom2_update_weight * cov.double() + layer_ks @ layer_ks.T,
                    layer_ks,
                )
            else:
                adj_k = torch.linalg.solve(
                    mom2_update_weight * cov.double() - layer_ks @ layer_ks.T,
                    layer_ks,
                )

            resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
            upd_matrix = resid @ adj_k.T

            # Adjust update matrix shape
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
            print("upd norm", torch.linalg.norm(upd_matrix))

            # Update model weights and record desired changes in `delta` variable
            with torch.no_grad():
                weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float().cuda()
                if alg_name == 'LAW':
                    deltas[weight_name] = upd_matrix.detach().cpu()
                else:
                    deltas[weight_name] = (
                        adj_k.detach().cpu(),
                        resid.detach().cpu(),
                    )

            if alg_ame == 'LAW':
                cov.cpu()
                # Clear GPU memory
                for x in [layer_ks]:
                    x.cpu()
                    del x
                torch.cuda.empty_cache()
            else:
                # Clear GPU memory
                cov.cpu()
                for x in [layer_ks, cur_zs, targets]:
                    x.cpu()
                    del x
                torch.cuda.empty_cache()

    else:

        if not hparams.random_initialize:

            zs = get_zs(requests, hparams, cache_template, model, tok, context_templates)

        if (not hparams.random_initialize) and hparams.compute_initialization_at_once:
            initialized_matrixes = {}
            for i, layer in enumerate(hparams.layers):
                layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
                print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

                if ds_name == 'wiki':
                    torch.save(layer_ks, f"./data/tmp/layer_ks_wiki_z_layer_layer_{layer}.pt")

                # Load covariance matrix
                force_recompute = False
                cov = get_cov(
                    model,
                    tok,
                    hparams.rewrite_module_tmp.format(layer),
                    hparams.mom2_dataset,
                    hparams.mom2_n_samples
                    if not force_recompute
                    else hparams.mom2_n_samples // 10,
                    hparams.mom2_dtype,
                    force_recompute=force_recompute,
                )

                layer_ks = layer_ks.cpu().double()
                cov = cov.cpu().double()

                weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
                # weight = weights[weight_name].T.double()
                orig_norm = torch.linalg.norm(weights[weight_name])
                print("orig norm", orig_norm)

                mom2_update_weight = hparams.mom2_update_weight

                cur_zs = get_module_input_output_at_words(
                    model,
                    tok,
                    hparams.layers[-1],
                    context_templates=[request["prompt"] for request in requests],
                    words=[request["subject"] for request in requests],
                    module_template=hparams.layer_module_tmp,
                    fact_token_strategy=hparams.fact_token,
                    track='out'
                ).T

                targets = zs - cur_zs

                repeat_factor = (layer_ks.size(1) // targets.size(1))
                targets = targets.repeat_interleave(repeat_factor, dim=1)

                layer_ks, targets = layer_ks.cpu(), targets.cpu()
                cov = cov.cpu()

                # Compute update in double precision
                layer_ks, targets = (
                    layer_ks.double(),
                    targets.double(),
                )

                weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
                orig_norm = torch.linalg.norm(weights[weight_name])
                print("orig norm", orig_norm)

                mom2_update_weight = hparams.mom2_update_weight

                while True:
                    adj_k = torch.linalg.solve(
                        mom2_update_weight * cov.double() + layer_ks @ layer_ks.T,
                        layer_ks,
                    )

                    resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
                    upd_matrix = resid @ adj_k.T

                    # Adjust update matrix shape
                    upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
                    print("upd norm", torch.linalg.norm(upd_matrix))

                    if hparams.skip_search:
                        print("Final lambda:", mom2_update_weight)
                        break

                    if torch.linalg.norm(upd_matrix) > orig_norm * 3 / 4:
                        print("Norm too large, reducing...")
                        mom2_update_weight += 100000
                    elif torch.linalg.norm(upd_matrix) < orig_norm / 3:
                        print("Norm too small, increasing...")
                        mom2_update_weight *= hparams.decay_factor
                    else:
                        print("Final lambda:", mom2_update_weight)
                        break
                
                initialized_matrixes[layer] = upd_matrix.detach()

                # Update model weights and record desired changes in `delta` variable
                with torch.no_grad():
                    weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float().cuda()

            # Restore state of original model
            with torch.no_grad():
                for k, v in weights.items():
                    v[...] = weights_copy[k]

        original_request_number = len(requests)
        previous_request_number = len(requests)
        # for i, layer in enumerate(hparams.layers[::-1]):

        if hparams.keep_original_requests:
            original_requests = deepcopy(requests)
            original_zs = deepcopy(zs.cpu())

        for i, layer in enumerate(hparams.layers):

            mom2_update_weight = hparams.mom2_update_weight

            print(f"\n\nLAYER {layer}\n")

            if (not hparams.random_initialize) and (not hparams.compute_initialization_at_once):

                # Get current model activations

                layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T

                print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

                # Load covariance matrix
                force_recompute = False
                # force_recompute = layer != hparams.layers[0]
                cov = get_cov(
                    model,
                    tok,
                    hparams.rewrite_module_tmp.format(layer),
                    hparams.mom2_dataset,
                    hparams.mom2_n_samples
                    if not force_recompute
                    else hparams.mom2_n_samples // 10,
                    hparams.mom2_dtype,
                    force_recompute=force_recompute,
                )

                layer_ks = layer_ks.cpu().double()
                cov = cov.cpu().double()

                weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
                # weight = weights[weight_name].T.double()
                orig_norm = torch.linalg.norm(weights[weight_name])
                print("orig norm", orig_norm)

                mom2_update_weight = hparams.mom2_update_weight
                # recompute upd_matrix
                tp = [request["prompt"] for request in requests]
                wd = [request["subject"] for request in requests]
                # cur_zs = get_module_input_output_at_words(
                #     model,
                #     tok,
                #     hparams.layers[-1],
                #     context_templates=tp,
                #     words=wd,
                #     module_template=hparams.layer_module_tmp,
                #     fact_token_strategy=hparams.fact_token,
                # )[1].T

                cur_zs = get_module_input_output_at_words(
                    model,
                    tok,
                    hparams.layers[-1],
                    context_templates=tp,
                    words=wd,
                    module_template=hparams.layer_module_tmp,
                    fact_token_strategy=hparams.fact_token,
                    track='out'
                ).T

                targets = zs - cur_zs

                repeat_factor = (layer_ks.size(1) // targets.size(1))
                targets = targets.repeat_interleave(repeat_factor, dim=1)

                layer_ks, targets = layer_ks.cpu(), targets.cpu()
                cov = cov.cpu()

                # Compute update in double precision
                layer_ks, targets = (
                    layer_ks.double(),
                    targets.double(),
                )

                weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
                orig_norm = torch.linalg.norm(weights[weight_name])
                print("orig norm", orig_norm)

                mom2_update_weight = hparams.mom2_update_weight

                while True:
                    adj_k = torch.linalg.solve(
                        mom2_update_weight * cov.double() + layer_ks @ layer_ks.T,
                        layer_ks,
                    )

                    resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
                    upd_matrix = resid @ adj_k.T

                    # Adjust update matrix shape
                    upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
                    print("upd norm", torch.linalg.norm(upd_matrix))

                    if torch.linalg.norm(upd_matrix) > orig_norm * 3 / 4:
                        print("Norm too large, reducing...")
                        mom2_update_weight += 100000
                    elif torch.linalg.norm(upd_matrix) < orig_norm / 3:
                        print("Norm too small, increasing...")
                        mom2_update_weight *= hparams.decay_factor
                    else:
                        print("Final lambda:", mom2_update_weight)
                        break
                    
                    # break
                
            else:
                
                

                if hparams.random_initialize: 
                    weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
                    # weight = weights[weight_name].T.double()
                    orig_norm = torch.linalg.norm(weights[weight_name])
                    print("orig norm", orig_norm)
                    # upd_matrix = torch.randn(weights[weight_name].shape).double()
                    upd_matrix = torch.randn(weights[weight_name].shape).double() * hparams.noise_scale

                else:
                    upd_matrix = initialized_matrixes[layer]

                layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
                print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")
                # Load covariance matrix
                force_recompute = False
                # force_recompute = layer != hparams.layers[0]
                cov = get_cov(
                    model,
                    tok,
                    hparams.rewrite_module_tmp.format(layer),
                    hparams.mom2_dataset,
                    hparams.mom2_n_samples
                    if not force_recompute
                    else hparams.mom2_n_samples // 10,
                    hparams.mom2_dtype,
                    force_recompute=force_recompute,
                )

                layer_ks = layer_ks.cpu().double()
                cov = cov.cpu().double()

                weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
                # weight = weights[weight_name].T.double()
                orig_norm = torch.linalg.norm(weights[weight_name])
                print("orig norm", orig_norm)
            
        

            alpha = None
            transposed = False
            if upd_matrix.shape[1] == cov.shape[0]:
                upd_matrix = upd_matrix.T
                transposed = True

            if hparams.random_initialize:
                # alpha = hparams.alpha
                alpha = None
                beta = hparams.beta

                # upd_matrix = upd_matrix / torch.sqrt(torch.trace(upd_matrix.T @ cov.double() @ upd_matrix) / torch.trace(cov) / beta)
            
            else:
                alpha = None
                beta = torch.trace(upd_matrix.T @ cov.double() @ upd_matrix) / torch.trace(cov)

            beta = beta * hparams.beta_ratio
            # TODO: mom2_update_weight is not defined if compute_initialization_at_once is False?
            print("mom2_update_weight:", mom2_update_weight)
            print("Before update, the loss value for K*:", torch.linalg.norm(upd_matrix.T @ layer_ks))

            # print("Before update, the loss value for K*:", torch.sum(torch.abs(upd_matrix.T @ layer_ks)))

            if hparams.simply_combine_objective:
                
                while True:

                    upd_matrix, success = update_upd_matrix_with_lambda(mom2_update_weight, 
                                    cov, layer_ks, hparams.learning_rate, upd_matrix.detach(), threshold=orig_norm / 4, 
                                    scheduler=hparams.scheduler, total_iters=hparams.total_iters, lr_after_hitting_boundary=hparams.lr_after_hitting_boundary,
                                    beta=beta, alpha=alpha, projection=False, verbose=True, loss_type='l2')

                    if success:
                        break
                        
                    mom2_update_weight *= 0.8

                    print("Drop lambda to:", mom2_update_weight)
            
            else:

                upd_matrix = update_upd_matrix(mom2_update_weight, 
                                cov, layer_ks, hparams.learning_rate, upd_matrix.detach(), threshold=orig_norm / 4, 
                                scheduler=hparams.scheduler, total_iters=hparams.total_iters, lr_after_hitting_boundary=hparams.lr_after_hitting_boundary,
                                beta=beta, alpha=alpha, projection=False, verbose=True, loss_type='l2')

            print("After update, the loss value for K*:", torch.linalg.norm(upd_matrix.T @ layer_ks))

            print("Delta norm:", torch.linalg.norm(upd_matrix))
            print("Delta * K norm:", torch.trace(upd_matrix.T @ cov.double() @ upd_matrix))
            print("Delta * K* norm:", torch.linalg.norm(upd_matrix.T @ layer_ks) ** 2)

            if transposed:
                upd_matrix = upd_matrix.T

            # Update model weights and record desired changes in `delta` variable
            with torch.no_grad():
                weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float().cuda()
                deltas[weight_name] = upd_matrix.detach().cpu()

            cov.cpu()
            # Clear GPU memory
            for x in [layer_ks]:
                x.cpu()
                del x
            torch.cuda.empty_cache()

            if hparams.prune_requests:

                if hparams.keep_original_requests:

                    predictions = get_predictions(None, tok, model, ds_name, original_requests)
                    true_prediction_case_ids = set([key for key, item in predictions.items() if item['correct']])
                    indices = [idx for idx, request in enumerate(original_requests) if request['case_id'] in true_prediction_case_ids]
                    requests = [request for request in original_requests if request['case_id'] in true_prediction_case_ids]
                    print(f"After layer {layer}, ", len(requests), f"requests left, including {len(requests) / original_request_number} percent")

                    if len(requests) == 0:
                        break
                    
                    zs = original_zs[:, torch.tensor(indices)].to(zs.device)
                
                else:

                    # need to get predictions to determine which requests need to be discarded
                    predictions = get_predictions(None, tok, model, ds_name, requests)
                    true_prediction_case_ids = set([key for key, item in predictions.items() if item['correct']])
                    
                    indices = [idx for idx, request in enumerate(requests) if request['case_id'] in true_prediction_case_ids]
                    
                    requests = [request for request in requests if request['case_id'] in true_prediction_case_ids]
                    print(f"After layer {layer}, ", len(requests), f"requests left, including {len(requests) / original_request_number} percent")

                    # if previous_request_number - len(requests) < 10:
                    #     with torch.no_grad():
                    #         weights[weight_name][...] = weights_copy[weight_name]
                    #         deltas[weight_name] = torch.zeros_like(upd_matrix).detach().cpu()

                    # previous_request_number = len(requests) 

                    if len(requests) == 0:
                        break

                    zs = zs[:, torch.tensor(indices)]

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    if 'gpt2-xl' in model_name:
        model_name = 'gpt2-xl'
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return (
        torch.inverse(COV_CACHE[key].to("cuda")) if inv else COV_CACHE[key].to("cuda")
    )


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
