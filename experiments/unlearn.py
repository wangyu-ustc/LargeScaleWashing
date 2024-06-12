import os
import json
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union
from scipy.linalg import svd
import numpy as np
from tqdm import tqdm
import copy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from memit import get_cov

from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    MQUAKEDataset,
    MultiMQUAKEDataset,
    MultiUnlearnDataset,
    MultiWikiDataset,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from memit import MEMITHyperParams, apply_law_to_model
from util import nethook, get_predictions
from util.globals import *
from rome.compute_existing_u import get_inv_cov

ALG_DICT = {
    "MEMIT": (MEMITHyperParams, apply_law_to_model),
    "LAW": (MEMITHyperParams, apply_law_to_model),
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
    'wiki': (MultiWikiDataset, compute_rewrite_quality_counterfact),
}


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    save_model: bool,
    from_pretrained: str,
    dir_name: str,
    num_edits: int = 1,
    use_cache: bool = False,
    edit_to: str = None,
    alpha: float = None,
    beta: float = None,
    expand_prompt: bool = False
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory
    # Create new dir if not continuing from prev run OR prev run doesn't exist
    if (
        continue_from_run is None
        or not (run_dir := RESULTS_DIR / dir_name / continue_from_run).exists()
    ):
        continue_from_run = None
    if continue_from_run is None:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")

    # Get run hyperparameters
    
    if hparams_fname is None:
        hparams = None
        hparams_dict = {"model": model_name}
        with open(run_dir / "params.json", "w") as f:
            json.dump(hparams_dict, f, indent=1)

    else:
        params_path = (
            run_dir / "params.json"
            if continue_from_run is not None
            else HPARAMS_DIR / alg_name / hparams_fname
        )
        hparams = params_class.from_json(params_path)

        if alpha is not None:
            print("Overwrite alpha as", alpha)
            hparams.alpha = alpha
        if beta is not None:
            print("Overwrite beta as", beta)
            hparams.beta = beta

        if not (run_dir / "params.json").exists():

            with open(params_path, "r") as f:
                hparams_dict = json.load(f)
            with open(run_dir / "params.json", "w") as f:
                hparams_dict['model'] = model_name
                hparams_dict['ds_name'] = ds_name
                hparams_dict['expand_prompt'] = expand_prompt
                if edit_to is not None:
                    hparams_dict['edit_to'] = edit_to
                json.dump(hparams_dict, f, indent=1)
    
    print(f"Executing {alg_name} with parameters {hparams}")

    # Instantiate vanilla model
    print("Instantiating model")
    if type(model_name) is str:
        if 'llama' in model_name:
            from transformers import LlamaForCausalLM, LlamaTokenizer
            if model_name == 'openllama-3b':
                model = LlamaForCausalLM.from_pretrained("openlm-research/open_llama_3b_v2").cuda()
                tok = LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b_v2")
            elif model_name == 'llama2-7b':
                model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").cuda()
                tok = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        
        elif "gpt2" in model_name:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            if from_pretrained is not None:
                model = GPT2LMHeadModel.from_pretrained(from_pretrained).cuda()
                tok = GPT2Tokenizer.from_pretrained(from_pretrained) 
            else:
                model = GPT2LMHeadModel.from_pretrained(model_name).cuda()
                tok = GPT2Tokenizer.from_pretrained(model_name)
        
        elif 'gpt-j-6b' in model_name:
            model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-j-6b').cuda()
            tok = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6b') 

        else:
            model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
            tok = AutoTokenizer.from_pretrained(model_name)
        
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path

    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit)

    # check if the model knows about the facts:
    if os.path.exists(str(PRED_DIR) + f'/{model_name.replace("/", "_")}_{alg_name}_{ds_name}.json'):
        predictions = json.load(open(str(PRED_DIR) + f'/{model_name.replace("/", "_")}_{alg_name}_{ds_name}.json', 'r'))
    else:
        predictions = get_predictions(ds, tok, model, ds_name)

        with open(str(PRED_DIR) + f'/{model_name.replace("/", "_")}_{alg_name}_{ds_name}.json', 'w') as f:
            json.dump(predictions, f, indent=1)

    if isinstance(predictions[str(ds[0]['case_id'])], dict):
        indices = [i for i, record in enumerate(ds) if predictions[str(record['case_id'])]['correct'] == 1]
    else:
        indices = [i for i, record in enumerate(ds) if predictions[str(record['case_id'])] == 1]

    print(f"{len(indices)} data out of {len(ds)} are known to the model")

    ds = [ds[i] for i in indices]

    if expand_prompt:
        # process the dataset to expand the prompts:
        for record in ds:
            prediction = predictions[str(record['case_id'])]['prediction']
            if ds_name == 'zsre':
                target = record['requested_rewrite']['target_new']['str']
            else:
                target = record['requested_rewrite']['target_true']['str']
            index = prediction.lower().index(target.lower())
            record['requested_rewrite']['prompt'] = record['requested_rewrite']['prompt'] + " " + prediction[:index].strip()

    if from_pretrained is not None:

        pred_path = str(PRED_DIR) + f"/{from_pretrained.split('/')[-2]}_{alg_name}_{ds_name}.json"
        if os.path.exists(pred_path):
            predictions = json.load(open(pred_path, 'r'))
        else:
            predictions = get_predictions(ds, tok, model, ds_name)
            with open(pred_path, 'w') as f:
                json.dump(predictions, f, indent=1)
            f.close()

            predictions = json.load(open(pred_path, 'r'))

        if isinstance(predictions[str(ds[0]['case_id'])], dict):
            indices = [i for i, record in enumerate(ds) if predictions[str(record['case_id'])]['correct'] == 1]
        else:
            indices = [i for i, record in enumerate(ds) if predictions[str(record['case_id'])] == 1]
        print(f"{len(indices)} data out of {len(ds)} are known to the model")
        ds = [ds[i] for i in indices]

    if edit_to is not None:
        if expand_prompt:
            kvdir = EDIT_NONE_KV_EXPAND_DIR
        else:
            kvdir = EDIT_NONE_KV_DIR
    else:
        kvdir = KV_DIR

    # Get cache templates
    cache_template = None
    if use_cache:
        # cache_template is used when calculating zs, 
        # zs is used for the initialized delta from MEMIT, 
        # so we need to specify the folder in cache_template as "MEMIT"
        cache_template = (
            kvdir
            / f"{model_name.replace('/', '_').split('_')[0]}_MEMIT"
            / (
                f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz" if from_pretrained is None
                else f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}_from_{from_pretrained.split('/')[-2]}.npz"
            )
        )
        print(f"Will load cache from {cache_template}")


    weights_copy = None

    # Iterate through dataset
    for record_chunks in chunks(ds, len(ds)):
        
        case_result_template = str(run_dir / "{}_edits-case_{}.json")

        # Is the chunk already done?
        already_finished = True
        for record in record_chunks:
            if not Path(
                case_result_template.format(len(ds), record["case_id"])
            ).exists():
                already_finished = False
                break
        if already_finished:
            continue

        # Compute weight changes + record weights that changed
        case_ids = [record["case_id"] for record in record_chunks]
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["LAW", "MEMIT"]) else dict()

        if edit_to is None:
            for record in record_chunks:
                if ds_name == 'zsre':
                    record['requested_rewrite']['target_true'] = record['requested_rewrite']['target_new']
                else:
                    record['requested_rewrite']['target_new'] = record['requested_rewrite']['target_true']
        else:
            for record in record_chunks:
                if ds_name == 'zsre':
                    record['requested_rewrite']['target_true'] = {
                        'str': record['requested_rewrite']['target_new']['str']
                    }
                record['requested_rewrite']['target_new'] = {
                    'str': edit_to
                }

        start = time()

        edited_model, weights_copy = apply_algo(
            model,
            tok,
            [
                {"case_id": record["case_id"], **record["requested_rewrite"]}
                for record in record_chunks
            ],
            hparams,
            ds_name=ds_name,
            copy=False,
            return_orig_weights=True,
            edit_to=edit_to,
            alg_name=alg_name,
            **args_conserve_memory,
            **etc_args,
        )
        
        exec_time = time() - start
        print("Execution took", exec_time)

        if save_model:
            new_model_name = f"{str(run_dir)}/{model_name.replace('/', '_')}_{alg_name}_{ds_name}"
            edited_model.save_pretrained(new_model_name)
            print(f"Model saved at {new_model_name}")
            tok.save_pretrained(new_model_name)
            print(f"Tokenizer saved at {new_model_name}")

    if weights_copy is not None:
        # Restore original weights
        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(model, k)[...] = v.to("cuda")

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["LAW", "MEMIT"],
        default="LAW",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        choices=["memory-openllama", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "gpt2_full_trained",
                "EleutherAI/gpt-j-6B", 'openllama-3b', 'llama-7b', 'llama2-7b', 'gpt-j-6b'],
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default=None,
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre", "mquake", 'all', 'wiki'],
        default="mcf",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=1000000,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=10000,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--save_model",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--edit_to',
        default=None,
        type=str,
    )
    parser.add_argument(
        '--from_pretrained',
        default=None,
        type=str,
    )
    parser.add_argument(
        "--alpha",
        default=None,
        type=float
    )
    parser.add_argument(
        '--beta',
        default=None,
        type=float
    )
    parser.add_argument(
        '--expand_prompt',
        default=False,
        action='store_true'
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        args.save_model,
        args.from_pretrained,
        dir_name=f"{args.alg_name}/{args.model_name.replace('/', '_')}_{args.ds_name}",
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        edit_to=args.edit_to,
        alpha=args.alpha,
        beta=args.beta,
        expand_prompt=args.expand_prompt
    )