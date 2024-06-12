"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import AttributeSnippets


def compute_rewrite_quality_zsre(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    snips: AttributeSnippets,
    # vec: TfidfVectorizer,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    paraphrase_prompts = [x + ' Answer:' for x in record["paraphrase_prompts"]]
    neighborhood_prompts = record["neighborhood_prompts"]

    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
    ]

    target_tok = tok(target_new["str"], add_special_tokens=False)["input_ids"]
    inp_prompts_og = list(chain(*prob_prompts))

    inp_prompts = [
        tok(el).input_ids + target_tok[:i]
        for el in inp_prompts_og
        for i in range(len(target_tok))
    ]
    inp_targets = [
        target_tok[i]
        for _ in range(len(inp_prompts_og))
        for i in range(len(target_tok))
    ]

    stuff_probs = test_batch_prediction_acc(model, tok, inp_prompts, inp_targets)

    neighborhood_target_tok = tok(record['neighborhood_prompts'][1], add_special_tokens=False)['input_ids']
    neighborhood_inp_prompt_tok = tok(record['neighborhood_prompts'][0], add_special_tokens=True)['input_ids']
    neighborhood_inp_prompts = [
        neighborhood_inp_prompt_tok + neighborhood_target_tok[:i] for i in range(len(neighborhood_target_tok))
    ]
    neighborhood_inp_targets = [
        neighborhood_target_tok[i]
        for i in range(len(neighborhood_target_tok))
    ]
    neighborhood_correct = test_batch_prediction_acc(
        model,
        tok,
        neighborhood_inp_prompts,
        neighborhood_inp_targets
    )

    probs = stuff_probs + neighborhood_correct

    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(
        [l * len(target_tok) for l in map(len, prob_prompts)]
    ).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    # Structure the restuls as a dictionary.
    ret = {
        f"{key}_correct": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
            ]
        )
    }
    ret["neighborhood_prompts_correct"] = neighborhood_correct

    return ret

def test_batch_prediction_acc(model, tok, prompts: typing.List[typing.List[int]], target):

    prompt_tok = [
        torch.tensor(prompt)
        for prompt in prompts
    ]

    with torch.no_grad():

        all_logits = []
        for input_ids in prompt_tok:
            logits = model(
                input_ids=input_ids.unsqueeze(0).cuda(),
                attention_mask=torch.ones(input_ids.shape[0], dtype=torch.long).unsqueeze(0).cuda()
            ).logits
            all_logits.append(
                logits.squeeze(0)
            )

        logits = [logits[-1] for logits in all_logits]
        logits = torch.stack(logits)
        ans = torch.argmax(logits, dim=1)

        correct_id = torch.tensor(target).cuda()

        return (ans == correct_id).detach().cpu().numpy().tolist()
