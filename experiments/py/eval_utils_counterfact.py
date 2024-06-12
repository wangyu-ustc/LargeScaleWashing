"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_counterfact` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain

import nltk
import numpy as np
import scipy
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer

from dsets import AttributeSnippets
from util.generate import generate_fast
from util.perplexity import perplexity


def compute_rewrite_quality_counterfact(
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
    if "paraphrase_prompts" in record:
        paraphrase_prompts = record["paraphrase_prompts"]
        neighborhood_prompts = record["neighborhood_prompts"]
        attribute_prompts = record["attribute_prompts"]
        generation_prompts = record["generation_prompts"]
    else:
        paraphrase_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
        neighborhood_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
        attribute_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
        generation_prompts = [record["requested_rewrite"]["prompt"].format(subject)]

    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
        neighborhood_prompts,
        attribute_prompts,
    ]

    # Flatten all the evaluated prefixes into one list.
    probs, entropies = test_batch_prediction(
        model, tok, list(chain(*prob_prompts)), target_new["str"], target_true["str"]
    )

    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    ret_entropies = [entropies[cutoffs[i - 1]: cutoffs[i]] for i in range(1, len(cutoffs))]

    # Structure the restuls as a dictionary.
    ret = {
        f"{key}_probs": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
                "attribute_prompts",
            ]
        )
    }

    ret.update({
        f"{key}_entropies": ret_entropies[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
                "attribute_prompts",
            ]
        )
    })

    if snips is not None:
        # # Gather reference texts
        # rel_id = record["requested_rewrite"]["relation_id"]
        # # consistency_texts = [x["text"] for x in snips[rel_id][target_new["id"]]]
        # essence_texts = [
        #     x["text"]
        #     for x in snips[rel_id][target_new["id"]]
        #     if x["name"] == record["requested_rewrite"]["subject"]
        # ]
        
        # assert (
        #     len(consistency_texts) > 0
        # ), "Must have consistency texts to evaluate generation"
        gen_stats = test_generation(
            model,
            tok,
            generation_prompts,
            # essence_texts,
            # consistency_texts,
            # vec,
        )
        ret.update(gen_stats)

    return ret


def test_batch_prediction(
    model,
    tok,
    prefixes: typing.List[str],
    target_new: str,
    target_true: str,
):
    """ """

    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in [target_new, target_true]
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # a_tok, b_tok = (tok(f" {n}", add_special_tokens=False)["input_ids"] for n in [target_new, target_true])
    if isinstance(tok, GPT2Tokenizer):
        a_tok, b_tok = (tok.encode(f" {n}") for n in [target_new, target_true])
    else:
        a_tok, b_tok = (tok(f"{n}", add_special_tokens=False)["input_ids"] for n in [target_new, target_true])
    choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])

    with torch.no_grad():

        all_logits = []
        for input_ids, attention_mask in zip(prompt_tok.input_ids, prompt_tok.attention_mask):
            logits = model(
                input_ids=input_ids.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
            ).logits
            all_logits.append(logits.squeeze(0))

        logits = all_logits

    results = np.zeros((len(logits),), dtype=np.float32)
    entropies = np.zeros((len(logits),), dtype=np.float32)

    for i in range(len(logits)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]

            results[i] += -torch.nn.functional.log_softmax(
                logits[i][prefix_lens[i // 2] + j - 1, :], dim=0
            )[cur_tok].item()

            # Convert logits to probabilities using softmax
            probs = torch.softmax(logits[i][prefix_lens[i // 2] + j - 1, :], dim=0)
            # Calculate entropy
            entropy = -(probs * torch.log(probs)).sum()
            entropies[i] += entropy.item()

        results[i] /= cur_len
        entropies[i] /= cur_len

    return [
        {"target_new": results[i].item(), "target_true": results[i + 1].item()}
        for i in range(0, len(results), 2)
    ], [
        {"target_new": entropies[i].item(), "target_true": entropies[i + 1].item()}
        for i in range(0, len(entropies), 2)
    ]

def test_generation(
    model,
    tok,
    prefixes: typing.List[str],
    # essence_texts: typing.List[str],
    # consistency_texts: typing.List[str],
    # vec: TfidfVectorizer,
):
    gen_texts = generate_fast(
        model,
        tok,
        prefixes,
        n_gen_per_prompt=1,
        max_out_len=100,
    )

    ngram_entropy = n_gram_entropy(gen_texts)

    ret = {
        "ngram_entropy": ngram_entropy,
        "text": gen_texts,
    }

    # if len(essence_texts) > 0:
    #     ppl = perplexity(model, tok, " ".join(essence_texts), max_input_length=100)
    #     ret.update({"essence_score": ppl, "essence_text": essence_texts})

    return ret


def n_gram_entropy(gen_texts, agg="arith"):
    assert agg in ["arith", "geom"]

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt) for txt in gen_texts]
    ).item()


def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()

        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


def tfidf_similarity(text_a, text_b, vec):
    encs = vec.transform([text_a, text_b]).A
    norm = np.linalg.norm
    return (np.dot(encs[0], encs[1]) / norm(encs[0]) / norm(encs[1])).item()
