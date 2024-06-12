import json
import copy
import torch
import argparse
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch.utils.data import Dataset, DataLoader
from metrics import qa_f1_score
from util.globals import *

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
from memit import MEMITHyperParams, apply_unlearn_memit_to_model
from rome import ROMEHyperParams, apply_unlearn_rome_to_model

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
    "mquake": (MultiMQUAKEDataset, compute_rewrite_quality_counterfact),
    'wiki': (MultiWikiDataset, compute_rewrite_quality_counterfact),
    'all': (MultiUnlearnDataset, compute_rewrite_quality_counterfact),
}

def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess the data")
    # parser.add_argument("--data_path", type=str, default='./data/combined_data.json', help="Path to the preprocessed training data file")
    parser.add_argument('--ds_name', type=str, default='mcf')
    parser.add_argument("--mode", type=str, default='knowledge', choices=['knowledge', 'reason'])
    parser.add_argument('--model', default='gpt2-xl', type=str, help='model name or path')
    parser.add_argument('--num', default=None, type=int, help='number of examples to evaluate')
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument('--pred_path', default=None, type=str)
    parser.add_argument("--save_path", default=None, type=str)
    return parser.parse_args()

args = parse_args()

# Initialize model
if "gpt2-xl" in args.model:
    model = GPT2LMHeadModel.from_pretrained(args.model).cuda()
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    except:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
        tokenizer.save_pretrained(args.model)
elif 'gpt-j' in args.model:
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model).cuda()
    except:
        config = AutoConfig.from_pretrained("EleutherAI/gpt-j-6B")
        config.save_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model).cuda()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model) 
    except:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        tokenizer.save_pretrained(args.model)
        model.config.save_pretrained(args.model)

tokenizer.pad_token_id = tokenizer.eos_token_id

ds_class, ds_eval_method = DS_DICT[args.ds_name]
ds = ds_class(DATA_DIR, tok=tokenizer)

if args.pred_path is not None:
    with open(args.pred_path, 'r') as file:
        preds = json.load(file)
    file.close()
else:
    # args.model is of the form "results/MEMIT/EleutherAI_gpt-j-6B_wiki/run_000/"
    model_name = "_".join(args.model.split("/")[2].split("_")[:-1])
    alg_name = args.model.split("/")[1]
    ds_name = args.model.split("/")[2].split("_")[-1]
    with open(f'./data/preds/{model_name.replace("/", "-")}_{alg_name}_{ds_name}.json', 'r') as file:
        preds = json.load(file)
    file.close()

preds = {k:v for k,v in preds.items() if (v == 1 if isinstance(v, int) else v['correct'] == 1)}

ds = [item for item in ds if str(item['case_id']) in preds]

if args.num is not None:
    ds = ds[:args.num]

print("Number of examples:", len(ds))

total_score = 0
total_score_paraphrase = 0
total_positive_score = 0
total_num = 0
total_num_paraphrase = 0
total_positive_score_paraphrase = 0
total_hit = 0
total_probs = 0

batch_size = args.batch_size

tokenizer.pad_token = tokenizer.eos_token

tokenizer2 = copy.deepcopy(tokenizer)

tokenizer.padding_side='left'

all_predicted_answers = []

with torch.no_grad():
    # for example in tqdm(ds, total=len(ds)):
    for batch in tqdm(chunks(ds, batch_size), total=len(ds) // batch_size):

        if args.ds_name == 'zsre':
            answers = [example['requested_rewrite']['target_new']['str'] for example in batch]
        else:
            answers = [example['requested_rewrite']['target_true']['str'] for example in batch]
        
        prompts = [example['requested_rewrite']['prompt'].format(example['requested_rewrite']['subject']) for example in batch]

        inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to("cuda")

        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]
        logits = torch.softmax(logits, dim=1)
        answer_ids = tokenizer2(answers, return_tensors='pt', padding=True).input_ids.to("cuda")[:, 0]
        probs = - torch.log(torch.gather(logits, 1, answer_ids.unsqueeze(1)))

        total_probs += probs.sum().item()

        predicted_answer = model.generate(
            **inputs,
            max_new_tokens=10,
            pad_token_id=tokenizer.eos_token_id,
        )

        for j, output in enumerate(predicted_answer):

            all_predicted_answers.append(tokenizer.decode(output, skip_special_tokens=True))

            output_text = tokenizer.decode(output[inputs['input_ids'][j].shape[0]:], skip_special_tokens=True)
            total_hit += answers[j].lower() in output_text.lower()

            score = qa_f1_score(output_text, answers[j])
            total_score += score
            total_num += 1
            if score > 0:
                total_positive_score += 1

    # if "paraphrase_prompts" in example:

    #     for question in example["paraphrase_prompts"]:
    #         question_ids = tokenizer(question, return_tensors='pt').input_ids.cuda()

    #         predicted_answer = model.generate(
    #             question_ids,
    #             max_new_tokens=2,
    #             pad_token_id=tokenizer.eos_token_id,
    #         )[:, question_ids.shape[1]:][0].cpu()

    #         predicted_answer = tokenizer.decode(predicted_answer, skip_special_tokens=True).strip()

    #         score = qa_f1_score(predicted_answer, answer)
    #         total_score_paraphrase += score

    #         total_num_paraphrase += 1

    #         if score > 0:
    #             total_positive_score_paraphrase += 1

print("score:", round(100 * total_score / total_num, 2))
# print("score paraphrase:", round(100 * total_score_paraphrase / total_num_paraphrase, 2))
print("positive score:", round(100 * total_positive_score / total_num, 2))
# print("positive score paraphrase:", round(100 * total_positive_score_paraphrase / total_num_paraphrase, 2))
print("Accuracy:", round(100 * total_hit / total_num, 2))
print("Average log probs:", total_probs / total_num)

if args.save_path is not None:
    with open(args.save_path, 'w') as file:
        json.dump(all_predicted_answers, file, indent=4)
    file.close()
