import os
import json
import torch
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from functools import partial
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt2-xl")
parser.add_argument("--use_lora", action='store_true', default=False)
parser.add_argument("--dataset", type=str, default='zsre')
parser.add_argument("--target_dir", default='../results/selu_target_models')
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--alpha", type=float, default=5.0)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=1e-6)
args = parser.parse_args()

    
from torch.utils.data import Dataset

def collate_fn(batch, tokenizer):

    item, answer = zip(*batch)

    tokenized_item = tokenizer(item, truncation=True, return_tensors='pt', padding='longest')
    tokenized_answer = tokenizer(answer, truncation=True).input_ids
    
    inputs = tokenized_item['input_ids']
    attention_mask = tokenized_item['attention_mask']
    labels = tokenized_item['input_ids'].clone()

    for i in range(len(labels)):
        labels[i, torch.where(attention_mask[i] == 0)[0]] = -100
        labels[i, :torch.sum(attention_mask[i]) - len(tokenized_answer[0])] = -100

    return {
        'input_ids': inputs,
        'attention_mask': attention_mask,
        'labels': labels  # assuming answers are used as labels
    }


class TextDataset(Dataset):
    def __init__(self, data, answer, tokenizer):
        self.data = data
        self.answer = answer
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        answer = self.answer[idx]
        return item, answer


def get_unlearn_dataset(model, dataset):

    if dataset == 'zsre':
        raw_data = json.load(open("../data/zsre_mend_eval.json", "r"))
        predictions = json.load(open(f"../data/preds/{model.replace('/', '-')}_MEMIT_{dataset}.json", "r"))
        data = []
        answers = []
        for i, record in enumerate(raw_data):
            if not predictions[str(i)]['correct']:
                continue
            data.append(record['src'] + " Answer: " + record['answers'][0])
            answers.append(record['answers'][0])

    elif dataset == 'mcf':
        raw_data = json.load(open("../data/multi_counterfact.json"))
        predictions = json.load(open(f"../data/preds/{model.replace('/', '-')}_MEMIT_{dataset}.json", "r"))
        data = []
        answers = []
        for record in raw_data:
            if not predictions[str(record['case_id'])]['correct']:
                continue
            record = record['requested_rewrite']
            data.append(record['prompt'].format(record['subject']) + ' ' + record['target_true']['str'])
            answers.append(record['target_true']['str'])

    elif dataset == 'wiki':
        raw_data = json.load(open("../data/wiki_facts.json"))
        predictions = json.load(open(f"../data/preds/{model.replace('/', '-')}_MEMIT_{dataset}.json", "r"))
        data = []
        answers = []
        for record in raw_data:
            if not predictions[str(record['case_id'])]['correct']:
                continue
            record = record['requested_rewrite']
            data.append(record['prompt'].format(record['subject']) + ' ' + record['target_true']['str'])
            answers.append(record['target_true']['str'])

    return data, answers


if __name__ == '__main__':
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).cuda()

    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    # set up the dataset (Initialize Dataset and DataLoader)

    data, answer = get_unlearn_dataset(args.model, args.dataset)

    tokenizer.pad_token = tokenizer.eos_token
    dataset = TextDataset(data, answer, tokenizer)
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    collate_fn_partial = partial(collate_fn, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_partial)

    # Training loop
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    epochs = 1
    for epoch in range(args.epochs):
        for batch in tqdm(dataloader, total=len(dataloader)):
            optimizer.zero_grad()
            inputs = batch['input_ids'].cuda()
            labels = batch['labels'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            
            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            loss = - outputs.loss
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    print("Training completed.")

    # Save the model
    model.save_pretrained(os.path.join(args.target_dir, f"{args.model.replace('/', '-')}_{args.dataset}_target"))
    