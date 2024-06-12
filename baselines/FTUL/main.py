import os
import json
import torch
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling, AdamW
from peft import get_peft_model, LoraConfig, TaskType  # Import necessary LoRA components

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt2-xl")
parser.add_argument("--use_lora", action='store_true', default=False)
parser.add_argument("--dataset", type=str, default='zsre')
parser.add_argument("--target_dir", default='../results/ftul_target_models')
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-8)
args = parser.parse_args()

# Custom Dataset
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer
        self.encodings = tokenizer(texts, truncation=True, padding=True)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item


def get_unlearn_dataset(model, dataset):

    if dataset == 'zsre':
        raw_data = json.load(open("../data/zsre_mend_eval.json", "r"))
        predictions = json.load(open(f"../data/preds/{model.replace('/', '-')}_MEMIT_{dataset}.json", "r"))
        data = []
        for i, record in enumerate(raw_data):
            if not predictions[str(i)]['correct']:
                continue
            data.append(record['src'] + " Answer: " + record['answers'][0])

    elif dataset == 'mcf':
        raw_data = json.load(open("../data/multi_counterfact.json"))
        predictions = json.load(open(f"../data/preds/{model.replace('/', '-')}_MEMIT_{dataset}.json", "r"))
        data = []
        for record in raw_data:
            if not predictions[str(record['case_id'])]['correct']:
                continue
            record = record['requested_rewrite']
            data.append(record['prompt'].format(record['subject']) + ' ' + record['target_true']['str'])

    elif dataset == 'wiki':
        raw_data = json.load(open("../data/wiki_facts.json"))
        predictions = json.load(open(f"../data/preds/{model.replace('/', '-')}_MEMIT_{dataset}.json", "r"))
        data = []
        for record in raw_data:
            if not predictions[str(record['case_id'])]['correct']:
                continue
            record = record['requested_rewrite']
            data.append(record['prompt'].format(record['subject']) + ' ' + record['target_true']['str'])

    return data


if __name__ == '__main__':
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).cuda()

    if args.use_lora:
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]
        )
        model = get_peft_model(model, lora_config)

    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    # set up the dataset (Initialize Dataset and DataLoader)

    data = get_unlearn_dataset(args.model, args.dataset)

    tokenizer.pad_token = tokenizer.eos_token
    dataset = TextDataset(data, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)

    # Training loop
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr)

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
    inputs = tokenizer("I want to", return_tensors='pt').to("cuda")
    outputs = model.generate(**inputs, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("output_text:", output_text)
    model.save_pretrained(os.path.join(args.target_dir, f"{args.model.replace('/', '-')}_{args.dataset}_target"))
