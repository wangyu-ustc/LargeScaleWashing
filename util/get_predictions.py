from tqdm import tqdm
import torch

def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


def get_predictions(ds=None, tok=None, model=None, ds_name=None, requests=None):

    padding_side = tok.padding_side
    with torch.no_grad():
        batch_size = 32
        tok.pad_token = tok.eos_token
        tok.padding_side='left'

        predictions = {}

        dataset = ds if ds is not None else requests

        for batch in tqdm(chunks(dataset, batch_size), total=len(dataset) // batch_size):

            if ds is not None:
                # if ds_name == 'zsre':
                #     prompts = [item['requested_rewrite']['prompt'].format(item['requested_rewrite']['subject']) for item in batch]
                #     targets = [item['requested_rewrite']['target_new']['str'].lower() for item in batch]
                # else:
                prompts = [item['requested_rewrite']['prompt'].format(item['requested_rewrite']['subject']) for item in batch]
                targets = [item['requested_rewrite']['target_true']['str'].lower() for item in batch]
            else:
                # if ds_name == 'zsre':
                #     prompts = [item['prompt'].format(item['subject']) for item in batch]
                #     targets = [item['target_new']['str'].lower() for item in batch]
                # else:
                prompts = [item['prompt'].format(item['subject']) for item in batch]
                targets = [item['target_true']['str'].lower() for item in batch]

                prompts = [p.strip() for p in prompts]
            
            inputs = tok(prompts, return_tensors='pt', padding=True).to('cuda')
            predicted_answer = model.generate(**inputs, max_new_tokens=10, pad_token_id=tok.eos_token_id)[:, len(inputs['input_ids'][0]):]

            for j, output in enumerate(predicted_answer):

                output_text = tok.decode(output, skip_special_tokens=True)

                if targets[j].strip() in output_text.lower():
                    predictions[str(batch[j]['case_id'])] = {
                        'correct': 1,
                        'prediction': output_text
                    }
                
                else:
                    predictions[str(batch[j]['case_id'])] = {
                        'correct': 0,
                        'prediction': output_text
                    }
    tok.padding_side = padding_side
    return predictions
