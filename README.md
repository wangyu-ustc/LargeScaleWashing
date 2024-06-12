# **Large Scale Washing**

This is the official implementation of the paper [**Large Scale Knowledge Washing**](https://arxiv.org/abs/2405.16720). This repo is built based on [MEMIT](https://github.com/kmeng01/memit).

## Requirements
Please check the requirements in [MEMIT](https://github.com/kmeng01/memit). 

## Dataset Preperation
The files required for datasets `zsRE` and `Wiki-Latest` are all put under the folder `data`. Simply running the code below will automatically download the data needed for `CounterFactual`. 

## Knowledge Washing

### Main Experiments with Algorithm **LaW**

To run the algorithm **LaW**, please use the following commands:  
1. GPT2-XL with zsRE, CounterFactual and Wiki
```
python -m experiments.unlearn --alg_name=LAW --model_name=gpt2-xl --hparams_fname=gpt2-xl-mcf.json 
--use_cache --save_model --edit_to "<|endoftext|>" --ds_name zsre 
python -m experiments.unlearn --alg_name=LAW --model_name=gpt2-xl --hparams_fname=gpt2-xl-mcf.json 
--use_cache --save_model --edit_to "<|endoftext|>" --ds_name mcf 
python -m experiments.unlearn --alg_name=LAW --model_name=gpt2-xl --hparams_fname=gpt2-xl-mcf.json 
--use_cache --save_model --edit_to "<|endoftext|>" --ds_name wiki 
```

2. GPT-J-6B with zsRE, CounterFactual and Wiki
```
python -m experiments.unlearn --alg_name=LAW --model_name=EleutherAI/gpt-j-6B --hparams_fname=EleutherAI_gpt-j-6B-mcf.json --use_cache --save_model --edit_to "<|endoftext|>" --ds_name zsre
python -m experiments.unlearn --alg_name=LAW --model_name=EleutherAI/gpt-j-6B --hparams_fname=EleutherAI_gpt-j-6B-mcf.json --use_cache --save_model --edit_to "<|endoftext|>" --ds_name mcf
python -m experiments.unlearn --alg_name=LAW --model_name=EleutherAI/gpt-j-6B --hparams_fname=EleutherAI_gpt-j-6B-mcf.json --use_cache --save_model --edit_to "<|endoftext|>" --ds_name wiki
```

After running these commands, the models will be saved at 
the folder `results/LAW`. For instance, after washing the dataset `mcf` in GPT2-XL, the model would be saved in:
`results/LAW/gpt2-xl_mcf/run_000/gpt2-xl_MEMIT_mcf`. Then we can run the following commands to evaluate the knowledge encapsulation and the reasoning ability of the model: 
```
python eval_model_knowledge.py --model results/LAW/gpt2-xl_mcf/run_000/gpt2-xl_MEMIT_mcf --ds_name mcf
lm_eval --model hf --tasks lambada_openai,hellaswag,arc_easy --device cuda:0 --batch_size 8 --model_args pretrained=results/LAW/gpt2-xl_mcf/run_000/gpt2-xl_MEMIT_mcf
```


### Baselines
To run the algorithm **MEMIT**, please use the following commands:
1. GPT2-XL with zsRE, CounterFactual and Wiki
```
python -m experiments.unlearn --alg_name=MEMIT --model_name=gpt2-xl --hparams_fname=gpt2-xl-mcf.json 
--use_cache --save_model --edit_to "<|endoftext|>" --ds_name zsre 
python -m experiments.unlearn --alg_name=MEMIT --model_name=gpt2-xl --hparams_fname=gpt2-xl-mcf.json 
--use_cache --save_model --edit_to "<|endoftext|>" --ds_name mcf 
python -m experiments.unlearn --alg_name=MEMIT --model_name=gpt2-xl --hparams_fname=gpt2-xl-mcf.json 
--use_cache --save_model --edit_to "<|endoftext|>" --ds_name wiki 
```

2. GPT-J-6B with zsRE, CounterFactual and Wiki
```
python -m experiments.unlearn --alg_name=MEMIT --model_name=EleutherAI/gpt-j-6B --hparams_fname=EleutherAI_gpt-j-6B-mcf.json --use_cache --save_model --edit_to "<|endoftext|>" --ds_name zsre
python -m experiments.unlearn --alg_name=MEMIT --model_name=EleutherAI/gpt-j-6B --hparams_fname=EleutherAI_gpt-j-6B-mcf.json --use_cache --save_model --edit_to "<|endoftext|>" --ds_name mcf
python -m experiments.unlearn --alg_name=MEMIT --model_name=EleutherAI/gpt-j-6B --hparams_fname=EleutherAI_gpt-j-6B-mcf.json --use_cache --save_model --edit_to "<|endoftext|>" --ds_name wiki
```

To run the algorithms `FT`, `FT-UL`, `SeUL`, and `WOH`, please check the `baselines` folder. Each algorithm has its own folder with a `main.sh` file that contains the commands to run the code for all models and datasets. 

## Citations
If you find this codebase useful, please consider citing our paper: 
```
@misc{wang2024large,
      title={Large Scale Knowledge Washing}, 
      author={Yu Wang and Ruihan Wu and Zexue He and Xiusi Chen and Julian McAuley},
      year={2024},
      eprint={2405.16720},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```