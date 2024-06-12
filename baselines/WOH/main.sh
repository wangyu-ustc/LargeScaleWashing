python main.py --model gpt2-xl --dataset mcf --alpha 0.5 --batch_size 2
python main.py --model gpt2-xl --dataset zsre --alpha 0.5 --batch_size 2
python main.py --model gpt2-xl --dataset wiki --alpha 0.5 --batch_size 2
python main.py --model "EleutherAI/gpt-j-6B" --dataset mcf --alpha 5 --use_lora --batch_size 2
python main.py --model "EleutherAI/gpt-j-6B" --dataset zsre --alpha 5 --use_lora --batch_size 2
python main.py --model "EleutherAI/gpt-j-6B" --dataset wiki --alpha 0.5 --use_lora --batch_size 2