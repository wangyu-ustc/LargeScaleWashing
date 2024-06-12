python main.py --model gpt2-xl --dataset mcf  --batch_size 2 --lr 1e-6 --epochs 1
python main.py --model gpt2-xl --dataset zsre  --batch_size 2 --lr 1e-6 --epochs 1
python main.py --model gpt2-xl --dataset wiki  --batch_size 2 --lr 1e-6 --epochs 1
python main.py --model "EleutherAI/gpt-j-6B" --dataset mcf  --use_lora --batch_size 2 --lr 1e-5 --epochs 5
python main.py --model "EleutherAI/gpt-j-6B" --dataset zsre  --use_lora --batch_size 2 --lr 1e-5 --epochs 5
python main.py --model "EleutherAI/gpt-j-6B" --dataset wiki --use_lora --batch_size 2 --lr 1e-5 --epochs 5
