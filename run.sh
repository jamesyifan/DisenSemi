#CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'MUTAG' --lr 0.001 --suffix 0 --epochs 500 --eta 1.0 --n_percents 5
#python3 main.py --dataset 'MUTAG' --target 0 --separate-encoder --use-unsup-loss

#python3 main.py --dataset 'MUTAG' --separate-encoder --use-unsup-loss

CUDA_VISIBLE_DEVICES=7 python3 -W ignore main.py --dataset 'MUTAG' --batch-size 256 --n-layer 3 --n-factor 4 --separate-encoder --use-unsup-loss
#CUDA_VISIBLE_DEVICES=0 python3 -W ignore main.py --dataset 'PTC_MR' --batch-size 256 --n-factor 4 --separate-encoder --use-unsup-loss
#CUDA_VISIBLE_DEVICES=2 python3 -W ignore main.py --dataset 'PROTEINS' --batch-size 256 --n-layer 5 --n-factor 4 --separate-encoder --use-unsup-loss
#CUDA_VISIBLE_DEVICES=2 python3 -W ignore main.py --dataset 'DD' --batch-size 128 --n-factor 4 --separate-encoder --use-unsup-loss
#CUDA_VISIBLE_DEVICES=0 python3 -W ignore main.py --dataset 'NCI1' --batch-size 256 --n-factor 4 --separate-encoder --use-unsup-loss

#CUDA_VISIBLE_DEVICES=7 python3 -W ignore main.py --dataset 'IMDB-BINARY' --batch-size 256 --n-layer 5 --n-factor 4 --separate-encoder --use-unsup-loss
#CUDA_VISIBLE_DEVICES=3 python3 -W ignore main.py --dataset 'IMDB-MULTI' --batch-size 256 --n-factor 4 --separate-encoder --use-unsup-loss

#CUDA_VISIBLE_DEVICES=4 python3 -W ignore main.py --dataset 'REDDIT-BINARY' --batch-size 128 --n-layer 2 --n-factor 4 --separate-encoder --use-unsup-loss
#CUDA_VISIBLE_DEVICES=7 python3 -W ignore main.py --dataset 'REDDIT-MULTI-5K' --batch-size 64 --n-factor 4 --separate-encoder --use-unsup-loss
#CUDA_VISIBLE_DEVICES=3 python3 -W ignore main.py --dataset 'COLLAB' --batch-size 48 --n-factor 4 --separate-encoder --use-unsup-loss


#CUDA_VISIBLE_DEVICES=1 python3 main.py --dataset 'PROTEINS' --batch-size 16 --n-factor 4 --separate-encoder --use-unsup-loss --n-percents 1
#CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset 'PROTEINS' --batch-size 16 --n-factor 4 --separate-encoder --use-unsup-loss --n-percents 5
#CUDA_VISIBLE_DEVICES=2 python3 main.py --dataset 'PROTEINS' --batch-size 16 --n-factor 4 --separate-encoder --use-unsup-loss --n-percents 7


#CUDA_VISIBLE_DEVICES=2 python3 main.py --dataset 'IMDB-BINARY' --batch-size 256 --n-factor 4 --separate-encoder --use-unsup-loss --n-percents 1
#CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset 'IMDB-BINARY' --batch-size 256 --n-factor 4 --separate-encoder --use-unsup-loss --n-percents 5
#CUDA_VISIBLE_DEVICES=7 python3 main.py --dataset 'IMDB-BINARY' --batch-size 16 --n-factor 4 --separate-encoder --use-unsup-loss --n-percents 7

#CUDA_VISIBLE_DEVICES=4 python3 -W ignore main.py --dataset 'REDDIT-BINARY' --batch-size 160 --n-factor 4 --separate-encoder --use-unsup-loss --n-percents 1
#CUDA_VISIBLE_DEVICES=6 python3 -W ignore main.py --dataset 'REDDIT-BINARY' --batch-size 160 --n-factor 4 --separate-encoder --use-unsup-loss --n-percents 5
#CUDA_VISIBLE_DEVICES=7 python3 -W ignore main.py --dataset 'REDDIT-BINARY' --batch-size 160 --n-factor 4 --separate-encoder --use-unsup-loss --n-percents 7

#CUDA_VISIBLE_DEVICES=3 python3 -W ignore main.py --dataset 'MUTAG' --batch-size 256 --n-factor 4 --separate-encoder --use-unsup-loss --n-percents 1
#CUDA_VISIBLE_DEVICES=5 python3 -W ignore main.py --dataset 'MUTAG' --batch-size 256 --n-factor 1 --separate-encoder --use-unsup-loss --n-percents 5
#CUDA_VISIBLE_DEVICES=3 python3 -W ignore main.py --dataset 'MUTAG' --batch-size 256 --n-factor 4 --separate-encoder --use-unsup-loss --n-percents 7

#CUDA_VISIBLE_DEVICES=4 python3 main.py --dataset 'PROTEINS' --separate-encoder --use-unsup-loss
