#!/bin/bash
##SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclude=gm[001-024],gv[013-018]
##SBATCH --partition=aquila
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --job-name=pretrain
#SBATCH --mail-type=END
##SBATCH --mail-user=zw2374@nyu.edu
#SBATCH --time=48:00:00
##SBATCH --dependency=singleton
#SBATCH --output=pt_%j_wiki103_size_2.out
#SBATCH --error=pt_%j_wiki103_size_2.err
##python pretrain/pretrain.py --embed_dim 512 --n_layer 6 --n_head 8 --outdir "chibiT_embed_dim512_n_layer6_n_head8"
##python pretrain/pretrain.py --embed_dim 256 --n_layer 4 --n_head 4 --outdir "chibiT_embed_dim256_n_layer4_n_head4"
## accelerate launch pretrain/pretrain_dist.py --batch_size 32768 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "chibiT_embed_dim256_n_layer4_n_head4"
## accelerate launch pretrain/pretrain_dist.py --batch_size 32768 --embed_dim 128 --n_layer 3 --n_head 1 --outdir "chibiT_embed_dim128_n_layer3_n_head1"
## accelerate launch pretrain/pretrain_dist.py --batch_size 16384 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "chibiT_embed_dim768_n_layer12_n_head12"
## accelerate launch pretrain/pretrain_dist.py --batch_size 16384 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "chibiT_embed_dim512_n_layer6_n_head8"


singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "
cd /scratch/zw2374/public/can-wikipedia-help-offline-rl-old/code
source /ext3/env.sh
conda activate rblm
export PYTHONPATH=$PYTHONPATH:/scratch/zw2374/public/can-wikipedia-help-offline-rl-old/code
nvidia-smi
echo $PATH
echo $LD_LIBRARY_PATH
accelerate launch --config_file /home/zw2374/.cache/huggingface/accelerate/quad_config.yaml pretrain/pretrain_dist.py --batch_size 16384 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "chibiT_embed_dim512_n_layer6_n_head8"
"

