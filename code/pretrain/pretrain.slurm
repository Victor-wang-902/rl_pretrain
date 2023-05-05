#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=gm[001-024],gv[013-018]
##SBATCH --partition=aquila
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --job-name=pretrain
#SBATCH --mail-type=END
##SBATCH --mail-user=zw2374@nyu.edu
#SBATCH --time=48:00:00
##SBATCH --dependency=singleton
#SBATCH --output=pt_%j_wiki103_size_3.out
#SBATCH --error=pt_%j_wiki103_size_3.err
##python pretrain/pretrain.py --embed_dim 512 --n_layer 6 --n_head 8 --outdir "chibiT_embed_dim512_n_layer6_n_head8"
##python pretrain/pretrain.py --embed_dim 256 --n_layer 4 --n_head 4 --outdir "chibiT_embed_dim256_n_layer4_n_head4"


singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "
cd /scratch/zw2374/public/can-wikipedia-help-offline-rl-old/code
source /ext3/env.sh
conda activate rblm
export PYTHONPATH=$PYTHONPATH:/scratch/zw2374/public/can-wikipedia-help-offline-rl-old/code
nvidia-smi
echo $PATH
echo $LD_LIBRARY_PATH
python pretrain/pretrain.py --embed_dim 768 --n_layer 12 --n_head 12 --outdir "chibiT_embed_dim768_n_layer12_n_head12"
"

