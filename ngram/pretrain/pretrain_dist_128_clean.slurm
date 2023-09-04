#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=100GB
#SBATCH --job-name=s1_train
#SBATCH --mail-type=END
#SBATCH --time=48:00:00
#SBATCH --output=pt_%j.out
#SBATCH --error=pt_%j.err
singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "
cd /scratch/zw2374/public/can-wikipedia-help-offline-rl-old/ngram
source /ext3/env.sh
conda activate rblm
export PYTHONPATH=$PYTHONPATH:/scratch/zw2374/public/can-wikipedia-help-offline-rl-old/ngram
nvidia-smi
echo $PATH
echo $LD_LIBRARY_PATH
lsof -i :29500
accelerate launch --config_file /home/zw2374/.cache/huggingface/accelerate+/duo_config.yaml --main_process_port 29500 pretrain/pretrain_dist.py --dataset data_online_new_new/data_ngram_1_nvocab_1_temperature_1.0.csv --batch_size 32768 --embed_dim 128 --n_layer 3 --n_head 1 --outdir "chibiT_syn_embed_dim128_n_layer3_n_head1_ngram_1_nvocab_1_temperature_1.0"
"

