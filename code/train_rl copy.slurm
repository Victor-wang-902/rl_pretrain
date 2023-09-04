#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
##SBATCH --exclude=agpu[1-6]
##SBATCH --partition=aquila
#SBATCH --cpus-per-task=3
#SBATCH --mem=128GB
#SBATCH --job-name=hopper_train
#SBATCH --mail-type=END
##SBATCH --mail-user=zw2374@nyu.edu
#SBATCH --time=48:00:00
##SBATCH --dependency=singleton
#SBATCH --output=dt_%j_finetune_75_7.out
#SBATCH --error=dt_%j_finetune_75_7.err
##python experiment.py --env hopper --dataset medium --model_type dt --seed 1024 --pretrained_lm gpt2  --outdir "checkpoints/gpt2_kmeans_medium_positions_hopper_1024" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/gpt2_lm_1000.pt" --gpt_kmeans_const 0.1 --dropout 0.2 --share_input_output_proj
##python experiment.py --env reacher2d --dataset medium --model_type dt --seed 666  --pretrained_lm gpt2  --outdir "checkpoints/gpt2_kmeans_medium_positions_reacher2d_666" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/gpt2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj
##python eval_model.py --env hopper --dataset medium --model_type dt --seed 666  --pretrained_lm gpt2 --load_checkpoint "/scratch/zw2374/public/can-wikipedia-help-offline-rl/code/checkpoints/gpt2_kmeans_medium_positions_hopper_666/model_3.pt"  --outdir "checkpoints/gpt2_kmeans_medium_positions_hopper_666_eval_again" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/gpt2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --env_targets "3600"

##python experiment.py --env hopper --dataset medium-expert --model_type dt --seed 666  --pretrained_lm chibiT  --outdir "checkpoints/cibiT_kmeans_medium_expert_positions_hopper_paper_666" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj
##python experiment.py --env hopper --dataset medium-expert --model_type dt --seed 42  --pretrained_lm chibiT  --outdir "checkpoints/cibiT_kmeans_medium_expert_positions_hopper_paper_42" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj
##python experiment.py --env hopper --dataset medium-expert --model_type dt --seed 1024  --pretrained_lm chibiT  --outdir "checkpoints/cibiT_kmeans_medium_expert_positions_hopper_paper_1024" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj

##python experiment.py --env walker2d --dataset medium --model_type dt --seed 666    --outdir "checkpoints/dt_kmeans_medium_positions_walker2d_paper_666" 
##python experiment.py --env walker2d --dataset medium --model_type dt --seed 42  --outdir "checkpoints/dt_kmeans_medium_positions_walker2d_paper_42"  
##python experiment.py --env walker2d --dataset medium --model_type dt --seed 1024  --outdir "checkpoints/dt_kmeans_medium_positions_walker2d_paper_1024"  

singularity exec --nv \
--overlay /scratch/zw2374/overlay-50G-10M-7.ext3:rw \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate wikirl-gym
nvidia-smi
echo $PATH
echo $LD_LIBRARY_PATH
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 666 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_halfcheetah_medium-expert_size_3_666" &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 24 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_halfcheetah_medium-expert_size_3_24" &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 1024 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_halfcheetah_medium-expert_size_3_1024" &
wait
"