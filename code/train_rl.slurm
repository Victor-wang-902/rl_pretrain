#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx8000:1
##SBATCH --exclude=gm[001-025],gv[013-016]
##SBATCH --partition=aquila
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --job-name=speedtest
#SBATCH --mail-type=END
##SBATCH --mail-user=zw2374@nyu.edu
#SBATCH --time=10:00:00
##SBATCH --dependency=singleton
#SBATCH --output=logs/speed_test.out
#SBATCH --error=logs/speed_test.err
##python experiment.py --env hopper --dataset medium --model_type dt --seed 1024 --pretrained_lm gpt2  --outdir "checkpoints_ngram/gpt2_kmeans_medium_positions_hopper_1024" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/gpt2_lm_1000.pt" --gpt_kmeans_const 0.1 --dropout 0.2 --share_input_output_proj
##python experiment.py --env reacher2d --dataset medium --model_type dt --seed 666  --pretrained_lm gpt2  --outdir "checkpoints_ngram/gpt2_kmeans_medium_positions_reacher2d_666" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/gpt2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj
##python eval_model.py --env hopper --dataset medium --model_type dt --seed 666  --pretrained_lm gpt2 --load_checkpoint "/scratch/zw2374/public/can-wikipedia-help-offline-rl/code/checkpoints/gpt2_kmeans_medium_positions_hopper_666/model_3.pt"  --outdir "checkpoints/gpt2_kmeans_medium_positions_hopper_666_eval_again" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/gpt2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --env_targets "3600"

##python experiment.py --env hopper --dataset medium-expert --model_type dt --seed 666  --pretrained_lm chibiT  --outdir "checkpoints/cibiT_kmeans_medium_expert_positions_hopper_paper_666" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj
##python experiment.py --env hopper --dataset medium-expert --model_type dt --seed 42  --pretrained_lm chibiT  --outdir "checkpoints/cibiT_kmeans_medium_expert_positions_hopper_paper_42" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj
##python experiment.py --env hopper --dataset medium-expert --model_type dt --seed 1024  --pretrained_lm chibiT  --outdir "checkpoints/cibiT_kmeans_medium_expert_positions_hopper_paper_1024" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj

##python experiment.py --env walker2d --dataset medium --model_type dt --seed 666    --outdir "checkpoints/dt_kmeans_medium_positions_walker2d_paper_666" 
##python experiment.py --env walker2d --dataset medium --model_type dt --seed 42  --outdir "checkpoints/dt_kmeans_medium_positions_walker2d_paper_42"  
##python experiment.py --env walker2d --dataset medium --model_type dt --seed 1024  --outdir "checkpoints/dt_kmeans_medium_positions_walker2d_paper_1024"  
singularity exec --nv -B /scratch/$USER/public/can-wikipedia-help-offline-rl-old/code:/code -B /scratch/$USER/sing/dt-sandbox/opt/conda/lib/python3.8/site-packages/mujoco_py/:/opt/conda/lib/python3.8/site-packages/mujoco_py/ -B /scratch/$USER/public/can-wikipedia-help-offline-rl-old/code/checkpoints:/checkpoints /scratch/$USER/sing/dt-sandbox bash -c "
cd /code
export PYTHONPATH=$PYTHONPATH:/code
nvidia-smi
echo $PATH
echo $LD_LIBRARY_PATH
python experiment_new_ant.py --env ant --dataset medium-expert --model_type dt --seed 666 --outdir "speed_test_dt"

"