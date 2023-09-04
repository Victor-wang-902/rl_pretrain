#!/bin/bash
#SBATCH --nodes=1
##SBATCH --gres=gpu:1
##SBATCH --exclude=gm[001-024],gv[013-018]
##SBATCH --partition=aquila
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --job-name=hopper_train
#SBATCH --mail-type=END
##SBATCH --mail-user=zw2374@nyu.edu
#SBATCH --time=1:00:00
##SBATCH --dependency=singleton
#SBATCH --output=logs/chibit_%j_extra.out
#SBATCH --error=logs/chibit_%j_extra.err
singularity exec --nv -B /scratch/$USER/public/can-wikipedia-help-offline-rl-old/code:/code -B /scratch/$USER/sing/dt-sandbox/opt/conda/lib/python3.8/site-packages/mujoco_py/:/opt/conda/lib/python3.8/site-packages/mujoco_py/ -B /scratch/$USER/public/can-wikipedia-help-offline-rl-old/code/checkpoints:/checkpoints /scratch/$USER/sing/dt-sandbox bash -c "
cd /code
export PYTHONPATH=$PYTHONPATH:/code
nvidia-smi
echo $PATH
echo $LD_LIBRARY_PATH
python experiment_new_ant.py --env walker2d --dataset medium-expert --model_type dt --seed 666 --outdir "random_same" --calculate_extra --device cpu
python experiment_new_ant.py --env walker2d --dataset medium-expert --model_type dt --seed 42 --outdir "random_same" --calculate_extra --device cpu
python experiment_new_ant.py --env walker2d --dataset medium-expert --model_type dt --seed 1024 --outdir "random_same" --calculate_extra --device cpu
python experiment_new_ant.py --env walker2d --dataset medium --model_type dt --seed 666 --outdir "random_same" --calculate_extra --device cpu
python experiment_new_ant.py --env walker2d --dataset medium --model_type dt --seed 42 --outdir "random_same" --calculate_extra --device cpu
python experiment_new_ant.py --env walker2d --dataset medium --model_type dt --seed 1024 --outdir "random_same" --calculate_extra --device cpu
"