#!/bin/bash
#SBATCH --nodes=1
##SBATCH --gres=gpu:1
##SBATCH --exclude=gm[001-024],gv[013-018]
##SBATCH --partition=aquila
#SBATCH --cpus-per-task=2
#SBATCH --mem=128GB
#SBATCH --job-name=hopper_train
#SBATCH --mail-type=END
##SBATCH --mail-user=zw2374@nyu.edu
#SBATCH --time=48:00:00
##SBATCH --dependency=singleton
#SBATCH --output=logs/generate_data.out
#SBATCH --error=logs/generate_data.err
singularity exec --nv -B /scratch/$USER/public/can-wikipedia-help-offline-rl-old/ngram:/ngram -B /scratch/$USER/sing/dt-sandbox/opt/conda/lib/python3.8/site-packages/mujoco_py/:/opt/conda/lib/python3.8/site-packages/mujoco_py/ -B /scratch/$USER/public/can-wikipedia-help-offline-rl-old/code/checkpoints:/checkpoints /scratch/$USER/sing/dt-sandbox bash -c "
cd /ngram
export PYTHONPATH=$PYTHONPATH:/code
nvidia-smi
echo $PATH
echo $LD_LIBRARY_PATH
python generate.py --nvocab 100 --ngram 3 --raw --outdir data
"