#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=gm[001-024],gv[013-018]
##SBATCH --partition=aquila
#SBATCH --cpus-per-task=3
#SBATCH --mem=128GB
#SBATCH --job-name=pretrain
#SBATCH --mail-type=END
##SBATCH --mail-user=zw2374@nyu.edu
#SBATCH --time=48:00:00
##SBATCH --dependency=singleton
#SBATCH --output=pt_%j_wiki103_size_2.out
#SBATCH --error=pt_%j_wiki103_size_2.err
singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "
cd /code
conda activate rblm
export PYTHONPATH=$PYTHONPATH:/code
nvidia-smi
echo $PATH
echo $LD_LIBRARY_PATH
python pretrain/pretrain.py --embed_dim 128 --n_layer 3 --n_head 1 --outdir "gpu_test" --data_size 0.1 --num_steps 100 --num_steps_per_save 10
"