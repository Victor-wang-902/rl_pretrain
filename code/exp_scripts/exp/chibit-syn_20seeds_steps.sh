#!/bin/bash
#SBATCH --verbose
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH --mail-type=ALL # select which email types will be sent
#SBATCH --mail-user=zw2374@nyu.edu # NOTE: put your netid here if you want emails

#SBATCH --array=0-479 # here the number depends on number of tasks in the array, e.g. 0-11 will create 12 tasks
#SBATCH --output=logs/datasize%A_%a.out # %A is SLURM_ARRAY_JOB_ID, %a is SLURM_ARRAY_TASK_ID
#SBATCH --error=logs/datasize%A_%a.err

# #####################################################
# #SBATCH --gres=gpu:1 # uncomment this line to request a gpu

sleep $(( (RANDOM%10) + 1 )) # to avoid issues when submitting large amounts of jobs

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

echo "Job ID: ${SLURM_ARRAY_TASK_ID}"

singularity exec --nv -B /scratch/$USER/public/can-wikipedia-help-offline-rl-old/code:/code -B /scratch/$USER/sing/dt-sandbox/opt/conda/lib/python3.8/site-packages/mujoco_py/:/opt/conda/lib/python3.8/site-packages/mujoco_py/ -B /scratch/$USER/public/can-wikipedia-help-offline-rl-old/code/checkpoints:/checkpoints /scratch/$USER/sing/dt-sandbox bash -c "
cd /code
export PYTHONPATH=$PYTHONPATH:/code
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/.mujoco/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=/workspace/.mujoco/mujoco210/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nvidia/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_GL=egl
python exp_scripts/exp/chibit-syn_20seeds_steps.py --setting ${SLURM_ARRAY_TASK_ID} --device cpu --extend_positions --dropout 0.2 --share_input_output_proj
"