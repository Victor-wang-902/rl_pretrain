#!/bin/bash
#SBATCH --verbose
#SBATCH --time=45:00:00
#SBATCH --nodes=1
#SBATCH --mem=12GB
#SBATCH --mail-type=ALL # select which email types will be sent
#SBATCH --mail-user=NETID@nyu.edu # NOTE: put your netid here if you want emails

#SBATCH --array=0-26 # here the number depends on number of tasks in the array, e.g. 0-11 will create 12 tasks
#SBATCH --output=logs/%A_%a.out # %A is SLURM_ARRAY_JOB_ID, %a is SLURM_ARRAY_TASK_ID
#SBATCH --error=logs/%A_%a.err

# #####################################################
# #SBATCH --gres=gpu:1 # uncomment this line to request a gpu

sleep $(( (RANDOM%10) + 1 )) # to avoid issues when submitting large amounts of jobs

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

echo "Job ID: ${SLURM_ARRAY_TASK_ID}"

singularity exec --nv -B /scratch/$USER/sing/combination/:/code -B /scratch/$USER/sing/d4rl-sandbox/opt/conda/lib/python3.8/site-packages/mujoco_py/:/opt/conda/lib/python3.8/site-packages/mujoco_py/ /scratch/$USER/sing/d4rl-sandbox bash -c "
cd /code/experiments/pre
export PYTHONPATH=$PYTHONPATH:/code
python 0410.py --setting ${SLURM_ARRAY_TASK_ID} --seed 0
"