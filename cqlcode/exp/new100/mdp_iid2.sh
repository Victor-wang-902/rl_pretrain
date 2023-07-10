#!/bin/bash
#SBATCH --verbose
#SBATCH -p aquila,parallel # for sh cluster
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=8GB
#SBATCH --mail-type=ALL # select which email types will be sent
#SBATCH --mail-user=cw1681@nyu.edu # NOTE: put your netid here if you want emails

#SBATCH --array=0-62 # here the number depends on number of tasks in the array, e.g. 0-11 will create 12 tasks
#SBATCH --output=../logs/%A_%a.out # %A is SLURM_ARRAY_JOB_ID, %a is SLURM_ARRAY_TASK_ID,
#SBATCH --error=../logs/%A_%a.err # MAKE SURE WHEN YOU RUN THIS, ../logs IS A VALID PATH

# #####################################################
# #SBATCH --gres=gpu:1 # uncomment this line to request a gpu
# #SBATCH --cpus-per-task=4
#SBATCH --constraint=g6132 # for sh cluster g6148,g6132,g6248

sleep $(( (RANDOM%10) + 1 )) # to avoid issues when submitting large amounts of jobs

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

echo "Job ID: ${SLURM_ARRAY_TASK_ID}"

singularity exec --nv -B /scratch/$USER/sing/rl_pretrain/code:/code -B /scratch/$USER/sing/rl_pretrain/rlcode:/rlcode -B /scratch/$USER/sing/rl_pretrain/cqlcode:/cqlcode -B /scratch/$USER/sing/cql-sandbox/opt/conda/lib/python3.8/site-packages/mujoco_py/:/opt/conda/lib/python3.8/site-packages/mujoco_py/ -B /scratch/$USER/sing/rl_pretrain/code/checkpoints:/checkpoints /scratch/$USER/sing/cql-sandbox bash -c "
cd /cqlcode
export PYTHONPATH=$PYTHONPATH:/code:/rlcode:/cqlcode
python exp/new100/mdp_iid2.py --setting ${SLURM_ARRAY_TASK_ID}
"