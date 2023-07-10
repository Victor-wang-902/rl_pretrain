#!/bin/bash
#SBATCH --verbose
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=8GB
##SBATCH --mail-type=ALL # select which email types will be sent
##SBATCH --mail-user=NETID@nyu.edu # NOTE: put your netid here if you want emails

#SBATCH --array=0-80 # here the number depends on number of tasks in the array, e.g. 0-11 will create 12 tasks
#SBATCH --output=../logs/%A_%a.out # %A is SLURM_ARRAY_JOB_ID, %a is SLURM_ARRAY_TASK_ID,
#SBATCH --error=../logs/%A_%a.err # MAKE SURE WHEN YOU RUN THIS, ../logs IS A VALID PATH

# #####################################################
# #SBATCH --gres=gpu:1 # uncomment this line to request a gpu
# #SBATCH --cpus-per-task=4

# sleep $(( (RANDOM%10) + 1 )) # to avoid issues when submitting large amounts of jobs
sleep $((SLURM_ARRAY_TASK_ID*10))

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

echo "Job ID: ${SLURM_ARRAY_TASK_ID}"

singularity exec --nv -B /scratch/$USER/rl_pretrain/code:/code -B /scratch/$USER/rl_pretrain/rlcode:/rlcode -B /scratch/$USER/rl_pretrain/cqlcode:/cqlcode -B /scratch/$USER/cql-sandbox/opt/conda/lib/python3.8/site-packages/mujoco_py/:/opt/conda/lib/python3.8/site-packages/mujoco_py/ -B /scratch/$USER/rl_pretrain/code/checkpoints:/checkpoints /scratch/$USER/cql-sandbox bash -c "
cd /cqlcode
export PYTHONPATH=$PYTHONPATH:/code:/rlcode:/cqlcode
python exp/cqlr3/cqlr3_base.py --setting ${SLURM_ARRAY_TASK_ID}
"


##!/bin/bash
##SBATCH --verbose
##SBATCH --time=48:00:00
##SBATCH --nodes=1
##SBATCH --mem=32GB
##SBATCH --cpus-per-task=4
##SBATCH --gres=gpu:1 # uncomment this line to request a gpu
#
##SBATCH --output=./logs/%A_%a.out # %A is SLURM_ARRAY_JOB_ID, %a is SLURM_ARRAY_TASK_ID,
##SBATCH --error=./logs/%A_%a.err # MAKE SURE WHEN YOU RUN THIS, ../logs IS A VALID PATH
#
#singularity exec --nv --bind /scratch/$USER --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro /scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
#source /ext3/env.sh;
#conda activate dt-atari;
#python run_dt_atari.py --seed 666 --context_length 30 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Breakout' --batch_size 128
#"
