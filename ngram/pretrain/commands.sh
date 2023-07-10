singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "
cd /scratch/zw2374/public/can-wikipedia-help-offline-rl-old/code
source /ext3/env.sh
conda activate rblm
export PYTHONPATH=$PYTHONPATH:/scratch/zw2374/public/can-wikipedia-help-offline-rl-old/code
nvidia-smi
echo $PATH
echo $LD_LIBRARY_PATH
python pretrain/pretrain.py --embed_dim 64 --n_layer 1 --n_head 2 --outdir "chibiT_embed_dim256_n_layer4_n_head4" --num_steps 1000 --num_steps_per_save 100 --batch_size 2048
"

2 x rtx8000 size 0 ~16:30:00
2 x rtx8000 size 1 ~22:10:00
4 x v100 size 2 ~19:40:00
4 x a100 size 3 ~?



singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "
cd /scratch/zw2374/public/can-wikipedia-help-offline-rl-old/code
source /ext3/env.sh
conda activate rblm
export PYTHONPATH=$PYTHONPATH:/scratch/zw2374/public/can-wikipedia-help-offline-rl-old/code
nvidia-smi
echo $PATH
echo $LD_LIBRARY_PATH
accelerate launch --config_file /home/zw2374/.cache/huggingface/accelerate/duo_config.yaml pretrain/pretrain_dist.py --batch_size 2048 --embed_dim 128 --n_layer 3 --n_head 1 --outdir "chibiT_embed_dim128_n_layer3_n_head1_test" --data_size 0.01 --num_steps 100 --num_steps_per_save 20
"

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "
cd /scratch/zw2374/public/can-wikipedia-help-offline-rl-old/code
source /ext3/env.sh
conda activate rblm
export PYTHONPATH=$PYTHONPATH:/scratch/zw2374/public/can-wikipedia-help-offline-rl-old/code
nvidia-smi
echo $PATH
echo $LD_LIBRARY_PATH
accelerate launch --config_file /home/zw2374/.cache/huggingface/accelerate/duo_config.yaml pretrain/pretrain_dist.py --batch_size 32768 --embed_dim 128 --n_layer 3 --n_head 1 --outdir "chibiT_embed_dim128_n_layer3_n_head1" --data_size 0.01 --num_steps 100 --num_steps_per_save 1000
"