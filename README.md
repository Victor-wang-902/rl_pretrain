
# Can Wikipedia Help Offline RL? 

Machel Reid, Yutaro Yamada and Shixiang Shane Gu.

Our paper is up on [arXiv](https://arxiv.org/abs/2201.12122).

## Overview

Official codebase for [Can Wikipedia Help Offline Reinforcement Learning?](https://arxiv.org/abs/2201.12122).
Contains scripts to reproduce experiments. (This codebase is based on that of https://github.com/kzl/decision-transformer)

![image info](./architecture.png)

## Instructions

We provide code our `code` directory containing code for our experiments.
### Installation

Experiments require MuJoCo.
Follow the instructions in the [mujoco-py repo](https://github.com/openai/mujoco-py) to install.
Then, dependencies can be installed with the following command:

```
conda env create -f environment.yml
```

### Downloading datasets

Datasets are stored in the `data` directory. LM co-training and vision experiments can be found in `lm_cotraining` and `vision` directories respectively.
Install the [D4RL repo](https://github.com/rail-berkeley/d4rl), following the instructions there.
Then, run the following script in order to download the datasets and save them in our format:

```
python download_d4rl_datasets.py
```

### Downloading ChibiT

ChibiT can be downloaded with gdown as follows:
```bash
gdown --id 1-ziehUyca2eyu5sQRux_q8BkKCnHqOn1
```

### Example usage

Once downloaded datasets and necessary checkpoints, modify example commands in `run.sh` file to reproduce or perturb weights.

To run the commands on Greene, an `ext3` overlay install with all the dependencies is needed for the Singularity container. 
For example, example commands can be run with: 
```
singularity exec --nv \
--overlay /path/to/your/overlay.ext3:rw \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate wikirl-gym
python experiment.py --env hopper --dataset medium --model_type dt --seed 666  --pretrained_lm chibiT  --outdir "checkpoints/cibiT_kmeans_medium_positions_hopper_perturb_8e0_666" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0
python experiment.py --env hopper --dataset medium --model_type dt --seed 42  --pretrained_lm chibiT  --outdir "checkpoints/cibiT_kmeans_medium_positions_hopper_perturb_8e0_42" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0
python experiment.py --env hopper --dataset medium --model_type dt --seed 1024  --pretrained_lm chibiT  --outdir "checkpoints/cibiT_kmeans_medium_positions_hopper_perturb_8e0_1024" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0
"
```
## Citation


```
@misc{reid2022wikipedia,
      title={Can Wikipedia Help Offline Reinforcement Learning?}, 
      author={Machel Reid and Yutaro Yamada and Shixiang Shane Gu},
      year={2022},
      eprint={2201.12122},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## License

MIT


### watcher test 

greene interactive session:
```
srun --pty --cpus-per-task=1 --mem 8000 -t 0-06:00 bash
srun --pty --gres=gpu:1 --cpus-per-task=2 --mem 8000 -t 0-06:00 bash

```


### Set up DT sandbox
Set up
```
module load singularity # not needed on greene
cd /scratch/$USER/sing/
singularity build --sandbox dt-sandbox docker://cwatcherw/dt:0.3
```

### Run
```
cd /scratch/$USER/sing
singularity exec --nv -B /scratch/$USER/sing/rl_pretrain/code:/code -B /scratch/$USER/sing/dt-sandbox/opt/conda/lib/python3.8/site-packages/mujoco_py/:/opt/conda/lib/python3.8/site-packages/mujoco_py/ /scratch/$USER/sing/dt-sandbox bash
```

### env variables
```
export PYTHONPATH=$PYTHONPATH:/code
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/.mujoco/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=/workspace/.mujoco/mujoco210/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nvidia/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_GL=egl
cd /code
```


### runs
```
python experiment.py --env hopper --dataset medium --model_type dt --seed 666  --outdir "checkpoints/cibiT_kmeans_medium_positions_hopper_perturb_8e0_666" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0

python experiment.py --env hopper --dataset medium --model_type dt --seed 666 --outdir "checkpoints/cibiT_kmeans_medium_positions_hopper_perturb_8e0_666"
--

```
