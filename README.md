
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

### grid experiments 

greene interactive session:
```
srun --pty --cpus-per-task=1 --mem 8000 -t 0-06:00 bash
srun --pty --gres=gpu:1 --cpus-per-task=4 --mem 8000 -t 0-06:00 bash

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
singularity exec --nv -B /scratch/$USER/sing/rl_pretrain/code:/code -B /scratch/$USER/sing/dt-sandbox/opt/conda/lib/python3.8/site-packages/mujoco_py/:/opt/conda/lib/python3.8/site-packages/mujoco_py/ -B /scratch/$USER/sing/rl_pretrain/code/checkpoints:/checkpoints /scratch/$USER/sing/dt-sandbox bash
```

### env variables
```
export PYTHONPATH=$PYTHONPATH:/code
cd /code
```


### runs

Very quick CPU debug run: 
```
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 666 --outdir "/checkpoints/debug_run_666" --device cpu --embed_dim 3 --max_iters 3 --num_steps_per_iter 10 --batch_size 4

```

Very quick CPU debug run for grid experiment: 
```
python exp_scripts/dt_debug.py --device cpu --embed_dim 3 --max_iters 3 --num_steps_per_iter 10 --batch_size 4

```


Send back files for plotting: 
```
cd /scratch/$USER/sing/rl_pretrain/code
rsync -av --exclude='*.pt' checkpoints/* sendback/
zip -r send.zip sendback/
```

### common debug commands
Check memory usage: 
```
ps -u $USER --no-headers -o rss | awk '{sum+=$1} END {printf "%.2f GB\n", sum/(1024*1024)}'
```

### data format
```
        # # data should have consistent format
        # dt_lr0.01_hopper_medium
        # - dt_lr0.01_hopper_medium_s42
        # - dt_lr0.01_hopper_medium_s666
        # - dt_lr0.01_hopper_medium_s1024
        #   - extra.json
        #   - progress.txt
        #      - TestEpRet
        #      - TestEpNormRet
        #      - Iter
        #      - Step

        # bc_lr0.01_hopper_medium

        """
        extra_dict = {
            'weight_diff':weight_diff,
            'feature_diff':feature_diff, # take 10% trajectory, seed 0
            'num_feature_data': 
            'final_test_returns':final_test_returns,
            'final_test_normalized_returns': final_test_normalized_returns,
            'best_return': best_return,
            'best_return_normalized':best_return_normalized,
            'convergence_step':convergence_step,
        }
        logger.save_extra_dict_as_json(extra_dict, 'extra.json')
        
        10 11 14 15 14 15 15 
        best normalized: 15 -> 
        """


```

### Plotting utils
Download test data for plotting: https://drive.google.com/file/d/1Alr_P4akkXuN3uAyImY8R58DUfcoTtpU/view?usp=sharing

Put data under `code/checkpoints/` (For example, you should see a folder here: `code/checkpoints/cpubase_dt_halfcheetah_medium`)

Run `plot_dt_test.py` and `pretrain_paper_table.py` to generate figures and latex table. 

## Offline RL experiments (CQL code)
Set up singularity:
```
singularity build --sandbox cql-sandbox docker://cwatcherw/cql:0.1
```
Run a CPU interactive job
```
srun --pty --cpus-per-task=1 --mem 8000 -t 0-04:00 bash
```

Or GPU interactive job
```
srun --pty --gres=gpu:1 --cpus-per-task=1 --mem 8000 -t 0-03:00 bash
```
Test with interactive job:
```
singularity exec --nv -B /scratch/$USER/sing/rl_pretrain/code:/code -B /scratch/$USER/sing/rl_pretrain/rlcode:/rlcode -B /scratch/$USER/sing/rl_pretrain/cqlcode:/cqlcode -B /scratch/$USER/sing/cql-sandbox/opt/conda/lib/python3.8/site-packages/mujoco_py/:/opt/conda/lib/python3.8/site-packages/mujoco_py/ -B /scratch/$USER/sing/rl_pretrain/code/checkpoints:/checkpoints /scratch/$USER/sing/cql-sandbox bash
```
After singulairty starts, run this to make path correct:
```
export PYTHONPATH=$PYTHONPATH:/code:/rlcode:/cqlcode
cd /cqlcode/exp/cqlr3
```

quick cql testing:
```
python run_cql_watcher.py --n_train_step_per_epoch 2 --n_epochs 22 --eval_n_trajs 1
```
quick cql testing with pretrain
```
python run_cql_watcher.py --pretrain_mode q_sprime --n_pretrain_epochs 3 --n_train_step_per_epoch 2 --n_epochs 22 --eval_n_trajs 1
```

For HPC grid jobs, for example, see `cqlcode/exp/cqlr3`. Here `cqlr3_base.py` and `cqlr3_base.sh` are used to run CQL baselines with and without pretraining, with random action = 3. 

Send back files for plotting: 
```
cd /scratch/$USER/sing/rl_pretrain/code
rsync -av --exclude='*.pt*' checkpoints/cqlr3* sendbackcql/
zip -r sendcql.zip sendbackcql/
```

### others
background no log job test:
```
python sth.py > /dev/null 2>&1 &
```

### new dt debug watcher
```
python experiment_watcherdebonly.py --env hopper --dataset medium --model_type dt --seed 666 --outdir "/checkpoints/ZZZdebug" --device cpu --embed_dim 3 --max_iters 3 --num_steps_per_iter 10 --batch_size 4 --calculate_extra

```