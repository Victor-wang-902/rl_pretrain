import argparse
import os

ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
ld_library_path += ':/workspace/.mujoco/mujoco210/bin:/usr/local/nvidia/lib:/usr/lib/nvidia'
os.environ['LD_LIBRARY_PATH'] = ld_library_path
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_PY_MUJOCO_PATH'] = '/workspace/.mujoco/mujoco210/'

import gym
import d4rl

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='halfcheetah')
parser.add_argument('--dataset', type=str, default='medium')
args = parser.parse_args()

env_name = f"{args.env}-{args.dataset}-v2"
dataset = d4rl.qlearning_dataset(gym.make(env_name).unwrapped)
print("D4RL dataset loaded for", env_name)
