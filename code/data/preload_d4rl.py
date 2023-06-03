import gym
import d4rl
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='halfcheetah')
parser.add_argument('--dataset', type=str, default='medium')
args = parser.parse_args()

env_name = f"{args.env}-{args.dataset}-v2"
dataset = d4rl.qlearning_dataset(gym.make(env_name).unwrapped)
print("D4RL dataset loaded for", env_name)
