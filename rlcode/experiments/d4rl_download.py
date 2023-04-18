import gym
import d4rl

"""code to pre-download d4rl datasets"""
list_of_env_names = []

mujoco_tasks = ['halfcheetah', 'walker2d', 'hopper', 'ant']
mujoco_dataset_types = ['random', 'medium', 'expert', 'medium-replay', 'medium-expert']
for task in mujoco_tasks:
    for dataset_type in mujoco_dataset_types:
        list_of_env_names.append('%s-%s-v2' % (task, dataset_type))


maze_tasks = ['antmaze']
maze_dataset_types = ['umaze','umaze-diverse','medium-diverse',
                      'medium-play','large-diverse','large-play']
for task in maze_tasks:
    for dataset_type in maze_dataset_types:
        list_of_env_names.append('%s-%s-v0' % (task, dataset_type))

adroit_tasks = ['pen','hammer','door','relocate',]
adroit_dataset_types = ['human', 'cloned', 'expert']
for task in adroit_tasks:
    for dataset_type in adroit_dataset_types:
        list_of_env_names.append('%s-%s-v0' % (task, dataset_type))

kitchen_tasks = ['kitchen']
kitchen_dataset_types = ['complete', 'partial', 'mixed']
for task in kitchen_tasks:
    for dataset_type in kitchen_dataset_types:
        list_of_env_names.append('%s-%s-v0' % (task, dataset_type))

for env_name in list_of_env_names:
    try:
        env = gym.make(env_name)
        dataset = d4rl.qlearning_dataset(env)
    except Exception as e:
        print("Error: %s dataset not loaded." % env_name)
