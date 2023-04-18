# list_of_env_names = []
# mujoco_tasks = ['halfcheetah', 'walker2d', 'hopper', 'ant']
# mujoco_dataset_types = ['random', 'medium', 'expert', 'medium-replay', 'medium-expert']
# for task in mujoco_tasks:
#     for dataset_type in mujoco_dataset_types:
#         list_of_env_names.append('%s-%s-v2' % (task, dataset_type))
# print(list_of_env_names)

MUJOCO_ALL = ['halfcheetah-random-v2', 'halfcheetah-medium-v2', 'halfcheetah-expert-v2',
              'halfcheetah-medium-replay-v2', 'halfcheetah-medium-expert-v2',
              'walker2d-random-v2', 'walker2d-medium-v2', 'walker2d-expert-v2',
              'walker2d-medium-replay-v2', 'walker2d-medium-expert-v2',
              'hopper-random-v2', 'hopper-medium-v2', 'hopper-expert-v2',
              'hopper-medium-replay-v2', 'hopper-medium-expert-v2',
              'ant-random-v2', 'ant-medium-v2', 'ant-expert-v2',
              'ant-medium-replay-v2', 'ant-medium-expert-v2']
MUJOCO_9 = ['halfcheetah-medium-v2', 'halfcheetah-medium-replay-v2', 'halfcheetah-medium-expert-v2',
            'walker2d-medium-v2', 'walker2d-medium-replay-v2', 'walker2d-medium-expert-v2',
            'hopper-medium-v2', 'hopper-medium-replay-v2', 'hopper-medium-expert-v2',
            ] # the ones prsented in IQL
