import numpy as np
import time
import os.path as osp

FORCE_DATESTAMP = False
MUJOCO_ALL = ['halfcheetah-random-v2', 'halfcheetah-medium-v2', 'halfcheetah-expert-v2',
              'halfcheetah-medium-replay-v2', 'halfcheetah-medium-expert-v2',
              'walker2d-random-v2', 'walker2d-medium-v2', 'walker2d-expert-v2',
              'walker2d-medium-replay-v2', 'walker2d-medium-expert-v2',
              'hopper-random-v2', 'hopper-medium-v2', 'hopper-expert-v2',
              'hopper-medium-replay-v2', 'hopper-medium-expert-v2',
              'ant-random-v2', 'ant-medium-v2', 'ant-expert-v2',
              'ant-medium-replay-v2', 'ant-medium-expert-v2']
MUJOCO_3_ENVS = ['hopper', 'walker2d', 'halfcheetah',  ]
MUJOCO_3_DATASETS = ['medium','medium-replay','medium-expert',]
MUJOCO_9 = ['halfcheetah-medium-v2', 'halfcheetah-medium-replay-v2', 'halfcheetah-medium-expert-v2',
            'walker2d-medium-v2', 'walker2d-medium-replay-v2', 'walker2d-medium-expert-v2',
            'hopper-medium-v2', 'hopper-medium-replay-v2', 'hopper-medium-expert-v2',
            ] # the ones prsented in IQL


def setup_logger_kwargs_dt(exp_name, seed=None, data_dir=None, datestamp=False):
    # Datestamp forcing
    datestamp = datestamp or FORCE_DATESTAMP

    # Make base path
    ymd_time = time.strftime("%Y-%m-%d_") if datestamp else ''
    relpath = ''.join([ymd_time, exp_name])

    if seed is not None:
        # Make a seed-specific subfolder in the experiment directory.
        if datestamp:
            hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            subfolder = ''.join([hms_time, '-', exp_name, '_s', str(seed)])
        else:
            subfolder = ''.join([exp_name, '_s', str(seed)])
        relpath = osp.join(relpath, subfolder)

    data_dir = data_dir
    logger_kwargs = dict(output_dir=osp.join(data_dir, relpath),
                         exp_name=exp_name)
    return logger_kwargs


def get_setting_and_exp_name_dt(settings, setting_number, exp_prefix, random_setting_seed=0, random_order=False):
    np.random.seed(random_setting_seed)
    hypers, lognames, values_list = [], [], []
    hyper2logname = {}
    n_settings = int(len(settings)/3)
    for i in range(n_settings):
        hypers.append(settings[i*3])
        lognames.append(settings[i*3+1])
        values_list.append(settings[i*3+2])
        hyper2logname[hypers[-1]] = lognames[-1]

    total = 1
    for values in values_list:
        total *= len(values)
    max_job = total

    new_indexes = np.random.choice(total, total, replace=False) if random_order else np.arange(total)
    new_index = new_indexes[setting_number]

    indexes = []  ## this says which hyperparameter we use
    remainder = new_index
    for values in values_list:
        division = int(total / len(values))
        index = int(remainder / division)
        remainder = remainder % division
        indexes.append(index)
        total = division
    actual_setting = {}
    for j in range(len(indexes)):
        actual_setting[hypers[j]] = values_list[j][indexes[j]]

    exp_name_full = exp_prefix
    for hyper, value in actual_setting.items():
        if hyper not in ['env', 'dataset', 'seed']:
            exp_name_full = exp_name_full + '_%s' % (hyper2logname[hyper] + str(value))
    exp_name_full = exp_name_full + '_%s_%s' % (actual_setting['env'], actual_setting['dataset'])

    return indexes, actual_setting, max_job, exp_name_full

