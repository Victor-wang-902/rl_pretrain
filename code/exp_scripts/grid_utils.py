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
MUJOCO_4_ENVS = ['hopper', 'walker2d', 'halfcheetah', 'ant']
MUJOCO_12 = ['halfcheetah-medium-v2', 'halfcheetah-medium-replay-v2', 'halfcheetah-medium-expert-v2',
            'walker2d-medium-v2', 'walker2d-medium-replay-v2', 'walker2d-medium-expert-v2',
            'hopper-medium-v2', 'hopper-medium-replay-v2', 'hopper-medium-expert-v2', 'ant-medium-v2',
             'ant-medium-replay-v2', 'ant-medium-expert-v2']

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


def get_setting_dt(settings, setting_number, random_setting_seed=0, random_order=False):
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
    #print(values_list)
    for values in values_list:
        total *= len(values)
    max_job = total
    #print(total)
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

    return indexes, actual_setting, max_job, hyper2logname

def get_auto_exp_name(actual_setting, hyper2logname, exp_prefix=None, suffix_before_env_dataset=''):
    # suffix_before_env_dataset should be sth like _layer2
    # if use this, make sure there is the underscore before
    exp_name_full = exp_prefix
    for hyper, value in actual_setting.items():
        if isinstance(value, tuple):
            if hyper not in ['env', 'dataset', 'seed']:
                if exp_name_full is not None:
                    exp_name_full = exp_name_full + '_%s' % (hyper2logname[hyper] + str(value[1]))
                else:
                    exp_name_full = '%s' % (hyper2logname[hyper] + str(value[1]))
        else:
            if hyper not in ['env', 'dataset', 'seed']:
                if exp_name_full is not None:
                    exp_name_full = exp_name_full + '_%s' % (hyper2logname[hyper] + str(value))
                else:
                    exp_name_full = '%s' % (hyper2logname[hyper] + str(value))
    exp_name_full = exp_name_full + suffix_before_env_dataset + '_%s_%s' % (actual_setting['env'], actual_setting['dataset'])
    return exp_name_full

def get_auto_exp_name_sac(actual_setting, hyper2logname, exp_prefix=None, suffix_before_env_dataset=''):
    # suffix_before_env_dataset should be sth like _layer2
    # if use this, make sure there is the underscore before
    exp_name_full = exp_prefix
    for hyper, value in actual_setting.items():
        if hyper not in ['env_name', 'seed']:
            if exp_name_full is not None:
                exp_name_full = exp_name_full + '_%s' % (hyper2logname[hyper] + str(value))
            else:
                exp_name_full = '%s' % (hyper2logname[hyper] + str(value))
    exp_name_full = exp_name_full + suffix_before_env_dataset + '_%s' % (actual_setting['env_name'])
    return exp_name_full

def get_setting_and_exp_name_dt(settings, setting_number, exp_prefix=None, random_setting_seed=0, random_order=False):
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
            if exp_name_full is not None:
                exp_name_full = exp_name_full + '_%s' % (hyper2logname[hyper] + str(value))
            else:
                exp_name_full = '%s' % (hyper2logname[hyper] + str(value))
    exp_name_full = exp_name_full + '_%s_%s' % (actual_setting['env'], actual_setting['dataset'])

    return indexes, actual_setting, max_job, exp_name_full

