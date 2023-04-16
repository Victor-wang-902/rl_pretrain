import os
ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
ld_library_path += ':/workspace/.mujoco/mujoco210/bin:/usr/local/nvidia/lib:/usr/lib/nvidia'
os.environ['LD_LIBRARY_PATH'] = ld_library_path
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_PY_MUJOCO_PATH'] = '/workspace/.mujoco/mujoco210/'

from grid_utils import *
from experiment_new import set_dt_args, experiment
import time
import argparse

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=int, default=0)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    setting_args, remaining_args = parser.parse_known_args()
    setting_id = setting_args.setting

    # arg parse for DT experiment
    dt_args = set_dt_args(remaining_args)
    variant = vars(dt_args)
    data_dir = '/checkpoints'

    # each 3-tuple is:
    # parameter name - abbreviation - values (this has to be a list)
    exp_prefix = None
    # env, dataset and seed values will be added to the end of the folder name string
    # for each 3-tuple,
    settings = [
        'env', '', MUJOCO_3_ENVS,
        'dataset', '', MUJOCO_3_DATASETS,
        'model_type', '', ['dt',],
                ]

    indexes, actual_setting, total, hyper2logname = get_setting_dt(settings, setting_id)
    exp_name_full = get_auto_exp_name(actual_setting, hyper2logname, exp_prefix)
    if 'seed' not in actual_setting:
        actual_setting['seed'] = setting_args.seed

    print("##### TOTAL NUMBER OF VARIANTS: %d #####" % total)

    # modify default dt parameters with ones specificed by the setting id
    for key, value in actual_setting.items():
        variant[key] = value

    logger_kwargs = setup_logger_kwargs_dt(exp_name_full, actual_setting['seed'], data_dir)
    variant["outdir"] = logger_kwargs["output_dir"]
    variant["exp_name"] = logger_kwargs["exp_name"]

    experiment("gym-experiment", variant=variant)
    print("Total time used: %.3f hours." % ((time.time() - start_time)/3600))

"""
before you submit the jobs:
- quick test your code to make sure things will run without bug
- compute the number of jobs, make sure that is consistent with the array number in the .sh file
- in the .sh file make sure you are running the correct python file 

if doing parallel jobs on gpu, can for example run 3 seeds at the same time, 
add seed as command line argument
# 'seed', '', [42, 666, 1024],
if doing cpu jobs then run 1 seed each job, add seeds to settings
"""


