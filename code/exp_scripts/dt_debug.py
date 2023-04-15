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
    setting_args, remaining_args = parser.parse_known_args()
    setting_id = setting_args.setting

    # arg parse for DT experiment
    dt_args = set_dt_args(remaining_args)
    variant = vars(dt_args)
    data_dir = '/train_logs'

    # parameter name - abbreviation - values
    settings = ['model_type', '', 'dt',
                'env', '', MUJOCO_3_ENVS,
                'dataset', '', MUJOCO_3_DATASETS,
                'seed', '', [42, 666, 1024],
                ]
    exp_prefix = 'dt'

    indexes, actual_setting, total, exp_name_full = get_setting_and_exp_name(settings, setting_id, exp_prefix)
    print("##### TOTAL NUMBER OF VARIANTS: %d #####" % total)

    # modify default dt parameters with ones specificed by the setting id
    for key, value in actual_setting.items():
        variant[key] = value

    logger_kwargs = setup_logger_kwargs(exp_name_full, actual_setting['seed'], data_dir)
    variant["outdir"] = logger_kwargs["outdir"]
    variant["exp_name"] = logger_kwargs["exp_name"]
    variant["device"] = 'cpu'

    experiment("gym-experiment", variant=variant)
    print("Total time used: %.3f hours." % ((time.time() - start_time)/3600))

"""
before you submit the jobs:
- quick test your code to make sure things will run without bug
- compute the number of jobs, make sure that is consistent with the array number in the .sh file
- in the .sh file make sure you are running the correct python file 
"""


