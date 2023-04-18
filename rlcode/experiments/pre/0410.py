import os
ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
ld_library_path += ':/workspace/.mujoco/mujoco210/bin:/usr/local/nvidia/lib:/usr/lib/nvidia'
os.environ['LD_LIBRARY_PATH'] = ld_library_path
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_PY_MUJOCO_PATH'] = '/workspace/.mujoco/mujoco210/'

from experiments.train_il import train_d4rl as function_to_run ## here make sure you import correct function
import time
from redq.utils.run_utils import setup_logger_kwargs
from experiments.grid_utils import get_setting_and_exp_name
from experiments.env_names import *

if __name__ == '__main__':
    import argparse
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=int, default=0)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    data_dir = '/code/data'

    exp_prefix = 'il'
    # parameter name - abbreviation - values
    settings = ['env_name','',MUJOCO_9,
                'seed','',[0, 1, 2],
                'do_pretrain','pre',[True, False],
                ]
    if args.debug:
        settings = settings + ['debug','debug',[True],]

    indexes, actual_setting, total, exp_name_full = get_setting_and_exp_name(settings, args.setting, exp_prefix)
    print("##### TOTAL NUMBER OF VARIANTS: %d #####" % total)

    logger_kwargs = setup_logger_kwargs(exp_name_full, actual_setting['seed'], data_dir)
    function_to_run(logger_kwargs=logger_kwargs, **actual_setting)
    print("Total time used: %.3f hours." % ((time.time() - start_time)/3600))

"""
before you submit the jobs:
- quick test your code to make sure things will run without bug
- compute the number of jobs, make sure that is consistent with the array number in the .sh file
- in the .sh file make sure you are running the correct python file 
"""
