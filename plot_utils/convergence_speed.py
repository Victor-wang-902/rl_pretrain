import os
import pandas as pd
import json
import numpy as np


def get_convergence_time(path, target):
    with open(path, "r") as f:
        df = pd.read_csv(f, delimiter="\t", header=0)
        test_returns = df["TestEpRet"].to_numpy()
        d = {
            'convergence_update': np.argmax(test_returns >= target)
        }
    return d


all_env = {}
for env in ['halfcheetah', 'walker2d', 'hopper', 'ant']:
    for dataset in ['medium', 'medium-expert', 'medium-replay']:
        all_env[env+'_'+dataset] = None

base_path = '../code/checkpoints/final'
for setting in all_env.keys():
    for root, dirs, files in os.walk(base_path):
        if 'prenone' in root and setting in root:
            test_return = 0
            n_seeds = 0
            for dir in dirs:
                subfolder = os.path.join(root, dir)
                for file in os.listdir(subfolder):
                    if file == 'progress.csv':
                        try:
                            path = os.path.join(subfolder, file)
                            with open(path, 'r') as f:
                                df = pd.read_csv(f, delimiter="\t", header=0)
                                test_return += df["TestEpRet"].to_numpy()
                                n_seeds += 1
                        except Exception as e:
                            print(e)
            # when doing seed subfolder, this will be zero
            test_return = test_return / n_seeds
            all_env[setting] = test_return[-1:-5:-1].mean() * 0.9

print(all_env)
for setting in all_env.keys():
    for root, dirs, files in os.walk(base_path):
        if 'iclr_cqlr3n' in root and setting in root:
            target = all_env[setting]
            for dir in dirs:
                # Go through every subfolder in this folder
                    subfolder = os.path.join(root, dir)
                    for file in os.listdir(subfolder):
                        if file == 'progress.csv':
                            try:
                                extra_measures_dict = get_convergence_time(os.path.join(subfolder, file), target)
                                ex = os.path.join(subfolder, 'extra.json')
                                ex_to_use = ex
                                if os.path.exists(ex):
                                    # load extra.json
                                    print(ex_to_use)
                                    with open(ex_to_use, 'r') as ex_file:
                                        extra_dict = json.load(ex_file)

                                    extra_dict.update(extra_measures_dict)
                                    with open(ex_to_use, 'w') as ex_file:
                                        json.dump(extra_dict, ex_file)
                            except Exception as e:
                                print(e)
