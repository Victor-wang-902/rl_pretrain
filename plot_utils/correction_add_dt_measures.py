import os
import pandas as pd
import json
import numpy as np

#target_returns = {"hopper": 3600, "ant": 5000, "walker2d": 5000, "halfcheetah": 6000}
target_returns = {"hopper_medium": 1677,
                    "hopper_medium-expert": 2465,
                    "hopper_medium-replay": 1443,
                    "ant_medium": 3055,
                    "ant_medium-expert": 3771,
                    "ant_medium-replay": 2608,
                    "walker2d_medium": 2931,
                    "walker2d_medium-replay": 2036,
                    "walker2d_medium-expert": 4320,
                    "halfcheetah_medium": 4455, 
                    "halfcheetah_medium-expert": 4947,
                    "halfcheetah_medium-replay": 3952,
}

def get_other_score_measures(path, target_number):
    # return a dictionary
    with open(path, "r") as f:
        df = pd.read_csv(f, delimiter="\t", header=0)
        test_returns_norm = df["TestEpNormRet"].to_numpy()
        n = test_returns_norm.shape[0]
        test_returns_norm_sorted = np.sort(test_returns_norm)
        res = ((df['TestEpRet'] > target_number).replace(False, np.NaN).replace(True, 1)).idxmax()
        if res is np.nan:
            res = -1
        print(res)
        d = {
            'best_5percent_normalized':test_returns_norm_sorted[-int(n *0.05):].mean(),
            'best_10percent_normalized':test_returns_norm_sorted[-int(n *0.1):].mean(),
            'best_25percent_normalized': test_returns_norm_sorted[-int(n *0.25):].mean(),
            'best_50percent_normalized': test_returns_norm_sorted[-int(n *0.5):].mean(),
            'best_100percent_normalized': test_returns_norm_sorted.mean(),
            'best_later_half_normalized': test_returns_norm[int(n * 0.5):].mean(),
            'best_last_four_normalized': test_returns_norm[-4:].mean(),
            'target_steps': int(df.loc[res, "Steps"])
        }
    return d

base_path = '../code/checkpoints'
# base_path = '../code/testonly'
for root, dirs, files in os.walk(base_path):
    if 'dt' in root or 'chibiT' in root or 'same' in root or 'random' in root:
        for dir in dirs:
            for keys in target_returns:
                if keys in dir:
                    target_number = target_returns[keys]
            # Go through every subfolder in this folder
            subfolder = os.path.join(root, dir)
            for file in os.listdir(subfolder):
                if file == 'progress.csv':
                    try:
                        extra_measures_dict = get_other_score_measures(os.path.join(subfolder, file), target_number)
                        #print(extra_measures_dict)
                        ex_new = os.path.join(subfolder, 'extra_new.json')
                        ex = os.path.join(subfolder, 'extra.json')
                        ex_to_use = ex
                        if os.path.exists(ex_new) or os.path.exists(ex):
                            if os.path.exists(ex_new):
                                ex_to_use = ex_new

                            # load extra.json, prefer newer version
                            print(ex_to_use)
                            with open(ex_to_use, 'r') as ex_file:
                                extra_dict = json.load(ex_file)

                            extra_dict.update(extra_measures_dict)

                            with open(ex_new, 'w') as ex_file:
                                json.dump(extra_dict, ex_file)
                    except Exception as e:
                        print(e)
