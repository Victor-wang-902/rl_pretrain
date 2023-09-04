import os
import pandas as pd
import json
import numpy as np
from infos import REF_MIN_SCORE, REF_MAX_SCORE

def get_normalized_score(env_name, score):
    name = env_name + "-medium-v0"
    return 100 * (score - REF_MIN_SCORE[name]) / (REF_MAX_SCORE[name] - REF_MIN_SCORE[name])


def get_other_score_measures(path):
    # return a dictionary

    with open(path, "r") as f:
        df = pd.read_csv(f, delimiter="\t", header=0)
        test_returns = df["current_itr_eval_5000_return_mean"].to_numpy()
        test_returns_norm = get_normalized_score("ant", df["current_itr_eval_5000_return_mean"].to_numpy())
        n = test_returns_norm.shape[0]
        test_returns_norm_sorted = np.sort(test_returns_norm)
        test_returns_sorted = np.sort(test_returns)
        d = {
            'best_5percent_normalized':test_returns_norm_sorted[-int(n *0.05):].mean(),
            'best_10percent_normalized':test_returns_norm_sorted[-int(n *0.1):].mean(),
            'best_25percent_normalized': test_returns_norm_sorted[-int(n *0.25):].mean(),
            'best_50percent_normalized': test_returns_norm_sorted[-int(n *0.5):].mean(),
            'best_100percent_normalized': test_returns_norm_sorted.mean(),
            'best_later_half_normalized': test_returns_norm[int(n * 0.5):].mean(),
            'best_last_four_normalized': test_returns_norm[-4:].mean(),
            'final_test_returns': test_returns[-1],
            'final_test_normalized_returns': test_returns_norm[-1],
            'best_return': test_returns_sorted[-1],
            'best_return_normalized': test_returns_norm_sorted[-1]
        }
    return d

base_path = '../code/checkpoints'
# base_path = '../code/testonly'
for root, dirs, files in os.walk(base_path):
    if 'dt' in root or 'chibiT' in root or 'same' in root:
        for dir in dirs:
            # Go through every subfolder in this folder
            subfolder = os.path.join(root, dir)
            for file in os.listdir(subfolder):
                if file == 'progress.csv':
                    try:
                        extra_measures_dict = get_other_score_measures(os.path.join(subfolder, file))
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
