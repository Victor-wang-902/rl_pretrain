import os
import pandas as pd
import json
import numpy as np


def get_other_score_measures(path):
    # return a dictionary
    with open(path, "r") as f:
        df = pd.read_csv(f, delimiter="\t", header=0)
        test_returns_norm = df["TestEpNormRet"].to_numpy()
        n = test_returns_norm.shape[0]
        test_returns_norm_sorted = np.sort(test_returns_norm)
        d = {
            'best_5percent_normalized':test_returns_norm_sorted[-int(n *0.05):].mean(),
            'best_10percent_normalized':test_returns_norm_sorted[-int(n *0.1):].mean(),
            'best_25percent_normalized': test_returns_norm_sorted[-int(n *0.25):].mean(),
            'best_50percent_normalized': test_returns_norm_sorted[-int(n *0.5):].mean(),
            'best_100percent_normalized': test_returns_norm_sorted.mean(),
            'best_later_half_normalized': test_returns_norm[int(n*0.5):].mean(),
            'last_four_normalized': test_returns_norm[-1:-5:-1].mean(),
        }
    return d


base_path = '../code/checkpoints/sendbackcql'
for root, dirs, files in os.walk(base_path):
    if 'cqlr3n' in root:
        for dir in dirs:
            # Go through every subfolder in this folder
            subfolder = os.path.join(root, dir)
            for file in os.listdir(subfolder):
                if file == 'progress.csv':
                    try:
                        extra_measures_dict = get_other_score_measures(os.path.join(subfolder, file))
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

# import shutil
# for pE in [200, 150, 100, 50, 25]:
#     for pR in [1, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001]:
#         for e in ['hopper', 'walker2d', 'halfcheetah', 'ant']:
#             for d in ['medium','medium-replay','medium-expert']:
#                 folder = base_path + f'/cqlr3n_premdp_same_noproj_preRatio{pR}_preEp{pE}_l2_ns100_pt1_sameTrue_{e}_{d}'
#                 for s in [2048, 4096]:
#                     path = folder + f'/cqlr3n_premdp_same_noproj_preRatio{pR}_preEp{pE}_l2_ns100_pt1_sameTrue_{e}_{d}_s{s}'
#                     try:
#                         shutil.rmtree(path)
#                     except:
#                         print(path)
