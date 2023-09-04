import os
import pandas as pd
import json

def get_correct_convergence(path):
    with open(path, "r") as f:
        df = pd.read_csv(f, delimiter="\t", header=0)
        final_test_returns = df["TestEpRet"].iloc[-1]
        final_test_normalized_returns = df["TestEpNormRet"].iloc[-1]
        best_return = max(df["TestEpRet"])
        best_return_normalized = max(df["TestEpNormRet"])
        convergence_iter = df["Iteration"].iloc[df["TestEpNormRet"].ge(best_return_normalized - 2.0).idxmax()]
        convergence_step = df["Steps"].iloc[df["TestEpNormRet"].ge(best_return_normalized - 2.0).idxmax()]
        best_step = df["Steps"][df["TestEpRet"] == best_return].iat[0]
        best_iter = df["Iteration"][df["TestEpRet"] == best_return].iat[0]
    return convergence_iter, convergence_step

base_path = '../code/checkpoints'
# base_path = '../code/testonly'
for root, dirs, files in os.walk(base_path):
    if 'dt' in root or 'chibiT' in root or "same" in root:
        for dir in dirs:
            # Go through every subfolder in this folder
            subfolder = os.path.join(root, dir)
            for file in os.listdir(subfolder):
                if file == 'progress.csv':
                    try:
                        iter, step = get_correct_convergence(os.path.join(subfolder, file))
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

                            print("correction:")
                            print(extra_dict['convergence_iter'], iter)
                            print(extra_dict['convergence_step'], step)

                            extra_dict['convergence_iter'] = int(iter)
                            extra_dict['convergence_step'] = int(step)

                            with open(ex_new, 'w') as ex_file:
                                json.dump(extra_dict, ex_file)
                    except Exception as e:
                        print(e)
