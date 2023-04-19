import os.path
import numpy as np
import pandas as pd
import json

# def get_final_performance_seeds(datafolder_path):
#     if not os.path.exists(datafolder_path):
#         raise FileNotFoundError("Path does not exist: %s" % datafolder_path)
#     # return a list, each entry is the final performance of a seed
#     performance_list = []
#     for subdir, dirs, files in os.walk(datafolder_path):
#         if 'progress.txt' in files:
#             # load progress file for this seed
#             progress_file_path = os.path.join(subdir, 'progress.txt')
#         elif 'progress.csv' in files:
#             progress_file_path = os.path.join(subdir, 'progress.csv')
#         else:
#             continue
#         df = pd.read_table(progress_file_path)
#         final_performance = df['TestEpNormRet'].tail(2).mean()
#         performance_list.append(final_performance)
#     return performance_list

base_measures = ['final_test_returns', 'final_test_normalized_returns',
                'best_return', 'best_return_normalized',
                'final_weight_diff', 'final_feature_diff',
                'best_weight_diff', 'best_feature_diff',
                'convergence_step', 'convergence_iter',
                'best_step', 'best_iter']
all_measures = ['final_test_returns', 'final_test_normalized_returns',
                'best_return', 'best_return_normalized',
                'final_weight_diff', 'final_feature_diff',
                'best_weight_diff', 'best_feature_diff',
                'convergence_step', 'convergence_iter',
                'best_step', 'best_iter',
                'final_test_returns_std', 'final_test_normalized_returns_std',
                'best_return_std', 'best_return_normalized_std',
                ]

def get_extra_dict_multiple_seeds(datafolder_path):
    # for a alg-dataset variant, obtain a dictionary with key-value pairs as measure:value_for_each_seed
    if not os.path.exists(datafolder_path):
        raise FileNotFoundError("Path does not exist: %s" % datafolder_path)
    # return a list, each entry is the final performance of a seed
    aggregate_dict = {}
    measures = base_measures
    for measure in measures:
        aggregate_dict[measure] = []

    for subdir, dirs, files in os.walk(datafolder_path):
        if 'extra.json' in files:
            extra_dict_file_path = os.path.join(subdir, 'extra.json')
            with open(extra_dict_file_path, 'r') as file:
                extra_dict = json.load(file)
                for measure in measures:
                    aggregate_dict[measure].append(float(extra_dict[measure]))
    for measure in measures:
        aggregate_dict[measure] = [np.mean(aggregate_dict[measure]), np.std(aggregate_dict[measure])]
    for measure in ['final_test_returns', 'final_test_normalized_returns', 'best_return', 'best_return_normalized']:
        aggregate_dict[measure + '_std'] = [aggregate_dict[measure][1],]
    return aggregate_dict

data_path = '../../code/checkpoints/'

MUJOCO_3_ENVS = ['hopper', 'walker2d', 'halfcheetah',  ]
MUJOCO_3_DATASETS = ['medium','medium-replay','medium-expert',]
envs = []
for e in MUJOCO_3_ENVS:
    for dataset in MUJOCO_3_DATASETS:
        envs.append('%s_%s' % (e, dataset))

# final table: for each variant name, for each measure, compute relevant values
alg_dataset_dict = {}
algs = ['cpubase_dt', 'rl_il']
for alg in algs:
    alg_dataset_dict[alg] = {}
    # compute a number of measures
    # 1. performance
    performance_list = []
    performance_std_list = []
    for env in envs:
        folderpath = os.path.join(data_path, '%s_%s' % (alg, env))
        alg_dataset_dict[alg][env] = get_extra_dict_multiple_seeds(folderpath)

# TODO compute performance gain from pretraining for ones use pretraining (compared to no pretrain baseline)

def get_aggregated_value(alg_dataset_dict, alg, measure):
    value_list = []
    for dataset, extra_dict in alg_dataset_dict[alg].items():
        value_list.append(extra_dict[measure][0]) # each entry is the value from a dataset
    return np.mean(value_list), np.std(value_list)

"""table generation"""
def generate_aggregate_table(max_value_bold=True, bold_threshold=0.95):
    print("\nNow generate latex table:\n")
    # each row is a measure, each column is an algorithm variant
    rows = ['final_test_normalized_returns', 'best_return_normalized',
            'final_test_normalized_returns_std', 'best_return_normalized_std',
            'final_weight_diff', 'best_weight_diff',
            'final_feature_diff', 'best_feature_diff',
            'convergence_step',
            ]
    row_names = ['Final Score', 'Best Score',
                 'Final Std over Seeds',  'Best Std over Seeds',
                 'Final Weight Diff', 'Best Weight Diff',
                 'Final Feature Diff', 'Best Feature Diff',
                 'Convergence Step']

    table = np.zeros((2, len(rows), len(algs)))
    # each iter we generate a row
    for i, row in enumerate(rows):
        for j, alg in enumerate(algs):
            table[0,i,j], table[1,i,j] = get_aggregated_value(alg_dataset_dict, alg, row)

    max_values = np.max(table[0], axis=1)

    for i, row_name in enumerate(row_names):
        row_string = row_name
        for j in range(len(algs)):
            mean, std = table[0, i, j], table[1, i, j]
            if max_value_bold and mean > bold_threshold*max_values[i]:
                row_string += (' & \\textbf{%.1f} $\pm$ %.1f' % (mean, std))
            else:
                row_string += (' & %.1f $\pm$ %.1f' % (mean, std))
        row_string += '\\\\'
        print(row_string)


def generate_per_env_score_table(max_value_bold=True, bold_threshold=0.95):
    print("\nNow generate latex table:\n")
    measure = 'best_return_normalized'
    # each row is a env-dataset pair, each column is an algorithm variant
    rows = []
    row_names = []
    for dataset in [ 'medium-expert', 'medium', 'medium-replay', ]:
        for e in ['halfcheetah', 'hopper', 'walker2d', ]:
            rows.append('%s_%s' % (e, dataset))
            row_names.append('%s-%s' % (e, dataset))


    table = np.zeros((2, len(rows), len(algs)))
    # each iter we generate a row
    for i, row in enumerate(rows):
        for j, alg in enumerate(algs):
            table[0,i,j], table[1,i,j] = alg_dataset_dict[alg][row][measure]

    max_values = np.max(table[0], axis=1)

    for i, row_name in enumerate(row_names):
        row_string = row_name
        for j in range(len(algs)):
            mean, std = table[0, i, j], table[1, i, j]
            if max_value_bold and mean > bold_threshold*max_values[i]:
                row_string += (' & \\textbf{%.1f} $\pm$ %.1f' % (mean, std))
            else:
                row_string += (' & %.1f $\pm$ %.1f' % (mean, std))
        row_string += '\\\\'
        print(row_string)

generate_aggregate_table()

generate_per_env_score_table()