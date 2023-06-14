import os.path
import numpy as np
import pandas as pd
import json
# use this to generate the main table

base_measures = ['best_return_normalized', 'best_return',
                 'final_test_returns', 'final_test_normalized_returns',
        'best_weight_diff',
        'best_weight_sim',

        'best_feature_diff',
        'best_feature_sim',

        "best_0_weight_diff",
        "best_1_weight_diff",
        "best_0_weight_sim",
        "best_1_weight_sim",

        'convergence_iter',

         'final_feature_diff',
         'final_weight_diff',
         'final_feature_sim',
         'final_weight_sim',
                 ]

def get_extra_dict_multiple_seeds(datafolder_path):
    # for a alg-dataset variant, obtain a dictionary with key-value pairs as measure:[avg across seeds, std across seeds]
    if not os.path.exists(datafolder_path):
        raise FileNotFoundError("Path does not exist: %s" % datafolder_path)
    # return a list, each entry is the final performance of a seed
    aggregate_dict = {}
    measures = base_measures
    for measure in measures:
        aggregate_dict[measure] = []
    aggregate_dict['weight_diff_100k'] = [] # TODO might want to extend later...
    aggregate_dict['feature_diff_100k'] = []

    for subdir, dirs, files in os.walk(datafolder_path):
        if 'extra_new.json' in files or 'extra.json' in files:
            if 'extra_new.json' in files:
                extra_dict_file_path = os.path.join(subdir, 'extra_new.json')
            else:
                extra_dict_file_path = os.path.join(subdir, 'extra.json')

            with open(extra_dict_file_path, 'r') as file:
                extra_dict = json.load(file)
                for measure in measures:
                    aggregate_dict[measure].append(float(extra_dict[measure]))
                # if 'weight_diff_100k' not in extra_dict:
                #     aggregate_dict['weight_diff_100k'].append(float(extra_dict['final_weight_diff']))
                #     aggregate_dict['feature_diff_100k'].append(float(extra_dict['final_feature_diff']))
                # else:
                #     print(extra_dict['feature_diff_100k'])
                #     aggregate_dict['weight_diff_100k'].append(float(extra_dict['weight_diff_100k']))
                #     aggregate_dict['feature_diff_100k'].append(float(extra_dict['feature_diff_100k']))

    for measure in measures:
        if len(aggregate_dict[measure]) == 0:
            print(datafolder_path, 'has nothing for measure:',measure)
        aggregate_dict[measure] = [np.mean(aggregate_dict[measure]), np.std(aggregate_dict[measure])]
    for measure in ['final_test_returns', 'final_test_normalized_returns', 'best_return', 'best_return_normalized']:
        aggregate_dict[measure + '_std'] = [aggregate_dict[measure][1],]
    return aggregate_dict

data_path = '../../code/checkpoints/'

MUJOCO_3_ENVS = [
                'hopper',
                 'halfcheetah',
                'walker2d',
]
MUJOCO_3_DATASETS = ['medium','medium-replay','medium-expert',]
all_envs = []
for e in MUJOCO_3_ENVS:
    for dataset in MUJOCO_3_DATASETS:
        all_envs.append('%s_%s' % (e, dataset))

def get_alg_dataset_dict(algs, envs):
    # load extra dict for all alg, all envs, all seeds
    alg_dataset_dict = {}
    for alg in algs:
        alg_dataset_dict[alg] = {}
        for env in envs:
            folderpath = os.path.join(data_path, '%s_%s' % (alg, env))
            alg_dataset_dict[alg][env] = get_extra_dict_multiple_seeds(folderpath)
    return alg_dataset_dict

def get_aggregated_value(alg_dataset_dict, alg, measure):
    # for an alg-measure pair, aggregate over all datasets
    value_list = []
    for dataset, extra_dict in alg_dataset_dict[alg].items():
        value_list.append(extra_dict[measure][0]) # each entry is the value from a dataset
    return np.mean(value_list), np.std(value_list)

"""table generation"""
def generate_aggregate_table(algs, alg_dataset_dict, column_names, best_value_bold=True, bold_threshold=0.05):
    print("\nNow generate latex table:\n")
    # each row is a measure, each column is an algorithm variant
    rows = [
        'best_return_normalized',
        'best_return_normalized_std',

        'best_feature_diff',
        'best_weight_diff',
        "best_0_weight_diff",
        "best_1_weight_diff",

        'best_feature_sim',
        'best_weight_sim',
        "best_0_weight_sim",
        "best_1_weight_sim",

        'convergence_iter',

        'final_test_normalized_returns',
        'final_feature_diff',
        'final_weight_diff',
        'final_feature_sim',
        'final_weight_sim',
    ]
    row_names = ['Best Score',
                 'Best Std over Seeds',

                 'Best Feature Diff',
                 'Best Weight Diff',
                 'Best Weight Diff L0',
                 'Best Weight Diff L1',

                 'Best Feature Sim',
                 'Best Weight Sim',
                 'Best Weight Sim L0',
                 'Best Weight Sim L1',

                 'Convergence Iter',

                 'Final Score',
                 'Final Feature Diff',
                 'Final Weight Diff',
                 'Final Feature Sim',
                 'Final Weight Sim',
                 ]
    row_names_higher_is_better = [
        'Best Score',
        'Final Score',
        'Best Weight Sim',
        'Best Feature Sim',
        'Best Weight Sim L0',
        'Best Weight Sim L1',
        'Best Weight Sim FC',
        'Final Feature Sim',
        'Final Weight Sim',
        'Prev Feature Sim',
        'Prev Weight Sim',
    ]

    table = np.zeros((2, len(rows), len(algs)))
    # each iter we generate a row
    for i, row in enumerate(rows):
        for j, alg in enumerate(algs):
            table[0,i,j], table[1,i,j] = get_aggregated_value(alg_dataset_dict, alg, row)

    max_values = np.max(table[0], axis=1)
    min_values = np.min(table[0], axis=1)

    col_name_line = ''
    for col in column_names:
        col_name_line += col +' & '
    col_name_line = col_name_line[:-2] + '\\\\'
    print(col_name_line)
    print("		\\hline ")
    for i, row_name in enumerate(row_names):
        row_string = row_name
        for j in range(len(algs)):
            mean, std = table[0, i, j], table[1, i, j]
            bold = False
            if best_value_bold:
                if row_name not in row_names_higher_is_better:
                    if mean <= (1+bold_threshold)*min_values[i]:
                        bold = True
                else:
                    if mean >= (1-bold_threshold)*max_values[i]:
                        bold = True
                if bold:
                    if 'Prev' in row_name:
                        row_string += (' & \\textbf{%.6f}' % (mean,))
                    elif row_name in ['Best Score', 'Best Std over Seeds', 'Convergence Iter']:
                        row_string += (' & \\textbf{%.1f} $\pm$ %.1f' % (mean, std))
                    else:
                        row_string += (' & \\textbf{%.3f} $\pm$ %.1f' % (mean, std))
                else:
                    if 'Prev' in row_name:
                        row_string += (' & %.6f' % (mean,))
                    elif row_name in ['Best Score', 'Best Std over Seeds', 'Convergence Iter']:
                        row_string += (' & %.1f $\pm$ %.1f' % (mean, std))
                    else:
                        row_string += (' & %.3f $\pm$ %.1f' % (mean, std))
        row_string += '\\\\'
        print(row_string)


def generate_per_env_score_table(max_value_bold=True, bold_threshold=0.95):
    # TODO need to fix the bold thing
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



def generate_table_nvocab_markov_chain():
    #################### table 1
    # DT table, 1-step markov chain, change the number of vocab
    algs = [
    'dt-rerun-data_size_dt_1.0',
        'chibiT-rerun',
        'chibiT-rerun-syn_ngram1_nvocab10_temperature1.0',
        'chibiT-rerun-syn_ngram1_nvocab100_temperature1.0',
        'chibiT-rerun-syn_ngram1_nvocab1000_temperature1.0',
        'chibiT-rerun-syn_ngram1_nvocab10000_temperature1.0',
        'chibiT-rerun-syn_ngram5_nvocab50257_temperature1.0',
    ]
    col_names = ['Measures', 'DT', 'ChibiT', '1-MC Voc 10','1-MC Voc 100','1-MC Voc 1000','1-MC Voc 10000', '5-MC voc 50257']
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_aggregate_table(algs, alg_dataset_dict, col_names)


def generate_table_nstep_markov_chain():
    #################### table 2
    # DT table, pretrain with markov chain data change number of step
    algs = [
    'dt-rerun-data_size_dt_1.0',
        'chibiT-rerun',
        # 'chibiT-rerun-syn_ngram1_nvocab50257_temperature1.0',
        # 'chibiT-rerun-syn_ngram2_nvocab50257_temperature1.0',
        # 'chibiT-rerun-syn_ngram3_nvocab50257_temperature1.0',
        'chibiT-rerun-syn_ngram4_nvocab50257_temperature1.0',
        'chibiT-rerun-syn_ngram5_nvocab50257_temperature1.0',
    ]
    col_names = ['Measures', 'DT', 'ChibiT', '1-MC Voc 10','1-MC Voc 100','1-MC Voc 1000','1-MC Voc 10000',]
    envs = ['halfcheetah_medium',]
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_aggregate_table(algs, alg_dataset_dict, col_names)



##################### table generation
# generate_table_nvocab_markov_chain()
generate_table_nstep_markov_chain()