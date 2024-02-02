import os.path
import numpy as np
import pandas as pd
import json
# use this to generate the main table
from log_alias import *

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
        'convergence_step',
        'target_steps',

         'final_feature_diff',
         'final_weight_diff',
         'final_feature_sim',
         'final_weight_sim',

         'best_5percent_normalized',
         'best_10percent_normalized',
         'best_25percent_normalized',
         'best_50percent_normalized',
         'best_100percent_normalized',
         'best_later_half_normalized',
         'best_last_four_normalized',

        'pretrain_best_weight_diff',
        'pretrain_best_weight_sim',

        'pretrain_best_feature_diff',
        'pretrain_best_feature_sim',

         'pretrain_final_feature_diff',
         'pretrain_final_weight_diff',
         'pretrain_final_feature_sim',
         'pretrain_final_weight_sim',
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
    for measure in ['final_test_returns', 'final_test_normalized_returns', 'best_return', 'best_return_normalized',
                    'best_5percent_normalized', 'best_10percent_normalized', 'best_25percent_normalized',
                    'best_50percent_normalized', 'best_100percent_normalized', 'best_later_half_normalized', 'best_last_four_normalized'
                    ]:
        aggregate_dict[measure + '_std'] = [aggregate_dict[measure][1],]
    return aggregate_dict


def get_extra_dict_multiple_seeds_pretrain(datafolder_path):
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
        pretrain_extra_dict_file_path = os.path.join(subdir, 'extra_pretrain.json')
        if 'extra_new.json' in files or 'extra.json' in files:
            if 'extra_new.json' in files:
                extra_dict_file_path = os.path.join(subdir, 'extra_new.json')
            else:
                extra_dict_file_path = os.path.join(subdir, 'extra.json')

            with open(extra_dict_file_path, 'r') as file:
                extra_dict = json.load(file)
                for measure in measures:
                    if 'pretrain' not in measure:
                        aggregate_dict[measure].append(float(extra_dict[measure]))
                # if 'weight_diff_100k' not in extra_dict:
                #     aggregate_dict['weight_diff_100k'].append(float(extra_dict['final_weight_diff']))
                #     aggregate_dict['feature_diff_100k'].append(float(extra_dict['final_feature_diff']))
                # else:
                #     print(extra_dict['feature_diff_100k'])
                #     aggregate_dict['weight_diff_100k'].append(float(extra_dict['weight_diff_100k']))
                #     aggregate_dict['feature_diff_100k'].append(float(extra_dict['feature_diff_100k']))
            with open(pretrain_extra_dict_file_path, 'r') as file:
                pretrain_extra_dict = json.load(file)
                for measure in measures:
                    if 'pretrain' in measure:
                        aggregate_dict[measure].append(float(pretrain_extra_dict[measure]))
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
    for measure in ['final_test_returns', 'final_test_normalized_returns', 'best_return', 'best_return_normalized',
                    'best_5percent_normalized', 'best_10percent_normalized', 'best_25percent_normalized',
                    'best_50percent_normalized', 'best_100percent_normalized', 'best_later_half_normalized', 'best_last_four_normalized'
                    ]:
        aggregate_dict[measure + '_std'] = [aggregate_dict[measure][1],]
    return aggregate_dict

data_path = '../../code/checkpoints/'

MUJOCO_3_ENVS = [
                'hopper',
                 'halfcheetah',
                'walker2d',
                "ant"
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


def get_alg_dataset_dict_pretrain(algs, envs):
    # load extra dict for all alg, all envs, all seeds
    alg_dataset_dict = {}
    for alg in algs:
        alg_dataset_dict[alg] = {}
        for env in envs:
            folderpath = os.path.join(data_path, '%s_%s' % (alg, env))
            alg_dataset_dict[alg][env] = get_extra_dict_multiple_seeds_pretrain(folderpath)
    return alg_dataset_dict

def get_aggregated_value(alg_dataset_dict, alg, measure):
    # for an alg-measure pair, aggregate over all datasets
    value_list = []
    for dataset, extra_dict in alg_dataset_dict[alg].items():
        value_list.append(extra_dict[measure][0]) # each entry is the value from a dataset
    return np.mean(value_list), np.std(value_list)


OLD_ROWS = [
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
OLD_ROW_NAMES = ['Best Score',
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

NEW_PERFORMANCE_ROWS = [
    'best_return_normalized',
    'best_5percent_normalized',
    'best_10percent_normalized',
    'best_25percent_normalized',
    'best_50percent_normalized',
    'best_100percent_normalized',
]
NEW_PERFORMANCE_ROW_NAMES = [
    'Best Score',
    'Best Score 5\\%',
    'Best Score 10\\%',
    'Best Score 25\\%',
    'Best Score 50\\%',
    'Best Score 100\\%',
]

change_std_rows = [
    'best_return_normalized',
    'best_5percent_normalized',
    'best_10percent_normalized',
    'best_25percent_normalized',
    'best_50percent_normalized',
    'best_100percent_normalized',
    'best_later_half_normalized',
    'best_last_four_normalized'
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
    'Best Score 5\\%',
    'Best Score 10\\%',
    'Best Score 25\\%',
    'Best Score 50\\%',
    'Best Score 100\\%',
]

row_names_use_1_precision = [
    'Best Score', 'Best Std over Seeds', 'Convergence Iter',
    'Final Score',
    'Best Score 5\\%',
    'Best Score 10\\%',
    'Best Score 25\\%',
    'Best Score 50\\%',
    'Best Score 100\\%',
]

"""table generation"""
def generate_aggregate_table(algs, alg_dataset_dict, column_names, best_value_bold=True, bold_threshold=0.01):
    print("\nNow generate latex table:\n")
    # each row is a measure, each column is an algorithm variant
    rows = NEW_PERFORMANCE_ROWS
    row_names = NEW_PERFORMANCE_ROW_NAMES

    table = np.zeros((2, len(rows), len(algs)))
    # each iter we generate a row
    for i, row in enumerate(rows):
        for j, alg in enumerate(algs):
            table[0,i,j], table[1,i,j] = get_aggregated_value(alg_dataset_dict, alg, row)
            if row in change_std_rows: # TODO
                std_mean, std_std = get_aggregated_value(alg_dataset_dict, alg, row+'_std')
                table[1, i, j] = std_mean
            if row == 'final_test_normalized_returns':
                std_mean, std_std = get_aggregated_value(alg_dataset_dict, alg, 'final_test_normalized_returns_std')
                table[1, i, j] = std_mean

    max_values = np.max(table[0], axis=1)
    min_values = np.min(table[0], axis=1)

    col_name_line = ''
    for col in column_names:
        col_name_line += str(col) +' & '
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
                    elif row_name in row_names_use_1_precision:
                        row_string += (' & \\textbf{%.1f} $\pm$ %.1f' % (mean, std))
                    else:
                        row_string += (' & \\textbf{%.3f} $\pm$ %.1f' % (mean, std))
                else:
                    if 'Prev' in row_name:
                        row_string += (' & %.6f' % (mean,))
                    elif row_name in row_names_use_1_precision:
                        row_string += (' & %.1f $\pm$ %.1f' % (mean, std))
                    else:
                        row_string += (' & %.3f $\pm$ %.1f' % (mean, std))
        row_string += '\\\\'
        print(row_string)


def generate_aggregate_performance(algs, alg_dataset_dict, column_names, best_value_bold=True, bold_threshold=0.01,
                                   measure='best_return_normalized', row_name='Average (All Settings)', higher_is_better=True):
    # each row is a measure, each column is an algorithm variant
    rows = [measure]
    row_names = [row_name]

    table = np.zeros((2, len(rows), len(algs)))
    # each iter we generate a row
    for i, row in enumerate(rows):
        for j, alg in enumerate(algs):
            table[0,i,j], table[1,i,j] = get_aggregated_value(alg_dataset_dict, alg, row)
            if row in change_std_rows: # TODO
                std_mean, std_std = get_aggregated_value(alg_dataset_dict, alg, row+'_std')
                table[1, i, j] = std_mean
            if row == 'final_test_normalized_returns':
                std_mean, std_std = get_aggregated_value(alg_dataset_dict, alg, 'final_test_normalized_returns_std')
                table[1, i, j] = std_mean

    max_values = np.max(table[0], axis=1)
    min_values = np.min(table[0], axis=1)

    print("		\\hline ")
    for i, row_name in enumerate(row_names):
        row_string = row_name
        for j in range(len(algs)):
            mean, std = table[0, i, j], table[1, i, j]
            bold = False
            if best_value_bold:
                if not higher_is_better:
                    if mean <= (1+bold_threshold)*min_values[i]:
                        bold = True
                else:
                    if mean >= (1-bold_threshold)*max_values[i]:
                        bold = True
                #if mean >= (1-bold_threshold)*max_values[i]:
                #    bold = True
                if bold:
                    row_string += (' & \\textbf{%.1f} $\pm$ %.1f' % (mean, std))
                else:
                    row_string += (' & %.1f $\pm$ %.1f' % (mean, std))
        row_string += '\\\\'
        print(row_string)




def generate_aggregate_performance_pretrain(algs, alg_dataset_dict, column_names, best_value_bold=True, bold_threshold=0.05,
                                   measure='best_return_normalized', row_name='Average (All Settings)', higher_is_better=True):
    # each row is a measure, each column is an algorithm variant
    if isinstance(measure, list):
        rows = measure
        row_names = row_name
    else:
        rows = [measure]
        row_names = [row_name]

    table = np.zeros((2, len(rows), len(algs)))
    # each iter we generate a row
    for i, row in enumerate(rows):
        for j, alg in enumerate(algs):
            table[0,i,j], table[1,i,j] = get_aggregated_value(alg_dataset_dict, alg, row)
            if row in change_std_rows: # TODO
                std_mean, std_std = get_aggregated_value(alg_dataset_dict, alg, row+'_std')
                table[1, i, j] = std_mean
            if row == 'final_test_normalized_returns':
                std_mean, std_std = get_aggregated_value(alg_dataset_dict, alg, 'final_test_normalized_returns_std')
                table[1, i, j] = std_mean

    max_values = np.max(table[0], axis=1)
    min_values = np.min(table[0], axis=1)

    print("		\\hline ")
    for i, row_name in enumerate(row_names):
        row_string = row_name
        for j in range(len(algs)):
            mean, std = table[0, i, j], table[1, i, j]
            bold = False
            if best_value_bold:
                if not higher_is_better:
                    if mean <= (1+bold_threshold)*min_values[i]:
                        bold = True
                else:
                    if mean >= (1-bold_threshold)*max_values[i]:
                        bold = True
                #if mean >= (1-bold_threshold)*max_values[i]:
                #    bold = True
                if bold:
                    if mean < 0.1:
                        row_string += (' & \\textbf{%.1E} ' % mean)
                    else:
                        row_string += (' & \\textbf{%.2f} ' % mean)
                else:
                    if mean < 0.1:
                        row_string += (' & %.1E ' % mean)
                    else:
                        row_string += (' & %.2f ' % mean)
        row_string += '\\\\'
        print(row_string)



# TODO add an aggregate score at the end
def generate_per_env_score_table_new(algs, alg_dataset_dict, column_names, best_value_bold=True, bold_threshold=0.01, measure='best_return_normalized', higher_is_better = True):
    print("\nNow generate latex table:\n")
    # measure = 'best_100percent_normalized'
    # each row is a env-dataset pair, each column is an algorithm variant
    rows = []
    row_names = []
    for dataset in ['medium-expert', 'medium', 'medium-replay', ]:
        for e in ['halfcheetah', 'hopper', 'walker2d',"ant" ]:
            rows.append('%s_%s' % (e, dataset))
            row_names.append('%s-%s' % (e, dataset))

    table = np.zeros((2, len(rows), len(algs)))
    # each iter we generate a row
    for i, row in enumerate(rows):
        for j, alg in enumerate(algs):
            try:
                table[0,i,j], table[1,i,j] = alg_dataset_dict[alg][row][measure]
            except:
                print(measure)
                print(alg)
                print(row)
                print(alg_dataset_dict[alg][row].keys())
                quit()

    max_values = np.max(table[0], axis=1)
    min_values = np.min(table[0], axis=1)

    col_name_line = ''
    for col in column_names:
        col_name_line += str(col) +' & '
    col_name_line = col_name_line[:-2] + '\\\\'
    print(col_name_line)
    print("		\\hline ")
    for i, row_name in enumerate(row_names):
        row_string = row_name
        for j in range(len(algs)):
            mean, std = table[0, i, j], table[1, i, j]
            bold = False
            if best_value_bold:
                if not higher_is_better:
                    if mean <= (1+bold_threshold)*min_values[i]:
                        bold = True
                else:
                    if mean >= (1-bold_threshold)*max_values[i]:
                        bold = True
                #if mean >= (1-bold_threshold)*max_values[i]:
                #    bold = True
                if bold:
                    row_string += (' & \\textbf{%.1f} $\pm$ %.1f' % (mean, std))
                else:
                    row_string += (' & %.1f $\pm$ %.1f' % (mean, std))
        row_string += '\\\\'
        print(row_string)


def generate_per_env_score_table_new_pretrain(algs, alg_dataset_dict, column_names, best_value_bold=True, bold_threshold=0.05, measure='best_return_normalized', higher_is_better = True):
    print("\nNow generate latex table:\n")
    # measure = 'best_100percent_normalized'
    # each row is a env-dataset pair, each column is an algorithm variant
    rows = []
    row_names = []
    for dataset in ['medium-expert', 'medium', 'medium-replay', ]:
        for e in ['halfcheetah', 'hopper', 'walker2d',"ant" ]:
            rows.append('%s_%s' % (e, dataset))
            row_names.append('%s-%s' % (e, dataset))

    table = np.zeros((2, len(rows), len(algs)))
    # each iter we generate a row
    for i, row in enumerate(rows):
        for j, alg in enumerate(algs):
            try:
                table[0,i,j], table[1,i,j] = alg_dataset_dict[alg][row][measure]
            except:
                print(measure)
                print(alg)
                print(row)
                print(alg_dataset_dict[alg][row].keys())
                quit()

    max_values = np.max(table[0], axis=1)
    min_values = np.min(table[0], axis=1)

    col_name_line = ''
    for col in column_names:
        col_name_line += str(col) +' & '
    col_name_line = col_name_line[:-2] + '\\\\'
    print(col_name_line)
    print("		\\hline ")
    for i, row_name in enumerate(row_names):
        row_string = row_name
        for j in range(len(algs)):
            mean, std = table[0, i, j], table[1, i, j]
                
            bold = False
            if best_value_bold:
                if not higher_is_better:
                    if mean <= (1+bold_threshold)*min_values[i]:
                        bold = True
                else:
                    if mean >= (1-bold_threshold)*max_values[i]:
                        bold = True
                #if mean >= (1-bold_threshold)*max_values[i]:
                #    bold = True
                if bold:
                    if mean < 0.1:
                        row_string += (' & \\textbf{%.1E} ' % mean)
                    else:
                        row_string += (' & \\textbf{%.2f} ' % mean)
                else:
                    if mean < 0.1:
                        row_string += (' & %.1E ' % mean)
                    else:
                        row_string += (' & %.2f ' % mean)

        row_string += '\\\\'
        print(row_string)

    # if add_aggregate_result_in_the_end:
    #     print("		\\hline ")
    #     agg_mean, agg_std = np.mean(table[0], axis=0), np.mean(table[1], axis=0)
    #     print(agg_mean)
    #     print(agg_std)



def generate_per_env_score_table(max_value_bold=True, bold_threshold=0.95):
    # TODO need to fix the bold thing
    print("\nNow generate latex table:\n")
    measure = 'best_return_normalized'
    # each row is a env-dataset pair, each column is an algorithm variant
    rows = []
    row_names = []
    for dataset in ['medium-expert', 'medium', 'medium-replay', ]:
        for e in ['halfcheetah', 'hopper', 'walker2d',"ant" ]:
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



def generate_dt_first_table():
    algs = [
        dt,
        chibiT,
        dt_mc_1step_vocab100,
    ]
    col_names = ['Measures', 'DT', 'ChibiT', '1-MC Voc 100']
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_aggregate_table(algs, alg_dataset_dict, col_names)

MUJOCO_3_ENVS = [
                # 'hopper',
                 'halfcheetah',
                # 'walker2d',
]
MUJOCO_3_DATASETS = ['medium','medium-replay','medium-expert',]
envs2 = []
for e in MUJOCO_3_ENVS:
    for dataset in MUJOCO_3_DATASETS:
        envs2.append('%s_%s' % (e, dataset))


def generate_table_nvocab_markov_chain():
    #################### table 1
    # DT table, 1-step markov chain, change the number of vocab
    algs = [
        'chibiT-rerun',
        'chibiT-rerun-syn_ngram1_nvocab10_temperature1.0',
        'chibiT-rerun-syn_ngram1_nvocab100_temperature1.0',
        'chibiT-rerun-syn_ngram1_nvocab1000_temperature1.0',
        'chibiT-rerun-syn_ngram1_nvocab10000_temperature1.0',
        dt_mc_1step_vocab50257,
        dt_mc_1step_vocab100000,
    ]
    col_names = ['Measures', 'ChibiT', '1-MC Voc 10','1-MC Voc 100','1-MC Voc 1000','1-MC Voc 10000',
                 '1-MC voc 50257','1-MC Voc 100000']
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_aggregate_table(algs, alg_dataset_dict, col_names)

MUJOCO_3_ENVS = [
                # 'hopper',
                 'halfcheetah',
                # 'walker2d',
]
MUJOCO_3_DATASETS = ['medium','medium-replay','medium-expert',]
envs2 = []
for e in MUJOCO_3_ENVS:
    for dataset in MUJOCO_3_DATASETS:
        envs2.append('%s_%s' % (e, dataset))


def generate_table_markov_chain_compare_number_of_steps():
    #################### table 2
    # DT table, pretrain with markov chain data change number of step
    algs = [
    'dt-rerun-data_size_dt_1.0',
        'chibiT-rerun',
        'chibiT-rerun-syn_ngram1_nvocab50257_temperature1.0',
        'chibiT-rerun-syn_ngram2_nvocab50257_temperature1.0',
        'chibiT-rerun-syn_ngram3_nvocab50257_temperature1.0',
        'chibiT-rerun-syn_ngram4_nvocab50257_temperature1.0',
        'chibiT-rerun-syn_ngram5_nvocab50257_temperature1.0',
    ]
    col_names = ['Measures', 'DT', 'ChibiT', '1-MC Voc 50257','2-MC Voc 50257','3-MC Voc 50257','4-MC Voc 50257','5-MC Voc 50257',]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_aggregate_table(algs, alg_dataset_dict, col_names)


def generate_dt_mc_table_compare_temperature():
    #################### table 2
    # DT table, pretrain with markov chain data change number of step
    algs = [
        # dt_mc_temp0_1_vocab50257,
        # dt_mc_temp0_2_vocab50257,
        dt_mc_temp0_4_vocab50257,
        dt_mc_temp0_8_vocab50257,
        dt_mc_temp1_0_vocab50257,
        dt_mc_temp10_0_vocab50257,
    ]
    col_names = ['Measures', '1step v50257 0.4','1step v50257 0.8','1step v50257 1','1step v50257 10',] # '0.1', '0.2',
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_aggregate_table(algs, alg_dataset_dict, col_names)


def generate_dt_mc_table_more1():
    algs = [
        dt_mc_1step_vocab100,
        dt_mc_2step_vocab100,
        dt_mc_3step_vocab100,
        dt_mc_4step_vocab100,
        dt_mc_5step_vocab100,
    ]
    col_names = ['Measures',
                 '1step v100', '2step v100','3step v100', '4step v100', '5step v100']
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_aggregate_table(algs, alg_dataset_dict, col_names)


def generate_dt_mc_table_more2():
    algs = [
        dt_mc_temp0_1_vocab100,
        dt_mc_temp0_2_vocab100,
        dt_mc_temp0_4_vocab100,
        dt_mc_temp0_8_vocab100,
        dt_mc_1step_vocab100,
        dt_mc_temp10_0_vocab100,
    ]
    col_names = ['Measures',
                 '1step v100 0.1','1step v100 0.2','1step v100 0.4','1step v100 0.8', '1step v100 1','1step v100 10',] # '0.1', '0.2',
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_aggregate_table(algs, alg_dataset_dict, col_names)


def generate_table_cql_cross_domain():
    # CQL table, cross domain pretraining
    algs = [
    'cqlr3_prenone_l2_qflrs1',
    'cqlr3_preq_sprime_l2_qflrs1',
    'cqlr3_preproj0_q_sprime_l2',
    'cqlr3_preproj1_q_sprime_l2',
    cql_random_pretrain,
    cql_random_1000_state,
    ]
    col_names = ['Measures', 'CQL', 'CQL pre', 'CQL proj0', 'CQL proj1', 'CQL rand pre', 'CQL rand 1000']
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_aggregate_table(algs, alg_dataset_dict, col_names)


def generate_table_cql_mdp_compare_n_state_action():
    algs = [
        # cql_base,
        # cql_fd_pretrain,
        # cql_random_pretrain,
        cql_mdp_pretrain_nstate1,
        cql_mdp_pretrain_nstate10,
        cql_mdp_pretrain_nstate100,
        cql_mdp_pretrain_nstate1000,
        cql_mdp_pretrain_nstate10000,
        cql_mdp_pretrain_nstate50257,
    ]
    col_names = ['Measures',
                 # 'CQL', 'CQL pre', 'CQL rand pre',
                '1', '10', '100', '1000', '10000', '50257'
                 ]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_aggregate_table(algs, alg_dataset_dict, col_names)


def generate_table_cql_mdp_compare_temperature():
    algs = [
        # cql_base,
        # cql_fd_pretrain,
        # cql_random_pretrain,
        cql_mdp_pretrain_temperature0_01,
        cql_mdp_pretrain_temperature0_1,
        cql_mdp_pretrain_temperature1,
        cql_mdp_pretrain_temperature10,
        cql_mdp_pretrain_temperature100,
    ]
    col_names = ['Measures',
                 # 'CQL', 'CQL pre', 'CQL rand pre',
                '0.01', '0.1', '1', '10', '100'
                 ]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_aggregate_table(algs, alg_dataset_dict, col_names)


def generate_table_cql_mdp_compare_state_action_dim():
    algs = [
        # cql_base,
        # cql_fd_pretrain,
        # cql_random_pretrain,
        cql_mdp_pretrain_state_action_dim1,
        cql_mdp_pretrain_state_action_dim5,
        cql_mdp_pretrain_state_action_dim20,
        cql_mdp_pretrain_state_action_dim50,
        cql_mdp_pretrain_state_action_dim200,
        cql_mdp_pretrain_state_action_dim1000,
    ]
    col_names = ['Measures',
                 # 'CQL', 'CQL pre', 'CQL rand pre',
                '1', '5', '20', '50', '200', '1000'
                 ]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_aggregate_table(algs, alg_dataset_dict, col_names)



def generate_table_cql_mdp_same_dim_with_and_no_projection():
    algs = [
        cql_fd_pretrain,
        cql_fd_pretrain_same_task_with_projection,
        cql_fd_pretrain_cross_task1,
        cql_fd_pretrain_cross_task2,
        cql_mdp_pretrain_same_dim_no_projection,
        cql_mdp_pretrain_same_dim_with_projection,
    ]
    col_names = ['Measures',
                'CQL pre', 'CQL pre (proj)',
                'CQL cross task', 'CQL cross task',
                 'CQL mdp same dim', 'CQL mdp same dim (proj)',
                 ]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_aggregate_table(algs, alg_dataset_dict, col_names)



def generate_table_cql_3x_data():
    algs = [
        cql_base,
        cql_fd_pretrain,
        cql_fd_3x_data,
        cql_fd_3x_data_with_projection,
        cql_fd_3x_data_cross_task,
    ]
    col_names = ['Measures',
                 'CQL',
                'CQL pre', 'CQL pre (3x)',
                 'CQL pre (3x, proj)', 'CQL cross-task 3x'
                 ]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_aggregate_table(algs, alg_dataset_dict, col_names)


def generate_table_no_action_predict_next_state():
    algs = [
        cql_base,
        cql_fd_pretrain,
        cql_no_action_predict_next_state
    ]
    col_names = ['Measures',
                 'CQL',
                'CQL pre',
                 'CQL fd no act',
                 ]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_aggregate_table(algs, alg_dataset_dict, col_names)

def generate_table_cql_pretrain_data_sizes():
    algs = [
        cql_fd_pretrain_data_ratio_0_01,
        cql_fd_pretrain_data_ratio_0_1,
        cql_fd_pretrain_data_ratio_0_25,
        cql_fd_pretrain_data_ratio_0_5,
        cql_fd_pretrain_data_ratio_0_75,
        cql_fd_pretrain_data_ratio_1,
    ]
    col_names = ['Measures',
                 'CQL pre 0.01',
                 'CQL pre 0.1',
                 'CQL pre 0.25',
                 'CQL pre 0.5',
                 'CQL pre 0.75',
                 'CQL pre 1',
                 ]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_aggregate_table(algs, alg_dataset_dict, col_names)

def generate_table_cql_mdp_pretrain_data_sizes():
    algs = [
        cql_mdp_pretrain_data_ratio_0_01,
        cql_mdp_pretrain_data_ratio_0_1,
        cql_mdp_pretrain_data_ratio_0_25,
        cql_mdp_pretrain_data_ratio_0_5,
        cql_mdp_pretrain_data_ratio_0_75,
        cql_mdp_pretrain_data_ratio_1,
    ]
    col_names = ['Measures',
                 'CQL MDP 0.01',
                 'CQL MDP 0.1',
                 'CQL MDP 0.25',
                 'CQL MDP 0.5',
                 'CQL MDP 0.75',
                 'CQL MDP 1',
                 ]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_aggregate_table(algs, alg_dataset_dict, col_names)

def generate_table_cql_finetune_data_sizes():
    algs = [
        cql_fd_finetune_data_ratio_0_01,
        cql_fd_finetune_data_ratio_0_1,
        cql_fd_finetune_data_ratio_0_25,
        cql_fd_finetune_data_ratio_0_5,
        cql_fd_finetune_data_ratio_0_75,
        cql_fd_finetune_data_ratio_1,
    ]
    col_names = ['Measures',
                 'CQL finetune 0.01',
                 'CQL finetune 0.1',
                 'CQL finetune 0.25',
                 'CQL finetune 0.5',
                 'CQL finetune 0.75',
                 'CQL finetune 1',
                 ]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_aggregate_table(algs, alg_dataset_dict, col_names)


def generate_table_mdp_pretrain_finetune_data_sizes():
    algs = [
        cql_mdp_finetune_data_ratio_0_01,
        cql_mdp_finetune_data_ratio_0_1,
        cql_mdp_finetune_data_ratio_0_25,
        cql_mdp_finetune_data_ratio_0_5,
        cql_mdp_finetune_data_ratio_0_75,
        cql_mdp_finetune_data_ratio_1,
    ]
    col_names = ['Measures',
                 'CQL mdp ft 0.01',
                 'CQL mdp ft 0.1',
                 'CQL mdp ft 0.25',
                 'CQL mdp ft 0.5',
                 'CQL mdp ft 0.75',
                 'CQL mdp ft 1',
                 ]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_aggregate_table(algs, alg_dataset_dict, col_names)


def generate_table_cql_finetune_both_data_sizes():
    algs = [
        cql_fd_finetune_both_ratio_0_01,
        cql_fd_finetune_both_ratio_0_1,
        cql_fd_finetune_both_ratio_0_25,
        cql_fd_finetune_both_ratio_0_5,
        cql_fd_finetune_both_ratio_0_75,
        cql_fd_finetune_both_ratio_1,
    ]
    col_names = ['Measures',
                 'CQL both 0.01',
                 'CQL both 0.1',
                 'CQL both 0.25',
                 'CQL both 0.5',
                 'CQL both 0.75',
                 'CQL both 1',
                 ]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_aggregate_table(algs, alg_dataset_dict, col_names)


def generate_table_cql_no_pretrain_finetune_data_sizes():
    algs = [
        cql_no_pretrain_0_1_data,
        cql_no_pretrain_0_25_data,
        cql_no_pretrain_0_5_data,
        cql_no_pretrain_0_75_data,
        cql_no_pretrain_1_data,
    ]
    col_names = ['Measures',
                 'CQL no pre 0.1',
                 'CQL no pre 0.25',
                 'CQL no pre 0.5',
                 'CQL no pre 0.75',
                 'CQL no pre 1',
                 ]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_aggregate_table(algs, alg_dataset_dict, col_names)


def generate_table_cql_with_target_networks():
    algs = [
        cql_fd_finetune_data_ratio_0_01,
        cql_fd_finetune_data_ratio_0_1,
        cql_fd_finetune_data_ratio_1,
        cql_rl_with_target_hard_update_0_01,
        cql_rl_with_target_hard_update_0_1,
        cql_rl_with_target_hard_update_1,
    ]
    col_names = ['Measures',
                 'rl 0.01',
                 'rl 0.1',
                 'rl 1',
                 'rl wtu 0.01',
                 'rl wtu 0.1',
                 'rl wtu 1',
                 ]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_aggregate_table(algs, alg_dataset_dict, col_names)


def generate_table_cql_with_target_networks2():
    algs = [
        cql_mdp_finetune_data_ratio_0_01,
        cql_mdp_finetune_data_ratio_0_1,
        cql_mdp_finetune_data_ratio_1,
        cql_mdp_with_target_hard_update_0_01,
        cql_mdp_with_target_hard_update_0_1,
        cql_mdp_with_target_hard_update_1,
    ]
    col_names = ['Measures',
                 'mdp 0.01',
                 'mdp 0.1',
                 'mdp 1',
                 'wtu 0.01',
                 'wtu 0.1',
                 'wtu 1',
                 ]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_aggregate_table(algs, alg_dataset_dict, col_names)


def generate_table_cql_with_target_networks3():
    algs = [
        cql_fd_finetune_data_ratio_0_25,
        cql_fd_finetune_data_ratio_0_5,
        cql_fd_finetune_data_ratio_0_75,
        cql_rl_with_target_hard_update_0_25,
        cql_rl_with_target_hard_update_0_5,
        cql_rl_with_target_hard_update_0_75,
    ]
    col_names = ['Measures',
                 'rl 0.25',
                 'rl 0.5',
                 'rl 0.75',
                 'rl wtu 0.25',
                 'rl wtu 0.5',
                 'rl wtu 0.75',
                 ]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_aggregate_table(algs, alg_dataset_dict, col_names)


def generate_table_cql_with_target_networks4():
    algs = [
        cql_mdp_finetune_data_ratio_0_25,
        cql_mdp_finetune_data_ratio_0_5,
        cql_mdp_finetune_data_ratio_0_75,
        cql_mdp_with_target_hard_update_0_25,
        cql_mdp_with_target_hard_update_0_5,
        cql_mdp_with_target_hard_update_0_75,
    ]
    col_names = ['Measures',
                 'mdp 0.25',
                 'mdp 0.5',
                 'mdp 0.75',
                 'wtu 0.25',
                 'wtu 0.5',
                 'wtu 0.75',
                 ]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_aggregate_table(algs, alg_dataset_dict, col_names)

################################################## paper july
def iclr_generate_dt_first_table_per_env():
    algs = [
        dt,
        chibiT,
        dt_mc_1step_vocab100,
    ]
    col_names = ['Best Score', 'DT', 'DT+Wiki', '1-MC S100']
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    col_names[0] = 'Average Score'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')

    col_names[0] = 'Average Later Half'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')

    col_names[0] = 'Average Last Four'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')

def iclr_generate_dt_first_table_per_env_20seeds():
    algs = [
        'dt-rerun-20seeds_dt',
        'chibiT-rerun-20seeds',
        "chibiT-syn-20seeds-steps-wo_step20000",
    ]
    col_names = ['Best Score', 'DT', 'DT+Wiki', 'DT+Synthetic']
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    col_names[0] = 'Average Last Four'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')


def iclr_generate_dt_analysis():
    algs = [
        'dt-rerun-20seeds_dt',
        'chibiT-rerun-20seeds',
        "chibiT-syn-20seeds-steps-wo_step20000",
        'chibiT-iid_wo_step20000',
        'chibiT-iid_wo_acl_acl1',
        'chibiT-iid_wo_acl_acl2'
    ]
    col_names = ['Best vs. Pre-trained Feature Difference', 'DT', 'DT+Wiki', 'DT+Synthetic', 'DT+IID', 'DT+Identity', 'DT+Mapping']
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict_pretrain(algs, envs)

    generate_per_env_score_table_new_pretrain(algs, alg_dataset_dict, col_names, measure='best_feature_diff', higher_is_better=False)
    generate_aggregate_performance_pretrain(algs, alg_dataset_dict, col_names, measure='best_feature_diff', higher_is_better=False)

    col_names[0] = 'Final vs. Pre-trained Feature Difference'
    generate_per_env_score_table_new_pretrain(algs, alg_dataset_dict, col_names, measure='final_feature_diff', higher_is_better=False)
    generate_aggregate_performance_pretrain(algs, alg_dataset_dict, col_names, measure='final_feature_diff', higher_is_better=False)

    col_names[0] = 'Best vs. Pre-trained Feature Similarity'
    generate_per_env_score_table_new_pretrain(algs, alg_dataset_dict, col_names, measure='best_feature_sim', higher_is_better=True)
    generate_aggregate_performance_pretrain(algs, alg_dataset_dict, col_names, measure='best_feature_sim', higher_is_better=True)

    col_names[0] = 'Final vs. Pre-trained Feature similarity'
    generate_per_env_score_table_new_pretrain(algs, alg_dataset_dict, col_names, measure='final_feature_sim', higher_is_better=True)
    generate_aggregate_performance_pretrain(algs, alg_dataset_dict, col_names, measure='final_feature_sim', higher_is_better=True)

    col_names[0] = 'Best vs. Pre-trained Weight Difference'
    generate_per_env_score_table_new_pretrain(algs, alg_dataset_dict, col_names, measure='best_weight_diff', higher_is_better=False)
    generate_aggregate_performance_pretrain(algs, alg_dataset_dict, col_names, measure='best_weight_diff', higher_is_better=False)

    col_names[0] = 'Final vs. Pre-trained Weight Difference'
    generate_per_env_score_table_new_pretrain(algs, alg_dataset_dict, col_names, measure='final_weight_diff', higher_is_better=False)
    generate_aggregate_performance_pretrain(algs, alg_dataset_dict, col_names, measure='final_weight_diff', higher_is_better=False)


    col_names[0] = 'Best vs. Pre-trained Weight Similarity'
    generate_per_env_score_table_new_pretrain(algs, alg_dataset_dict, col_names, measure='best_weight_sim', higher_is_better=True)
    generate_aggregate_performance_pretrain(algs, alg_dataset_dict, col_names, measure='best_weight_sim', higher_is_better=True)

    col_names[0] = 'Final vs. Pre-trained Weight Similarity'
    generate_per_env_score_table_new_pretrain(algs, alg_dataset_dict, col_names, measure='final_weight_sim', higher_is_better=True)
    generate_aggregate_performance_pretrain(algs, alg_dataset_dict, col_names, measure='final_weight_sim', higher_is_better=True)


    col_names[0] = 'Best vs. Random Init. Feature Difference'
    generate_per_env_score_table_new_pretrain(algs, alg_dataset_dict, col_names, measure='pretrain_best_feature_diff', higher_is_better=False)
    generate_aggregate_performance_pretrain(algs, alg_dataset_dict, col_names, measure='pretrain_best_feature_diff', higher_is_better=False)

    col_names[0] = 'Final vs. Random Init. Feature Difference'
    generate_per_env_score_table_new_pretrain(algs, alg_dataset_dict, col_names, measure='pretrain_final_feature_diff', higher_is_better=False)
    generate_aggregate_performance_pretrain(algs, alg_dataset_dict, col_names, measure='pretrain_final_feature_diff', higher_is_better=False)

    col_names[0] = 'Best vs. Random Init. Feature Similarity'
    generate_per_env_score_table_new_pretrain(algs, alg_dataset_dict, col_names, measure='pretrain_best_feature_sim', higher_is_better=True)
    generate_aggregate_performance_pretrain(algs, alg_dataset_dict, col_names, measure='pretrain_best_feature_sim', higher_is_better=True)

    col_names[0] = 'Final vs. Random Init. Feature similarity'
    generate_per_env_score_table_new_pretrain(algs, alg_dataset_dict, col_names, measure='pretrain_final_feature_sim', higher_is_better=True)
    generate_aggregate_performance_pretrain(algs, alg_dataset_dict, col_names, measure='pretrain_final_feature_sim', higher_is_better=True)

    col_names[0] = 'Best vs. Random Init. Weight Difference'
    generate_per_env_score_table_new_pretrain(algs, alg_dataset_dict, col_names, measure='pretrain_best_weight_diff', higher_is_better=False)
    generate_aggregate_performance_pretrain(algs, alg_dataset_dict, col_names, measure='pretrain_best_weight_diff', higher_is_better=False)

    col_names[0] = 'Final vs. Random Init. Weight Difference'
    generate_per_env_score_table_new_pretrain(algs, alg_dataset_dict, col_names, measure='pretrain_final_weight_diff', higher_is_better=False)
    generate_aggregate_performance_pretrain(algs, alg_dataset_dict, col_names, measure='pretrain_final_weight_diff', higher_is_better=False)


    col_names[0] = 'Best vs. Random Init. Weight Similarity'
    generate_per_env_score_table_new_pretrain(algs, alg_dataset_dict, col_names, measure='pretrain_best_weight_sim', higher_is_better=True)
    generate_aggregate_performance_pretrain(algs, alg_dataset_dict, col_names, measure='pretrain_best_weight_sim', higher_is_better=True)

    col_names[0] = 'Final vs. Random Init. Weight Similarity'
    generate_per_env_score_table_new_pretrain(algs, alg_dataset_dict, col_names, measure='pretrain_final_weight_sim', higher_is_better=True)
    generate_aggregate_performance_pretrain(algs, alg_dataset_dict, col_names, measure='pretrain_final_weight_sim', higher_is_better=True)



def iclr_generate_dt_analysis_agg():
    algs = [
        'dt-rerun-20seeds_dt',
        'chibiT-rerun-20seeds',
        "chibiT-syn-20seeds-steps-wo_step20000",
        'chibiT-iid_wo_step20000',
        'chibiT-iid_wo_acl_acl1',
        'chibiT-iid_wo_acl_acl2'
    ]
    col_names = ['FT vs. PT Feature Diff.', 'FT vs. RI Feature Diff.', 'DT', 'DT+Wiki', 'DT+Synthetic', 'DT+IID', 'DT+Identity', 'DT+Mapping']
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict_pretrain(algs, envs)

    generate_aggregate_performance_pretrain(algs, alg_dataset_dict, col_names, measure=['final_feature_diff', 'pretrain_final_feature_diff'], row_name = ['FT vs. PT Feature Diff.', 'FT vs. RI Feature Diff.'], higher_is_better=False)

    col_names = ['FT vs. PT Feature Sim.', 'FT vs. RI Feature Sim.', 'DT', 'DT+Wiki', 'DT+Synthetic', 'DT+IID', 'DT+Identity', 'DT+Mapping']
    generate_aggregate_performance_pretrain(algs, alg_dataset_dict, col_names, measure=['final_feature_sim', 'pretrain_final_feature_sim'], row_name = ['FT vs. PT Feature Sim.', 'FT vs. RI Feature Sim.'], higher_is_better=True)

    col_names = ['FT vs. PT Weight Diff.', 'FT vs. RI Weight Diff.', 'DT', 'DT+Wiki', 'DT+Synthetic', 'DT+IID', 'DT+Identity', 'DT+Mapping']
    generate_aggregate_performance_pretrain(algs, alg_dataset_dict, col_names, measure=['final_weight_diff', 'pretrain_final_weight_diff'], row_name = ['FT vs. PT Weight Diff.', 'FT vs. RI Weight Diff.'], higher_is_better=False)

    col_names = ['FT vs. PT Weight Sim.', 'FT vs. RI Weight Sim.', 'DT', 'DT+Wiki', 'DT+Synthetic', 'DT+IID', 'DT+Identity', 'DT+Mapping']
    generate_aggregate_performance_pretrain(algs, alg_dataset_dict, col_names, measure=['final_weight_sim', 'pretrain_final_weight_sim'], row_name = ['FT vs. PT Weight Sim.', 'FT vs. RI Weight Sim.'], higher_is_better=True)

def iclr_generate_dt_first_table_per_env_20seeds_diff_pretrain():
    algs = [
        'dt-rerun-20seeds_dt',
        'chibiT-rerun-20seeds',
        'chibiT-iid_wo_acl_acl1',
        'chibiT-iid_wo_acl_acl2',
        "chibiT-syn-20seeds-steps-wo_step20000",

    ]
    col_names = ['Best Score', 'DT', 'DT+Wiki', 'DT+Identity', 'DT+Mapping', 'DT+Synthetic']
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    col_names[0] = 'Average Last Four'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')





def iclr_generate_dt_first_table_per_env_conv_iter_20seeds():
    algs = [
        'dt-rerun-20seeds_dt',
        'chibiT-rerun-20seeds',
        "chibiT-syn-20seeds-steps-wo_step20000",
    ]
    col_names = ['Number of Updates',  "DT", 'DT+Wiki', 'DT+Synthetic']
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure="target_steps")
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure="target_steps")


def iclr_generate_dt():
    algs = [
        'dt-rerun-20seeds_dt',
    ]
    col_names = ['score', 'DT']
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure="final_test_returns")
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure="final_test_returns")


def iclr_generate_dt_first_table_per_env_conv_iter():
    algs = [
        dt,
        chibiT,
        "chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_20000"    
        ]
    col_names = ['Convergence Iteration', 'DT', 'DT+Wiki', 'DT+Synthetic']
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure="convergence_iter")
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure="convergence_iter")

    col_names[0] = 'Convergence Step'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure="convergence_step")
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure="convergence_step")


def iclr_generate_dt_kmeans():
    algs = [
        dt,
        chibiT,
        "dt-rerun-20seeds_dt",
        "chibiT-rerun_wo",
        "chibiT_w_random",
        "chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_step1_wo"
        ]
    col_names = ['Best Score', 'DT', 'DT+Wiki', 'DT 20seeds', "Wiki wo kmeans", "Wiki w kmeans", "Syn wo kmeans"]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    col_names[0] = 'Average Last Four'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')

def iclr_generate_dt_kmeans_2():
    algs = [
        dt,
        chibiT,
        "chibiT-rerun_wo",
        "chibiT_w_random",
        "chibiT_wo_random",
        "chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_step1_wo",
        "chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_step10_wo",
        "chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_step100_wo",
        "chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_step1000_wo",
        "chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_step10000_wo",
        "chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_step80000_wo",

        ]
    col_names = ['Best Score', 'DT', 'DT+Wiki', 'Wiki wo cos', "Wiki w cos", "wiki wo new", "1 step wo", 
    "10 step wo","100 step wo","1000 step wo","10000 step wo", "80000 step wo"]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    col_names[0] = 'Average Last Four'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')




def iclr_generate_dt_iid_20seeds():
    algs = [
        "dt-rerun-20seeds_dt",
        chibiT,
        "chibiT-rerun_wo",
        "chibiT-iid_wo_step1",
        "chibiT-iid_wo_step8000",
        "chibiT-iid_wo_step10000",
        "chibiT-iid_wo_step20000",
        "chibiT-iid_wo_step40000",
        "chibiT-iid_wo_step80000",
        

        ]
    col_names = ['Best Score', 'DT', 'DT+Wiki', 'DT+Wiki wo' 'Synthetic iid s1', "Synthetic iid s8000", "Synthetic iid s10000", "Synthetic iid s20000", 
    "Synthetic iid s40000","Synthetic iid s80000"]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    col_names[0] = 'Average Last Four'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')



def iclr_generate_dt_syn_20seeds():
    algs = [
        "dt-rerun-20seeds_dt",
        "chibiT-syn-20seeds-steps-wo_step1000",
        "chibiT-syn-20seeds-steps-wo_2_step10000",
        "chibiT-syn-20seeds-steps-wo_step20000",
        "chibiT-syn-20seeds-steps-wo_step40000",
        'chibiT-syn-20seeds-steps-wo_step60000',
        "chibiT-syn-20seeds-steps-wo_2_step80000",
        

        ]
    col_names = ['Best Score', 'DT', '1k updates ', "10k updates", "20k updates", "40k updates", '60k updates', 
    "80k updates"]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    col_names[0] = 'Average Last Four'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')





def iclr_generate_dt_short_table_per_env():
    algs = [
        dt,
        chibiT,
        "chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_10000",
        "chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_20000",
        "chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_30000",
        "chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_40000",
        "chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_50000",
        "chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_60000",
        "chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_70000",
        dt_mc_1step_vocab100,
    ]
    col_names = ['Best Score', 'DT', 'DT+Wiki', 'DT+Synthetic 10K', 'DT+Synthetic 20K', "DT+Synthetic 30K",
    'DT+Synthetic 40K', 'DT+Synthetic 50K', 'DT+Synthetic 60K', 'DT+Synthetic 70K', 'DT+Synthetic 80K']
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    col_names[0] = 'Average Last Four'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')

def iclr_generate_dt_very_short_table_per_env():
    algs = [
        dt,
        "chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_step1",
        "chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_step10",
        "chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_step100",
        "chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_step1000",
        "chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_10000",
        dt_mc_1step_vocab100,
    ]
    col_names = ['Best Score', 'DT', 'DT+Synthetic 1 step', 'DT+Synthetic 10 steps', "DT+Synthetic 100 steps",
    'DT+Synthetic 1000 steps', 'DT+Synthetic 10000 steps', 'DT+Synthetic full']
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    col_names[0] = 'Average Last Four'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')


def iclr_generate_dt_medium_short_table_per_env():
    algs = [
        dt,
        "chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_step1000",
        "chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_step2000",
        "chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_step3000",
        "chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_step4000",
        "chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_step5000",
        "chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_step6000",
        "chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_step7000",
        "chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_step8000",
        "chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_step9000",
        "chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_10000",
        dt_mc_1step_vocab100,
    ]
    col_names = ['Best Score', 'DT', 'DT+Synthetic 1000 step', 'DT+Synthetic 2000 steps', "DT+Synthetic 3000 steps",
    'DT+Synthetic 4000 steps', 'DT+Synthetic 5000 steps', 'DT+Synthetic 6000 steps', 'DT+Synthetic 7000 steps', 'DT+Synthetic 8000 steps', 'DT+Synthetic 9000 steps', 'DT+Synthetic 10000 steps', 'DT+Synthetic full']
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    col_names[0] = 'Average Last Four'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')




def iclr_generate_dt_mc_table_100_states_different_steps():
    algs = [
        dt,
        "chibiT-random2_nvocab100_seed666",
        dt_mc_1step_vocab100,
        dt_mc_2step_vocab100,
        # dt_mc_3step_vocab100,
        # dt_mc_4step_vocab100,
        dt_mc_5step_vocab100,
    ]
    col_names = ['Best Score',
                 'DT',
                 'DT+0-MC'
                 'DT+1-MC', 'DT+2-MC',
                 # '3-MC S100', '4-MC S100',
                 'DT+5-MC']
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    #col_names[0] = 'Average Score'
    #generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')
    #generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')

    #col_names[0] = 'Average Later Half'
    #generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')
    #generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')

    col_names[0] = 'Average Last Four'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')





def iclr_generate_dt_mc_table_100_states_different_steps_20seeds():
    algs = [
        'dt-rerun-20seeds_dt',
        #"chibiT-random2_nvocab100_seed666",
        'chibiT-syn-20seeds-steps-wo_step20000',
        'chibiT-syn-20seeds-steps-wo_MC-2',
        # dt_mc_3step_vocab100,
        # dt_mc_4step_vocab100,
        'chibiT-syn-20seeds-steps-wo_MC-5',
    ]
    col_names = ['Best Score',
                 'DT',
                 '1-MC', '2-MC',
                 # '3-MC S100', '4-MC S100',
                 '5-MC']
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    #col_names[0] = 'Average Score'
    #generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')
    #generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')

    #col_names[0] = 'Average Later Half'
    #generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')
    #generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')

    col_names[0] = 'Average Last Four'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')


def iclr_generate_dt_long():
    algs = [
        dt,
        'dt-long',
        chibiT,
        dt_mc_1step_vocab100,
        'same_new_ft'
    ]
    col_names = ['Best Score',
                 'DT', 'DT 2x steps', 'DT+Wiki',
                 'Synthetic', 'RL pretraining',
                 ]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    col_names[0] = 'Average Score'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')

    col_names[0] = 'Average Later Half'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')



def iclr_generate_dt_long_new():
    algs = [
        dt,
        'dt-long-20seeds_dt_24',
        'dt-long-20seeds_dt_36',
        'chibiT-rerun-20seeds',
        'chibiT-syn-20seeds-steps-wo_step20000',
    ]
    col_names = ['Best Score',
                 'DT', 'DT 20K more', 'DT 80K more', 'DT+Wiki',
                 'DT+Synthetic'
                 ]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    #col_names[0] = 'Average Score'
    #generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')
    #generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')

    #col_names[0] = 'Average Later Half'
    #generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')
    #generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')

    col_names[0] = 'Average Last Four'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')




def iclr_generate_dt_random_different_states():
    algs = [
        dt,
        chibiT,
        'chibiT-random_nvocab10',
        'chibiT-random_nvocab100',
        'chibiT-random_nvocab1000',
        'chibiT-random1_nvocab10',
        'chibiT-random1_nvocab100',
        'chibiT-random1_nvocab1000',  
    ]
    col_names = ['Best Score',
                 'DT', 'DT+Wiki',
                 'Random S10','Random S100','Random S1000','Random Alt S10','Random Alt S100','Random Alt S1000',]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    col_names[0] = 'Average Score'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')

    col_names[0] = 'Average Later Half'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')

    col_names[0] = 'Average Last Four'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')


def iclr_generate_dt_random_different():
    algs = [
        dt,
        chibiT,
        dt_mc_1step_vocab100,
        "chibiT-random2_nvocab100_seed666",
        "chibiT-random_nvocab100",
        "random_same",
        "random_itself",
        "random_short",
        "random_short_same",
        "random_short_itself"
    ]
    col_names = ['Best Score',
                 'DT', 'DT+Wiki',
                 'DT+Synthetic','DT+Synthetic 0-step','DT+random','DT+same','DT+itself','DT+short random',"DT+short same","DT+short itself"]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    col_names[0] = 'Average Score'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')

    col_names[0] = 'Average Later Half'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')

    col_names[0] = 'Average Last Four'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')


def iclr_generate_dt_more_seeds():
    algs = [
        dt,
        chibiT,
        dt_mc_1step_vocab100,
        "chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_seed42",
        "chibiT-random2_nvocab100_seed42",
        'chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_shuffled',
        'same-no-return_new_ft',
        'same-no-return-shuffled_ft',
        'chibiT-random1_nvocab100',

    ]
    col_names = ['Best Score',
                 'DT', 'DT+Wiki', '1-MC 3',
                 '1-MC 9', '0-step 9',
                 '1-MC shuffle','same data', 'same data shuffle',"Bad 0-step 9"]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    col_names[0] = 'Average Score'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')

    col_names[0] = 'Average Later Half'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')

def iclr_generate_dt_early_checkpoint():
    algs = [
        dt,
        chibiT,
        dt_mc_1step_vocab100,
        'chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_30000'
        #"chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_seed42",
        #"chibiT-random2_nvocab100_seed42",
        #'chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_shuffled',
        #'same-no-return_new_ft',
        #'same-no-return-shuffled_ft',
        #'chibiT-random1_nvocab100',

    ]
    col_names = ['Best Score',
                 'DT', 'DT+Wiki', '1-MC 3', '1-MC early stop']
                 #'1-MC 9', '0-step 9',
                 #'1-MC shuffle','same data', 'same data shuffle',"Bad 0-step 9"]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    col_names[0] = 'Average Score'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')

    col_names[0] = 'Average Later Half'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')



def iclr_generate_dt_separate_data():
    algs = [
        'chibiT-random2_nvocab100_seed42',
        'chibiT-random2_nvocab100_seed1024',
        'chibiT-random2_nvocab100_seed666',
        'chibiT-random1_nvocab100_seed42',
        'chibiT-random1_nvocab100_seed1024',
        'chibiT-random1_nvocab100',
        'chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_seed42',
        'chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_seed1024',
        'chibiT-rerun-syn_ngram1_nvocab100_temperature1.0_seed666',
    ]
    col_names = ['Best Score',
                 '0-step s42', '0-step s1024', '0-step s666',
                 'Bad s42', 'Bad s1024', 'Bad s666', 
                 '1-step MC s42', '1-step MC s1024', '1-step MC s666', ]    
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    col_names[0] = 'Average Score'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')

    col_names[0] = 'Average Later Half'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')



def iclr_generate_dt_mc_table_different_number_of_states():
    algs = [
        dt,
        "chibiT-rerun-syn_ngram1_nvocab1_temperature1.0",
        dt_mc_1step_vocab10,
        dt_mc_1step_vocab100,
        dt_mc_1step_vocab1000,
        dt_mc_1step_vocab10000,
        dt_mc_1step_vocab100000,
    ]
    col_names = ['Best Score',
                 'DT',
                 '1-MC S1',
                 '1-MC S10', '1-MC S100','1-MC S1000', '1-MC S10000', '1-MC S100000']
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    col_names[0] = 'Average Score'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')

    col_names[0] = 'Average Later Half'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')

    col_names[0] = 'Average Last Four'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')


def iclr_generate_dt_mc_table_different_number_of_states_20seeds():
    algs = [
        'dt-rerun-20seeds_dt',
        #"chibiT-rerun-syn_ngram1_nvocab1_temperature1.0",
        #dt_mc_1step_vocab10,
        #dt_mc_1step_vocab100,
        #dt_mc_1step_vocab1000,
        #dt_mc_1step_vocab10000,
        #dt_mc_1step_vocab100000,
        'chibiT-syn-20seeds-states-wo_S10',
        'chibiT-syn-20seeds-steps-wo_step20000',
        'chibiT-syn-20seeds-states-wo_S1000',
        'chibiT-syn-20seeds-states-wo_S10000',
        'chibiT-syn-20seeds-states-wo_S100000',
    ]
    col_names = ['Best Score',
                 'DT',
                 'S10', 'S100','S1000', 'S10000', 'S100000']
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    
    col_names[0] = 'Average Last Four'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')



def iclr_generate_dt_mc_table_different_temperatures():
    algs = [
        dt,
        "chibiT-rerun-syn_ngram1_nvocab100_temperature0.001",
        "chibiT-rerun-syn_ngram1_nvocab100_temperature0.01",
        dt_mc_temp0_1_vocab100,
        #dt_mc_temp0_2_vocab100,
        #dt_mc_temp0_4_vocab100,
        # dt_mc_temp0_8_vocab100,
        dt_mc_1step_vocab100,
        dt_mc_temp10_0_vocab100,
        "chibiT-random_nvocab100"
        #"chibiT-rerun-syn_ngram1_nvocab100_temperature100.0",
    ]
    col_names = ['Best Score',
                 'DT',
                 '1-MC S100 t0.001', '1-MC S100 t0.01', '1-MC S100 t0.1', '1-MC S100 t1', '1-MC S100 t10','IID uniform']
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    col_names[0] = 'Average Score'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')

    col_names[0] = 'Average Later Half'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')

    col_names[0] = 'Average Last Four'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')


def iclr_generate_dt_mc_table_different_temperatures_20seeds():
    algs = [
        'dt-rerun-20seeds_dt',
        'chibiT-syn-20seeds-temps-wo_T0.01',
        'chibiT-syn-20seeds-temps-wo_T0.1',
        'chibiT-syn-20seeds-steps-wo_step20000',
        'chibiT-syn-20seeds-temps-wo_T10.0',
        'chibiT-syn-20seeds-temps-wo_T100.0',
        'chibiT-iid_wo_step20000'

    ]
    col_names = ['Best Score',
                 'DT',
                 '\\tau = 0.01', '\\tau = 0.1', '\\tau = 1', '\\tau = 10', '\\tau = 100', 'IID uniform']
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    

    col_names[0] = 'Average Last Four'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')



def iclr_generate_dt_mc_table_data_size():
    algs = [
        'dt-rerun-20seeds_dt',
        "chibiT-syn-20seeds-ratio-wo_ratio0.125",
        "chibiT-syn-20seeds-ratio-wo_ratio0.25",
        "chibiT-syn-20seeds-ratio-wo_ratio0.375",
        "chibiT-syn-20seeds-ratio-wo_ratio0.5",
        "chibiT-syn-20seeds-ratio-wo_ratio0.625",
        "chibiT-syn-20seeds-ratio-wo_ratio0.75",
        "chibiT-syn-20seeds-ratio-wo_ratio0.875",
        'chibiT-syn-20seeds-steps-wo_step20000',

            ]
    col_names = ['Best Score',
                 'DT',
                 '12.5\%', '25\%', '37.5\%', '50\%', '62.5\%', '75\%', '87.5\%', '100\%' ]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    col_names[0] = 'Average Last Four'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_last_four_normalized')



def iclr_generate_cql_section_table():
    algs = [
        cql_jul,
        cql_jul_mdp_noproj_s100_t1,
    ]
    col_names = ['Best Score',
                 'CQL',
                 'CQL MDP'
                 ]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    col_names[0] = 'Average Score'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')

    col_names[0] = 'Average Later Half'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')

def iclr_generate_dt_first_table_per_walker_2():
    algs = [
        dt,
        chibiT,
        dt_mc_1step_vocab100,
        "same-no-return_new_ft"
    ]
    col_names = ['Best Score', 'DT', 'DT+Wiki', '1-MC S100', 'RL pretraining no return']
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    col_names[0] = 'Average Score'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')

    col_names[0] = 'Average Later Half'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')



def iclr_generate_cql_mdp_compare_n_state():
    # algs = [
    #     cql_base,
    #     cql_mdp_pretrain_nstate1,
    #     cql_mdp_pretrain_nstate10,
    #     cql_mdp_pretrain_nstate100,
    #     cql_mdp_pretrain_nstate1000,
    #     cql_mdp_pretrain_nstate10000,
    # ]
    algs = [
        cql_jul,
        cql_jul_mdp_noproj_s1_t1,
        cql_jul_mdp_noproj_s10_t1,
        cql_jul_mdp_noproj_s100_t1,
        cql_jul_mdp_noproj_s1000_t1,
        cql_jul_mdp_noproj_s10000_t1,
        cql_jul_mdp_noproj_s100000_t1,
    ]
    col_names = ['Best Score',
                 'CQL',
                 'CQL MDP S1',
                 'CQL MDP S10',
                 'CQL MDP S100',
                 'CQL MDP S1000',
                 'CQL MDP S10000',
                 'CQL MDP S100000',
                 ]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    col_names[0] = 'Average Score'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')

    col_names[0] = 'Average Later Half'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')



def iclr_generate_cql_mdp_compare_temp():
    algs = [
        cql_base,
        cql_jul_mdp_noproj_s100_t0_0001,
        cql_jul_mdp_noproj_s100_t0_001,
        # cql_jul_mdp_noproj_s100_t0_01,
        cql_jul_mdp_noproj_s100_t0_1,
        cql_jul_mdp_noproj_s100_t1,
        # cql_jul_mdp_noproj_s100_t10,
        cql_jul_mdp_noproj_s100_t100,
        cql_jul_mdp_noproj_s100_t1000,
    ]
    col_names = ['Best Score',
                 'CQL',
                 'S100 t0.0001',
                 'S100 t0.001',
                 # 'S100 t0.01',
                 'S100 t0.1',
                 'S100 t1',
                 # 'S100 t10',
                 'S100 t100',
                 'S100 t1000',
                 ]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    col_names[0] = 'Average Score'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')

    col_names[0] = 'Average Later Half'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')


def iclr_generate_same_task_pretrain():
    algs = [
        cql_jul,
        cql_jul_mdp_noproj_s100_t1,
        cql_jul_fd_pretrain,
    ]
    col_names = ['Best Score',
                 'CQL',
                 'CQL MDP S100 t1',
                 'CQL same task',
                 ]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    col_names[0] = 'Average Score'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')

    col_names[0] = 'Average Later Half'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')


def july_test_only1():
    algs = [
        cql_base,
        cql_fd_pretrain,
        cql_jul,
        cql_jul_fd_pretrain,
    ]
    col_names = ['Best Score',
                 'CQL',
                 'CQL same task',
                 'CQL jul',
                 'CQL same task jul',
                 ]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    col_names[0] = 'Average Score'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')




##################### table generation
# generate_table_nvocab_markov_chain()
# generate_table_markov_chain_compare_number_of_steps()
# generate_dt_mc_table_compare_temperature()
# generate_dt_mc_table_more1()
# generate_dt_mc_table_more2()

# generate_table_cql_cross_domain()

# generate_table_cql_mdp_compare_n_state_action()
# generate_table_cql_mdp_compare_temperature()
# generate_table_cql_mdp_compare_state_action_dim()
# generate_table_cql_mdp_same_dim_with_and_no_projection()

# generate_table_cql_3x_data()
# generate_table_no_action_predict_next_state()

# TODO following all re-table 6-25 morning
# generate_table_cql_pretrain_data_sizes()
# generate_table_cql_mdp_pretrain_data_sizes()
# generate_table_cql_finetune_data_sizes()
# generate_table_mdp_pretrain_finetune_data_sizes()
# generate_table_cql_finetune_both_data_sizes()
# generate_table_cql_no_pretrain_finetune_data_sizes()



# generate_table_cql_with_target_networks()
# generate_table_cql_with_target_networks2()
# generate_table_cql_with_target_networks3()
# generate_table_cql_with_target_networks4()

# TODO ICLR paper run these
# iclr_generate_dt_first_table_per_env()
# iclr_generate_dt_mc_table_100_states_different_steps()
# iclr_generate_dt_mc_table_different_number_of_states()
# iclr_generate_dt_mc_table_different_temperatures()


# iclr_generate_cql_section_table()
# iclr_generate_cql_mdp_compare_n_state()
# iclr_generate_cql_mdp_compare_temp()
# iclr_generate_same_task_pretrain()
#iclr_generate_dt_first_table_per_walker_2()
# iclr_generate_dt_random_different()


#iclr_generate_dt_first_table_per_env()
#iclr_generate_dt_first_table_per_env_20seeds()
#iclr_generate_dt_first_table_per_env_conv_iter_20seeds()
#iclr_generate_dt()
#iclr_generate_dt_first_table_per_env_conv_iter()
#iclr_generate_dt_mc_table_100_states_different_steps()
#iclr_generate_dt_mc_table_100_states_different_steps_20seeds()
#iclr_generate_dt_mc_table_different_number_of_states()
#iclr_generate_dt_mc_table_different_number_of_states_20seeds()
#iclr_generate_dt_mc_table_different_temperatures()
#iclr_generate_dt_mc_table_different_temperatures_20seeds()
#iclr_generate_dt_random_different_states()
#iclr_generate_dt_long()
#iclr_generate_dt_short_table_per_env()
#iclr_generate_dt_very_short_table_per_env()
#iclr_generate_dt_mc_table_data_size()
#iclr_generate_dt_medium_short_table_per_env()
#iclr_generate_dt_kmeans()
#iclr_generate_dt_kmeans_2()
#iclr_generate_dt_iid_20seeds()
#iclr_generate_dt_syn_20seeds()
#iclr_generate_dt_more_seeds()
#iclr_generate_dt_separate_data()
#iclr_generate_dt_early_checkpoint()
#iclr_generate_dt_long_new()
#iclr_generate_dt_first_table_per_env_20seeds_diff_pretrain()

#iclr_generate_dt_analysis()
iclr_generate_dt_analysis_agg()