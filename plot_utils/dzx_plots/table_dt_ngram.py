import os.path
import numpy as np
import pandas as pd
import json
# use this to generate the main table
from plot_utils.log_alias import *

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

         'best_5percent_normalized',
         'best_10percent_normalized',
         'best_25percent_normalized',
         'best_50percent_normalized',
         'best_100percent_normalized',
         'best_later_half_normalized'
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
                    'best_50percent_normalized', 'best_100percent_normalized', 'best_later_half_normalized'
                    ]:
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
def generate_aggregate_table(algs, alg_dataset_dict, column_names, best_value_bold=True, bold_threshold=0.05):
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


def generate_aggregate_performance(algs, alg_dataset_dict, column_names, best_value_bold=True, bold_threshold=0.05,
                                   measure='best_return_normalized', row_name='Average (All Settings)'):
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
                if mean >= (1-bold_threshold)*max_values[i]:
                    bold = True
                if bold:
                    row_string += (' & \\textbf{%.1f} $\pm$ %.1f' % (mean, std))
                else:
                    row_string += (' & %.1f $\pm$ %.1f' % (mean, std))
        row_string += '\\\\'
        print(row_string)

# TODO add an aggregate score at the end
def generate_per_env_score_table_new(algs, alg_dataset_dict, column_names, best_value_bold=True, bold_threshold=0.05, measure='best_return_normalized'):
    print("\nNow generate latex table:\n")
    # measure = 'best_100percent_normalized'
    # each row is a env-dataset pair, each column is an algorithm variant
    rows = []
    row_names = []
    for dataset in ['medium-expert', 'medium', 'medium-replay', ]:
        for e in ['halfcheetah', 'hopper', 'walker2d', ]:
            rows.append('%s_%s' % (e, dataset))
            row_names.append('%s-%s' % (e, dataset))

    table = np.zeros((2, len(rows), len(algs)))
    # each iter we generate a row
    for i, row in enumerate(rows):
        for j, alg in enumerate(algs):
            try:
                table[0,i,j], table[1,i,j] = alg_dataset_dict[alg][row][measure]
            except:
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
                if mean >= (1-bold_threshold)*max_values[i]:
                    bold = True
                if bold:
                    row_string += (' & \\textbf{%.1f} $\pm$ %.1f' % (mean, std))
                else:
                    row_string += (' & %.1f $\pm$ %.1f' % (mean, std))
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
    # generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized', row_name='Average Test (All)')

    col_names = ['Average Score', 'DT', 'DT+Wiki', '1-MC S100']
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')

def iclr_generate_dt_mc_table_100_states_different_steps():
    algs = [
        dt,
        dt_mc_1step_vocab100,
        dt_mc_2step_vocab100,
        # dt_mc_3step_vocab100,
        # dt_mc_4step_vocab100,
        dt_mc_5step_vocab100,
    ]
    col_names = ['Best Score',
                 'DT',
                 '1-MC S100', '2-MC S100',
                 # '3-MC S100', '4-MC S100',
                 '5-MC S100']
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    col_names[0] = 'Average Score'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')


def iclr_generate_dt_mc_table_different_number_of_states():
    algs = [
        dt,
        dt_mc_1step_vocab10,
        dt_mc_1step_vocab100,
        dt_mc_1step_vocab1000,
        dt_mc_1step_vocab10000,
        dt_mc_1step_vocab100000,
    ]
    col_names = ['Best Score',
                 'DT',
                 '1-MC S10', '1-MC S100','1-MC S1000', '1-MC S10000', '1-MC S100000']
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    col_names[0] = 'Average Score'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')



def iclr_generate_dt_mc_table_different_temperatures():
    algs = [
        dt,
        dt_mc_temp0_1_vocab100,
        dt_mc_temp0_2_vocab100,
        dt_mc_temp0_4_vocab100,
        # dt_mc_temp0_8_vocab100,
        dt_mc_1step_vocab100,
        dt_mc_temp10_0_vocab100,
    ]
    col_names = ['Best Score',
                 'DT',
                 '1-MC S100 t0.1', '1-MC S100 t0.2', '1-MC S100 t0.4', '1-MC S100 t1', '1-MC S100 t10',]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    col_names[0] = 'Average Score'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')


def iclr_generate_dt_same_data_table():
    algs = [
        dt,
        chibiT,
        dt_mc_1step_vocab100,
        dt_same_data,
    ]
    col_names = ['Best Score',
                 'DT',
                 'DT + Wiki', 'DT + synthetic', 'DT + same data']
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)

    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    # col_names[0] = 'Average Score'
    # generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')
    # generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_100percent_normalized')

    col_names[0] = 'Average 2nd Half'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')



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
        # cql_jul_mdp_noproj_s100_t0_0001,
        cql_jul_mdp_noproj_s100_t0_001,
        cql_jul_mdp_noproj_s100_t0_01,
        cql_jul_mdp_noproj_s100_t0_1,
        cql_jul_mdp_noproj_s100_t1,
        cql_jul_mdp_noproj_s100_t10,
        cql_jul_mdp_noproj_s100_t100,
        # cql_jul_mdp_noproj_s100_t1000,
    ]
    col_names = ['Best Score',
                 'CQL',
                 # 'S100 t0.0001',
                 't0.001',
                 't0.01',
                 't0.1',
                 't1',
                 't10',
                 't100',
                 # 'S100 t1000',
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



def iclr_generate_cql_mdp_compare_temp_with_random():
    algs = [
        cql_base,
        cql_jul_mdp_noproj_s100_t1,
        cql_jul_mdp_noproj_s100_t1000,
        cql_random_pretrain,
    ]
    col_names = ['Best Score',
                 'CQL',
                 'S100 t1',
                 'S100 t1000',
                 'Sinf t1000',
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
                 'CQL same data',
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


def iclr_generate_cql_iid():
    algs = [
        cql_1x,
        cql_same_new,
        cql_mdp_t1,
        cql_mdp_t1000
    ]
    col_names = ['Best Score',
                 'CQL',
                 'S1',
                 'S100 t0.0001',
                 'S100 t1',
                 'S100 t1000',
                 'S100000 t1',
                 ]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    col_names[0] = 'Average Later Half'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')



def iclr_data_ratio1():
    algs = [
        cql_jul_pdr1_dr0_001,
        cql_jul_pdr1_dr1,
        cql_jul_same_data_pdr1_dr0_001,
        cql_jul_same_data_pdr1_dr1,
        cql_jul_mdp_pdr1_dr0_001,
        cql_jul_mdp_pdr1_dr1,
    ]
    col_names = ['Best Score',
                 'cql finetune 0.1\\%',
                 'cql finetune 100\\%',
                 'same finetune 0.1\\%',
                 'same finetune 100\\%',
                 'mdp finetune 0.1\\%',
                 'mdp finetune 100\\%',
                 ]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    col_names[0] = 'Average Later Half'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')


def iclr_data_ratio2():
    algs = [
        cql_jul_pdr1_dr1,
        cql_jul_same_data_pdr0_001_dr1,
        cql_jul_same_data_pdr1_dr1,
        cql_jul_mdp_pdr0_001_dr1,
        cql_jul_mdp_pdr1_dr1,
    ]
    col_names = ['Best Score',
                 'cql',
                 'same pretrain 0.1\\%',
                 'same pretrain 100\\%',
                 'mdp pretrain 0.1\\%',
                 'mdp pretrain 100\\%',
                 ]
    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names)

    col_names[0] = 'Average Later Half'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='best_later_half_normalized')


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

# test only
# iclr_generate_cql_mdp_compare_temp_with_random()


# TODO ICLR paper run these
# iclr_generate_dt_first_table_per_env()
# iclr_generate_dt_mc_table_100_states_different_steps()
# iclr_generate_dt_mc_table_different_number_of_states()
# iclr_generate_dt_mc_table_different_temperatures()

# iclr_generate_dt_same_data_table()


# iclr_generate_cql_section_table()
# iclr_generate_cql_mdp_compare_n_state()
# iclr_generate_cql_mdp_compare_temp()
# iclr_generate_same_task_pretrain()


# TODO after july 10
iclr_generate_cql_iid()
# iclr_data_ratio1()
# iclr_data_ratio2()
