import os.path
import numpy as np
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# use this to generate plots for example x-axis is data size, y axis is best performance

# TODO
#  function call looks like this:
#  get mean and std ([data1, data2, data3, data4], best return)
#  and then put the mean and std into a plotting function

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
plot_measures = ['final_test_normalized_returns', 'best_return_normalized',
                 'final_weight_diff', 'best_weight_diff',
                 'final_feature_diff', 'best_feature_diff',
                 'final_test_normalized_returns_std', 'best_return_normalized_std',
                 'convergence_iter',
                 ]
plot_y_labels = ['Final Normalized Score', 'Best Normalized Score',
                 'Final Weight l2 Diff', 'Best Weight l2 Diff',
                 'Final Feature L2 Diff', 'Best Feature L2 Diff',
                 'Final Normalized Score Std', 'Best Normalized Score Std',
                 'Convergence Iter',
                 ]
def get_y_best_final(y): # TODO work on this later
    d = {
        'both_normalized_returns': ['best_return_normalized', 'final_test_normalized_returns'],
    }
    return d[y]

def get_extra_dict_multiple_seeds(datafolder_path):
    # for a alg-dataset variant, obtain a dictionary with key-value pairs as measure:[avg across seeds, std across seeds]
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

def get_value_list(alg_dataset_dict, alg, measure):
    # for an alg-measure pair, aggregate over all datasets
    value_list = []
    for dataset, extra_dict in alg_dataset_dict[alg].items():
        value_list.append(extra_dict[measure][0]) # each entry is the value from a dataset
    return value_list

def get_values_df(algs, alg_dataset_dict, measure='best_return_normalized'):
    xs = []
    values = []
    for i, alg in enumerate(algs):
        new_values = get_value_list(alg_dataset_dict, alg, measure)
        xs += [i] * len(new_values)
        values += new_values
    data = {'x':xs, 'y':values}
    return pd.DataFrame(data)

def get_all_mujoco_env_datasets():
    MUJOCO_3_ENVS = ['hopper', 'walker2d', 'halfcheetah', ]
    MUJOCO_3_DATASETS = ['medium', 'medium-replay', 'medium-expert', ]
    envs = []
    for e in MUJOCO_3_ENVS:
        for dataset in MUJOCO_3_DATASETS:
            envs.append('%s_%s' % (e, dataset))
    return envs

def get_alg_dataset_dict(algs, envs):
    alg_dataset_dict = {}
    for alg in algs:
        alg_dataset_dict[alg] = {}
        for env in envs:
            folderpath = os.path.join(data_path, '%s_%s' % (alg, env))
            alg_dataset_dict[alg][env] = get_extra_dict_multiple_seeds(folderpath)
    return alg_dataset_dict

def get_xaxis_variants_figure(labels, algs_list, colors, envs, x_ticks, x_label, y_list, y_label_list,
                              save_name_prefix, verbose=True,
                              legend_loc='best', no_legend=False,
                              legend_font_size=24, axis_label_font_size=20, tick_font_size=16,
                              ):
    algs_all = []
    for algs in algs_list:
        algs_all += algs
    alg_dataset_dict = get_alg_dataset_dict(algs_all, envs)

    errorbar = None
    # make a plot for each y
    for y, y_label in zip(y_list, y_label_list): # maybe use y to control whether solid lines or solid+dashed
        if y in all_measures:
            for label, algs, color in zip(labels, algs_list, colors):
                df = get_values_df(algs, alg_dataset_dict, y)
                sns.lineplot(data=df, x='x', y='y', marker='o', color=color, linewidth=2, errorbar=errorbar, label=label)
            # else we do y_best and y_final
        else:
            y_best, y_final = get_y_best_final(y)
            for label, algs, color in zip(labels, algs_list, colors):
                df_best = get_values_df(algs, alg_dataset_dict, y_best)
                df_final = get_values_df(algs, alg_dataset_dict, y_final)
                sns.lineplot(data=df_best, x='x', y='y', marker='o', color=color, linewidth=2, errorbar=errorbar, label=label)
                sns.lineplot(data=df_final, x='x', y='y', marker='o', color=color, linewidth=2, errorbar=errorbar, linestyle='--', label='_')

        # Customize the plot
        plt.xticks(np.arange(0, len(x_ticks)), x_ticks, fontsize=tick_font_size)
        plt.yticks(fontsize=tick_font_size)
        plt.xlabel(x_label, fontsize=axis_label_font_size)
        plt.ylabel(y_label, fontsize=axis_label_font_size)
        if not no_legend:
            plt.legend(loc=legend_loc, fontsize=legend_font_size)
        plt.tight_layout()

        save_folder_path_with_y = os.path.join(save_folder_path, y)
        if save_folder_path is not None:
            if not os.path.isdir(save_folder_path_with_y):
                path = Path(save_folder_path_with_y)
                path.mkdir(parents=True)
            save_path_full = os.path.join(save_folder_path_with_y, save_name_prefix + '_' + y + '.png')
            plt.savefig(save_path_full)
            if verbose:
                print(save_path_full)
            plt.close()
        else:
            plt.show()

    return

def plot_cql_pretrain_epochs():
    save_name_prefix = 'cql_pretrain_epochs'
    labels = ['CQL Pretrain',]
    algs_list = [['cql_prenone','cql_preq_sprime_pe2', 'cql_preq_sprime_pe20', 'cql_preq_sprime',],
                 ]
    colors = ['tab:blue',]
    envs = get_all_mujoco_env_datasets()
    x_ticks = ['0', '2', '20', '200']
    x_label = 'Number of Pretrain Epochs'
    y_list = plot_measures
    y_label_list = plot_y_labels
    get_xaxis_variants_figure(labels, algs_list, colors, envs, x_ticks, x_label, y_list, y_label_list, save_name_prefix)


def plot_cql_compare_layers():
    save_name_prefix = 'cql_layers'
    labels = ['CQL', 'CQL Pretrain']
    algs_list = [['cql_prenone','cql_prenone_pe200_layer4', 'cql_prenone_pe200_layer6', 'cql_prenone_pe200_layer8',],
                 ['cql_preq_sprime', 'cql_preq_sprime_pe200_layer4', 'cql_preq_sprime_pe200_layer6', 'cql_preq_sprime_pe200_layer8']]
    colors = ['tab:blue', 'tab:orange']
    envs = get_all_mujoco_env_datasets()
    x_ticks = ['2 layers', '4 layers', '6 layers', '8 layers']
    x_label = 'Number of Layers'
    y_list = plot_measures
    y_label_list = plot_y_labels
    get_xaxis_variants_figure(labels, algs_list, colors, envs, x_ticks, x_label, y_list, y_label_list, save_name_prefix)

def plot_dt_rl_datasize(): # TODO currently only DTx2, later add other variants
    save_name_prefix = 'dt_rl_datasize'
    labels = ['DT', 'ChibiT']
    algs_list = [
        ['dt_data_size0.1', 'dt_data_size0.25', 'dt_data_size0.5', 'dt_data_size0.75', 'dt',],
        ['chibiT_data_size0.1', 'chibiT_data_size0.25', 'chibiT_data_size0.5', 'chibiT_data_size0.75', 'chibiT'],
                 ]
    colors =  ['tab:blue', 'tab:orange']
    envs = get_all_mujoco_env_datasets()
    x_ticks = ['10%', '25%', '50%', '75%', '100%']
    x_label = 'RL Data Ratio'
    y_list = plot_measures
    y_label_list = plot_y_labels
    get_xaxis_variants_figure(labels, algs_list, colors, envs, x_ticks, x_label, y_list, y_label_list, save_name_prefix)

def plot_dt_modelsize(): # TODO currently only DT with no pretrain # TODO also might want to compute num of parameters
    save_name_prefix = 'dt_modelsize'
    labels = ['DT',]
    algs_list = [
        ['dt', 'dt_embed_dim256_n_layer4_n_head4', 'dt_embed_dim512_n_layer6_n_head8', 'dt_embed_dim768_n_layer12_n_head12',],
                 ]
    colors =  ['tab:blue',]
    envs = get_all_mujoco_env_datasets()
    x_ticks = ['3L 128D', '4L 256D', '6L 512D', '12L 768D', ]
    x_label = 'DT Model Size'
    y_list = plot_measures
    y_label_list = plot_y_labels
    get_xaxis_variants_figure(labels, algs_list, colors, envs, x_ticks, x_label, y_list, y_label_list, save_name_prefix)

def plot_dt_perturb(): # TODO currently only DT later add other variants?
    save_name_prefix = 'dt_pretrain_perturb'
    labels = ['ChibiT',]
    algs_list = [
        ['chibiT','chibiT_perturb_per_layer1e-1', 'chibiT_perturb_per_layer1e0',
         'chibiT_perturb_per_layer2e0','chibiT_perturb_per_layer4e0', 'chibiT_perturb_per_layer8e0'],
    ]
    colors =  ['tab:blue',]
    envs = get_all_mujoco_env_datasets()
    x_ticks = ['0', '0.1', '1', '2', '4', '8']
    x_label = 'Perturb Noise Std'
    y_list = plot_measures
    y_label_list = plot_y_labels
    get_xaxis_variants_figure(labels, algs_list, colors, envs, x_ticks, x_label, y_list, y_label_list, save_name_prefix)

data_path = '../../code/checkpoints/'
save_folder_path = '../../figures/'

plot_cql_pretrain_epochs()
plot_cql_compare_layers()
plot_dt_rl_datasize()
plot_dt_modelsize()
plot_dt_perturb()

