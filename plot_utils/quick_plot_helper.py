import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import OrderedDict
from plot_utils.redq_plot_helper import *
from pathlib import Path

# the path leading to where the experiment file are located
DEFAULT_BASE_PATH = '../code/checkpints/'
DEFAULT_SAVE_PATH = '../figures/'
DEFAULT_ENVS = ('hopper',)
DEFAULT_LINESTYLES = tuple(['solid' for _ in range(8)])
DEFAULT_COLORS = ('tab:red', 'tab:orange', 'tab:blue', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:green', 'tab:purple')
DEFAULT_Y_VALUE = 'AverageTestEpRet'
DEFAULT_SMOOTH = 1
y_to_y_label = {
    'TestEpNormRet':'Normalized Score',
    'TestEpRet': 'Score',
    'total_time': 'Hours',
'sac_qf1_loss': 'Standard Q Loss',
'current_itr_train_loss_mean': 'DT Training Loss',
}

# TODO we provide a number of things to a function to do plotting
#  labels, the data folder for each label, the colors, the dashes
#  and then, the y label to use
#  and also what name to save and where to save (we put default values as macros in a certain file? )

def do_smooth(x, smooth):
    y = np.ones(smooth)
    z = np.ones(len(x))
    smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
    return smoothed_x

def combine_data_in_seeds(seeds, column_name, skip=0, smooth=1):
    vals_list = []
    for d in seeds:
        if isinstance(d,pd.DataFrame):
            vals_to_use = d[column_name].to_numpy().reshape(-1)
        else:
            vals_to_use = d[column_name].reshape(-1)
        # yhat = savgol_filter(vals_to_use, 21, 3)  # window size 51, polynomial order 3
        # TODO might want to use sth else...
        if smooth > 1 and smooth <= len(vals_to_use):
            yhat = do_smooth(vals_to_use, smooth)
        else:
            yhat = vals_to_use

        if skip > 1:
            yhat = yhat[::skip]
        vals_list.append(yhat)
    return np.concatenate(vals_list)

def quick_plot(labels, data_folders, colors=DEFAULT_COLORS, linestyles=DEFAULT_LINESTYLES, envs=DEFAULT_ENVS, base_data_folder_path=DEFAULT_BASE_PATH,
               save_name='test_save_figure', save_folder_path=DEFAULT_SAVE_PATH, y_value=DEFAULT_Y_VALUE, verbose=True, ymin=None, ymax=None):
    # this plots
    label2seeds = OrderedDict()
    for env in envs:
        data_folders_with_env = []
        for data_folder in data_folders:
            data_folders_with_env.append(data_folder + '_' + env)

        for i, label in enumerate(labels): # for each variant
            seeds = []
            data_folder_full_path = os.path.join(base_data_folder_path, data_folders_with_env[i])
            print("check data folder:", data_folder_full_path)
            try:
                for subdir, dirs, files in os.walk(data_folder_full_path):
                    if 'progress.txt' in files:
                        progress_file_path = os.path.join(subdir, 'progress.txt')
                    elif 'progress.csv' in files:
                        progress_file_path = os.path.join(subdir, 'progress.csv')
                    else:
                        continue
                    # load progress file
                    seeds.append(pd.read_table(progress_file_path))
                if len(seeds) > 0:
                    print("Loaded %d seeds from: %s" % (len(seeds), data_folder_full_path))
                else:
                    print("No seed loaded from: %s" % data_folder_full_path)
            except Exception as e:
                print("Failed to load data from:", data_folder_full_path)
                print(e)
            label2seeds[label] = seeds

        save_name_with_env = save_name + '_' + env
        if not isinstance(y_value, list):
            y_value = [y_value,]

        for y_to_plot in y_value:
            for i, (label, seeds) in enumerate(label2seeds.items()):
                x = combine_data_in_seeds(seeds, 'Steps')
                y = combine_data_in_seeds(seeds, y_to_plot, smooth=DEFAULT_SMOOTH)
                if y_to_plot == 'total_time':
                    y = y / 3600
                ax = sns.lineplot(x=x, y=y, n_boot=20, label=label, color=colors[i], linestyle=linestyles[i], linewidth = 2)
            plt.xlabel('Number of Data')
            y_label = y_to_y_label[y_to_plot] if y_to_plot in y_to_y_label else y_to_plot
            plt.ylabel(y_label)
            ax.set_ylim([ymin, ymax])
            plt.tight_layout()
            plt.tight_layout()
            save_folder_path_with_y = os.path.join(save_folder_path, y_to_plot)
            if save_folder_path is not None:
                if not os.path.isdir(save_folder_path_with_y):
                    path = Path(save_folder_path_with_y)
                    path.mkdir(parents=True)
                save_path_full = os.path.join(save_folder_path_with_y, save_name_with_env + '_' + y_to_plot + '.png')
                plt.savefig(save_path_full)
                if verbose:
                    print(save_path_full)
                plt.close()
            else:
                plt.show()

def quick_plot_with_full_name(labels, data_folder_full_names, colors=DEFAULT_COLORS, linestyles=DEFAULT_LINESTYLES, base_data_folder_path=DEFAULT_BASE_PATH,
                              save_name_prefix='test_save_figure', save_name_suffix=None, save_folder_path=DEFAULT_SAVE_PATH,
                              y_value=DEFAULT_Y_VALUE, verbose=True, ymin=None, ymax=None,
                              y_use_log=None, x_to_use='Steps', xlabel='Number of Updates', axis_font_size=10,
                              y_log_scale=False, x_always_scientific=True, smooth=1):
    # this plots
    label2seeds = OrderedDict()
    for label, data_folder_full_name_list in zip(labels, data_folder_full_names):
        seeds_all = []
        if not isinstance(data_folder_full_name_list, list):
            data_folder_full_name_list = [data_folder_full_name_list,]
        for full_name in data_folder_full_name_list:
            seeds = []
            full_path = os.path.join(base_data_folder_path, full_name)
            print("check data folder:", full_path)
            for subdir, dirs, files in os.walk(full_path):
                if 'progress.txt' in files:
                    progress_file_path = os.path.join(subdir, 'progress.txt')
                elif 'progress.csv' in files:
                    progress_file_path = os.path.join(subdir, 'progress.csv')
                else:
                    continue
                # load progress file
                try:
                    seeds.append(pd.read_table(progress_file_path))
                except:
                    print("Failed to load from progress:", progress_file_path)
            if len(seeds) > 0:
                print("Loaded %d seeds from: %s" % (len(seeds), full_path))
            else:
                print("No seed loaded from: %s" % full_path)

            seeds_all = seeds_all + seeds
        label2seeds[label] = seeds_all

    if not isinstance(y_value, list):
        y_value = [y_value, ]

    for y_to_plot in y_value:
        for i, (label, seeds) in enumerate(label2seeds.items()):
            x = combine_data_in_seeds(seeds, x_to_use)
            y = combine_data_in_seeds(seeds, y_to_plot, smooth=smooth)
            if '_sim_' in y_to_plot:
                y = 1-y
            if y_to_plot == 'total_time':
                y = y / 3600
            ax = sns.lineplot(x=x, y=y, n_boot=20, label=label, color=colors[i], linestyle=linestyles[i], linewidth = 2)
        plt.xlabel(xlabel, fontsize=axis_font_size)
        y_label = y_to_y_label[y_to_plot] if y_to_plot in y_to_y_label else y_to_plot
        plt.ylabel(y_label, fontsize=axis_font_size)
        if ymin is not None:
            ax.set_ylim(ymin=ymin)
        if ymax is not None:
            ax.set_ylim(ymax=ymax)
        if y_log_scale or (isinstance(y_use_log, list) and y_to_plot in y_use_log):
            plt.yscale('log')
        if x_always_scientific:
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        plt.yticks(fontsize=axis_font_size)
        plt.xticks(fontsize=axis_font_size)
        plt.legend(fontsize=axis_font_size)
        plt.tight_layout()
        plt.tight_layout()
        save_folder_path_with_y = os.path.join(save_folder_path, y_to_plot)
        if save_folder_path is not None:
            if not os.path.isdir(save_folder_path_with_y):
                path = Path(save_folder_path_with_y)
                path.mkdir(parents=True)
            if save_name_suffix:
                suffix = '_' + save_name_suffix + '.png'
            else:
                suffix = '.png'
            save_path_full = os.path.join(save_folder_path_with_y, save_name_prefix + '_' + y_to_plot + suffix)

            plt.savefig(save_path_full)
            if verbose:
                print(save_path_full)
            plt.close()
        else:
            plt.show()
