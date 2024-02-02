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
DEFAULT_LINESTYLES = tuple(['solid' for _ in range(6)])
DEFAULT_COLORS = ('tab:red', 'tab:orange', 'tab:blue', 'tab:brown', 'tab:pink','tab:grey')
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

def combine_data_in_seeds(seeds, column_name, skip=0, smooth=1, offset=False, offset_amount=None, offset_direction=None, truncate=False):
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
        if offset:
            if truncate:
                if offset_direction:
                    yhat = yhat[:-offset_amount]
                else:
                    yhat = yhat[offset_amount:]
            else:
                if not offset_direction:
                    yhat += offset_amount * 5000

        vals_list.append(yhat)
    return np.concatenate(vals_list)


def combine_data_in_seeds_for_last_four(seeds, column_name, skip=0, smooth=1):
    vals_list = []
    for d in seeds:
        yhats = []
        for real_seed in d:
            if isinstance(real_seed,pd.DataFrame):
                vals_to_use = real_seed[column_name].to_numpy().reshape(-1)
            else:
                vals_to_use = real_seed[column_name].reshape(-1)
            # yhat = savgol_filter(vals_to_use, 21, 3)  # window size 51, polynomial order 3
            # TODO might want to use sth else...
            if smooth > 1 and smooth <= len(vals_to_use):
                yhat = do_smooth(vals_to_use, smooth)
            else:
                yhat = vals_to_use

            yhat = np.mean(yhat[-4:])
            yhats.append(yhat)

        vals_list.append(np.array(yhats))
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
                              y_use_log=None, x_to_use='Steps', xlabel='Number of Updates', axis_font_size=20,
                              y_log_scale=False, x_always_scientific=True, smooth=1, offset_labels = None, offset_amount = None):
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
            idx = 0
            offset = False
            if offset_labels is not None:
                for idx, offset_label in enumerate(offset_labels):
                    #print(offset_label)
                    #print(label)
                    if label == offset_label:
                        offset = True
                        break
            else:
                offset = False
            #print(offset)
            #print(label)
            #print(idx)
            if offset_amount is not None:
                x = combine_data_in_seeds(seeds, x_to_use, offset=offset, offset_amount=offset_amount[idx], offset_direction=False)
                #print(x)
                y = combine_data_in_seeds(seeds, y_to_plot, smooth=smooth, offset=offset, offset_amount=offset_amount[idx], offset_direction=True)
                #print(y)
            else:
                x = combine_data_in_seeds(seeds, x_to_use, offset=offset, offset_direction=False)
                #print(x)
                y = combine_data_in_seeds(seeds, y_to_plot, smooth=smooth, offset=offset, offset_direction=True)
                #print(y)
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
        #plt.xticks(fontsize=axis_font_size)
        plt.xticks([0, 30000, 60000, 90000, 120000, 150000, 180000])
        plt.xticks(fontsize=axis_font_size)
        plt.xticks(rotation=45)
        #plt.legend(fontsize=20)
        #if y_label == "Normalized Score":
        #    plt.legend(framealpha=0., fontsize=20, loc="lower left")
        #else:
        #    plt.legend(framealpha=0., fontsize=20, loc="upper right")
        plt.legend(framealpha=0., fontsize=17)
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



def quick_plot_with_full_name_new(labels, data_folder_full_names, colors=DEFAULT_COLORS, linestyles=DEFAULT_LINESTYLES, base_data_folder_path=DEFAULT_BASE_PATH,
                              save_name_prefix='test_save_figure', save_name_suffix=None, save_folder_path=DEFAULT_SAVE_PATH,
                              y_value=DEFAULT_Y_VALUE, verbose=True, ymin=None, ymax=None,
                              y_use_log=None, x_to_use='Steps', xlabel='Number of Updates', axis_font_size=20,
                              y_log_scale=False, x_always_scientific=True, smooth=1, offset_labels = None, offset_amount = None, take_every=None, dot=False):
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
            idx = 0
            offset = False
            if offset_labels is not None:
                for idx, offset_label in enumerate(offset_labels):
                    #print(offset_label)
                    #print(label)
                    if label == offset_label:
                        offset = True
                        break
            else:
                offset = False
            #print(offset)
            #print(label)
            #print(idx)
            if offset_amount is not None:
                x = combine_data_in_seeds(seeds, x_to_use, offset=offset, offset_amount=offset_amount[idx], offset_direction=False)
                #print(x)
                y = combine_data_in_seeds(seeds, y_to_plot, smooth=smooth, offset=offset, offset_amount=offset_amount[idx], offset_direction=True)
                #print(y)
            else:
                x = combine_data_in_seeds(seeds, x_to_use, offset=offset, offset_direction=False)
                #print(x)
                y = combine_data_in_seeds(seeds, y_to_plot, smooth=smooth, offset=offset, offset_direction=True)
                #print(y)
            #print(x)
            if take_every:
                def uneven_interleave(a, b):
                    # Resultant array
                    result = []

                    # Iterators for a and b
                    iter_a = iter(a)
                    iter_b = iter(b)

                    # Flags and counters
                    use_a = True
                    b_counter = 0

                    # Interleaving logic
                    while True:
                        if use_a:
                            try:
                                result.append(next(iter_a))
                            except StopIteration:
                                result.extend(iter_b)  # Add remaining elements from b if a is exhausted
                                break
                            use_a = False
                        else:
                            try:
                                result.append(next(iter_b))
                                b_counter += 1
                                if b_counter % (20 // take_every) == 0:
                                    use_a = True
                            except StopIteration:
                                break

                    return np.array(result)
                x = uneven_interleave(x[1::20], x[3::take_every])
                y = uneven_interleave(y[1::20], y[3::take_every])
            if '_sim_' in y_to_plot:
                y = 1-y
            if y_to_plot == 'total_time':
                y = y / 3600
            ax = sns.lineplot(x=x, y=y, ci=None, n_boot=20, label=label, color=colors[i], linestyle=linestyles[i], linewidth = 2)
            if dot:
                data = pd.DataFrame({'x': x, 'y': y})

                # Group by 'x' and calculate the mean of 'y'
                mean_values = data.groupby('x', as_index=False).mean()
                sns.scatterplot(data=mean_values, x='x', y='y', color=colors[i], ax=ax)
        
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
        
        plt.xticks([10000, 20000, 40000, 60000, 80000, 100000])
        plt.xticks(fontsize=axis_font_size)
        plt.xticks(rotation=45)
        plt.legend(framealpha=0., fontsize=17)


        plt.tight_layout()
        plt.tight_layout()
        #plt.legend(fontsize="x-large")
        #plt.legend(loc="lower right")
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



def quick_plot_with_full_name_data_ratio(labels, data_folder_full_names, colors=DEFAULT_COLORS, linestyles=DEFAULT_LINESTYLES, base_data_folder_path=DEFAULT_BASE_PATH,
                              save_name_prefix='test_save_figure', save_name_suffix=None, save_folder_path=DEFAULT_SAVE_PATH,
                              y_value=DEFAULT_Y_VALUE, verbose=True, ymin=None, ymax=None,
                              y_use_log=None, x_to_use='Steps', xlabel='Number of Updates', axis_font_size=20,
                              y_log_scale=False, x_always_scientific=True, smooth=1, offset_labels = None, offset_amount = None, take_every=None, dot=False):
    # this plots
    label2seeds = OrderedDict()
    for label, data_folder_full_name_list in zip(labels, data_folder_full_names):
        seeds_all = []
        if not isinstance(data_folder_full_name_list, list):
            data_folder_full_name_list = [data_folder_full_name_list,]
        for full_names in data_folder_full_name_list:

            seeds = []
            for full_name in full_names:
                sub_seed = []
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
                        sub_seed.append(pd.read_table(progress_file_path))
                    except:
                        print("Failed to load from progress:", progress_file_path)
                seeds.append(sub_seed)
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
            #x = combine_data_in_seeds_for_last_four(seeds, x_to_use)
            #print(x)
            y = combine_data_in_seeds_for_last_four(seeds, y_to_plot, smooth=smooth)
            #print(y)
            x = np.zeros_like(y)
            x[::6] = 0.1
            x[1::6] = 0.2
            x[2::6] = 0.4
            x[3::6] = 0.6
            x[4::6] = 0.8
            x[5::6] = 1.0
            #print(x)

            if '_sim_' in y_to_plot:
                y = 1-y
            if y_to_plot == 'total_time':
                y = y / 3600
            ax = sns.lineplot(x=x, y=y, ci=None, n_boot=20, label=label, color=colors[i], linestyle=linestyles[i], linewidth = 2)
            if dot:
                data = pd.DataFrame({'x': x, 'y': y})

                # Group by 'x' and calculate the mean of 'y'
                mean_values = data.groupby('x', as_index=False).mean()
                sns.scatterplot(data=mean_values, x='x', y='y', color=colors[i], ax=ax)
        
        plt.xlabel(xlabel, fontsize=axis_font_size)
        y_label = y_to_y_label[y_to_plot] if y_to_plot in y_to_y_label else y_to_plot
        plt.ylabel(y_label, fontsize=axis_font_size)
        if ymin is not None:
            ax.set_ylim(ymin=ymin)
        if ymax is not None:
            ax.set_ylim(ymax=ymax)
        #if y_log_scale or (isinstance(y_use_log, list) and y_to_plot in y_use_log):
        #    plt.yscale('log')
        if x_always_scientific:
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        plt.yticks(fontsize=axis_font_size)
        
        plt.xticks([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.xticks(fontsize=axis_font_size)
        plt.xticks(rotation=45)
        plt.legend(framealpha=0., fontsize=17)


        plt.tight_layout()
        plt.tight_layout()
        #plt.legend(fontsize="x-large")
        #plt.legend(loc="lower right")
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