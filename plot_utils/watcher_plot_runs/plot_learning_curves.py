import sys
sys.path.append('../..')

from quick_plot_helper import quick_plot, quick_plot_with_full_name, quick_plot_with_full_name_new, quick_plot_with_full_name_data_ratio
from log_alias import *
# standard plotting, might not be useful for us...


def get_full_names_with_envs(base_names, envs):
    # envs can be offline or online envs, offline envs will need to also have dataset name
    n = len(base_names)
    to_return = []
    for i in range(n):
        new_list = []
        for env in envs:
            full_name = base_names[i] + '_' + env
            new_list.append(full_name)
        to_return.append(new_list)
    return to_return

def get_full_names_with_envs_data_ratio(base_names, envs):
    # envs can be offline or online envs, offline envs will need to also have dataset name
    n = len(base_names)
    to_return = []
    for i in range(n):
        new_list = []
        for env in envs:
            sub_list = []
            for sub_base_name in base_names[i]:
                full_name = sub_base_name + '_' + env
                sub_list.append(full_name)
            new_list.append(sub_list)
        to_return.append(new_list)
    return to_return

MUJOCO_3_ENVS = ['hopper', 'walker2d', 'halfcheetah',  'ant']
MUJOCO_3_DATASETS = ['medium','medium-replay','medium-expert',]
d4rl_9_datasets_envs = []
for e in MUJOCO_3_ENVS:
    for d in MUJOCO_3_DATASETS:
        d4rl_9_datasets_envs.append('%s_%s' % (e, d))

d4rl_q_loss_maxs = [60, 60, 60, 120, 120, 120, 175, 175, 175]
d4rl_combined_loss_maxs = [None for _ in range(9)]
d4rl_combined_loss_mins = [None for _ in range(9)]
d4rl_combined_loss_mins[7] = 0
d4rl_combined_loss_maxs[7] = 150

online_mujoco_5 = ['Hopper-v2', 'HalfCheetah-v2', 'Walker2d-v2', 'Ant-v2', 'Humanoid-v2']

data_path = '../../code/checkpoints/'
save_path = '../../figures/'


twocolordoulbe = ['tab:blue', 'tab:orange', 'tab:blue', 'tab:orange',]
twosoliddashed = ['dashed', 'dashed',  'solid', 'solid', ]
threecolordoulbe = ['tab:blue', 'tab:orange', 'tab:red', 'tab:blue', 'tab:orange', 'tab:red']
threesoliddashed = ['dashed', 'dashed', 'dashed', 'solid', 'solid', 'solid', ]
standard_6_colors = ('tab:red', 'tab:orange', 'tab:blue', 'tab:brown', 'tab:pink','tab:grey')


standard_ys = ['TestEpRet', 'weight_diff_last_iter',
               'feature_diff_last_iter',
               'weight_sim_last_iter', 'feature_sim_last_iter']
standard_ys = ['AverageTestEpRet']

d4rl_test_performance_raw_score_col_name = 'TestEpRet'
d4rl_test_performance_col_name = 'TestEpNormRet'
d4rl_q_loss_col_name = 'sac_qf1_loss'
d4rl_x_axis_col_name = 'Steps'
d4rl_q_value_col_name = 'sac_average_qf1'
d4rl_combined_loss_col_name = 'sac_combined_loss'

d4rl_dt_loss_col_name = 'current_itr_train_loss_mean'

default_performance_smooth = 5
default_cql_q_smooth = 5
default_cql_combined_loss_smooth = 5

def plot_cql_performance_curves():
    labels = [
        'CQL',
        'CQL same task',
        'CQL MDP',
    ]
    base_names = [
        cql_jul,
        cql_jul_fd_pretrain,
        cql_jul_mdp_noproj_s100_t1,
        ]

    y = d4rl_test_performance_col_name
    ymax = None

    # aggregate
    aggregate_name = 'agg-cql'
    quick_plot_with_full_name(  # labels, folder name prefix, envs
        labels,
        get_full_names_with_envs(base_names, d4rl_9_datasets_envs),
        save_name_prefix=aggregate_name,
        base_data_folder_path=data_path,
        save_folder_path=save_path,
        y_value=[y],
        x_to_use=d4rl_x_axis_col_name,
        ymax=ymax,
        smooth=default_performance_smooth
    )

    # separate
    for env_dataset_name in d4rl_9_datasets_envs:
        quick_plot_with_full_name(  # labels, folder name prefix, envs
            labels,
            get_full_names_with_envs(base_names, [env_dataset_name]),
            save_name_prefix='ind-cql',
            base_data_folder_path=data_path,
            save_folder_path=save_path,
            y_value=[y],
            x_to_use=d4rl_x_axis_col_name,
            ymax=ymax,
            save_name_suffix=env_dataset_name,
        smooth=default_performance_smooth
        )


def plot_cql_q_loss_curves():
    labels = [
        'CQL',
        'CQL same task',
        'CQL MDP',
    ]
    base_names = [
        cql_jul,
        cql_jul_fd_pretrain,
        cql_jul_mdp_noproj_s100_t1,
        ]

    y = d4rl_q_loss_col_name
    ymax = 80
    ymin = 0
    # aggregate
    aggregate_name = 'agg-cql'
    quick_plot_with_full_name(  # labels, folder name prefix, envs
        labels,
        get_full_names_with_envs(base_names, d4rl_9_datasets_envs),
        save_name_prefix=aggregate_name,
        base_data_folder_path=data_path,
        save_folder_path=save_path,
        y_value=[y],
        x_to_use=d4rl_x_axis_col_name,
        ymax=ymax,
        ymin=ymin,
        smooth=default_cql_q_smooth,
    )

    # separate
    for env_dataset_name, ymax in zip(d4rl_9_datasets_envs, d4rl_q_loss_maxs):
        quick_plot_with_full_name(  # labels, folder name prefix, envs
            labels,
            get_full_names_with_envs(base_names, [env_dataset_name]),
            save_name_prefix='ind-cql',
            base_data_folder_path=data_path,
            save_folder_path=save_path,
            y_value=[y],
            x_to_use=d4rl_x_axis_col_name,
            ymax=ymax,
            ymin=ymin,
            save_name_suffix=env_dataset_name,
        smooth=default_cql_q_smooth,
        )

def plot_cql_combined_loss_curves():
    labels = [
        'CQL',
        'CQL same task',
        'CQL MDP',
    ]
    base_names = [
        cql_jul,
        cql_jul_fd_pretrain,
        cql_jul_mdp_noproj_s100_t1,
        ]

    y = d4rl_combined_loss_col_name
    ymax = None
    ymin = None
    # aggregate
    aggregate_name = 'agg-cql'
    quick_plot_with_full_name(  # labels, folder name prefix, envs
        labels,
        get_full_names_with_envs(base_names, d4rl_9_datasets_envs),
        save_name_prefix=aggregate_name,
        base_data_folder_path=data_path,
        save_folder_path=save_path,
        y_value=[y],
        x_to_use=d4rl_x_axis_col_name,
        ymax=ymax,
        ymin=ymin,
        smooth=default_cql_combined_loss_smooth,
    )

    # separate
    for env_dataset_name, ymax, ymin in zip(d4rl_9_datasets_envs, d4rl_combined_loss_maxs, d4rl_combined_loss_mins):
        quick_plot_with_full_name(  # labels, folder name prefix, envs
            labels,
            get_full_names_with_envs(base_names, [env_dataset_name]),
            save_name_prefix='ind-cql',
            base_data_folder_path=data_path,
            save_folder_path=save_path,
            y_value=[y],
            x_to_use=d4rl_x_axis_col_name,
            ymax=ymax,
            ymin=ymin,
            save_name_suffix=env_dataset_name,
        smooth=default_cql_combined_loss_smooth,

        )

def plot_dt_performance_curves():
    labels = [
        'DT',
        'DT Wiki',
        'DT MC',
    ]
    base_names = [
        'dt-rerun-20seeds_dt',
        'chibiT-rerun-20seeds',
        "chibiT-syn-20seeds-steps-wo_step20000",
        ]

    y = d4rl_test_performance_col_name
    ymax = None

    # aggregate
    aggregate_name = 'agg-dt'
    quick_plot_with_full_name(  # labels, folder name prefix, envs
        labels,
        get_full_names_with_envs(base_names, d4rl_9_datasets_envs),
        save_name_prefix=aggregate_name,
        base_data_folder_path=data_path,
        save_folder_path=save_path,
        y_value=[y],
        x_to_use=d4rl_x_axis_col_name,
        ymax=ymax,
        smooth=default_performance_smooth,
        
    )

    # separate
    for env_dataset_name in d4rl_9_datasets_envs:
        quick_plot_with_full_name(  # labels, folder name prefix, envs
            labels,
            get_full_names_with_envs(base_names, [env_dataset_name]),
            save_name_prefix='ind-dt',
            base_data_folder_path=data_path,
            save_folder_path=save_path,
            y_value=[y],
            x_to_use=d4rl_x_axis_col_name,
            ymax=ymax,
            save_name_suffix=env_dataset_name,
        smooth=default_performance_smooth
        )
        
def plot_dt_performance_curves_offset():
    labels = [
        'DT',
        'DT+Wiki',
        'DT+Synthetic',
    ]
    base_names = [
        'dt-long-20seeds_dt_36',
        'chibiT-rerun-20seeds',
        "chibiT-syn-20seeds-steps-wo_step20000",
        ]

    y = d4rl_test_performance_col_name
    ymax = None

    # aggregate
    aggregate_name = 'agg-dt-offset-new'
    quick_plot_with_full_name(  # labels, folder name prefix, envs
        labels,
        get_full_names_with_envs(base_names, d4rl_9_datasets_envs),
        save_name_prefix=aggregate_name,
        base_data_folder_path=data_path,
        save_folder_path=save_path,
        y_value=[y],
        x_to_use=d4rl_x_axis_col_name,
        ymax=ymax,
        smooth=default_performance_smooth,
        offset_labels = ['DT+Synthetic', 'DT+Wiki'],
        offset_amount = [4, 16]
        
    )

    # separate
    for env_dataset_name in d4rl_9_datasets_envs:
        quick_plot_with_full_name(  # labels, folder name prefix, envs
            labels,
            get_full_names_with_envs(base_names, [env_dataset_name]),
            save_name_prefix='ind-dt-offset-new',
            base_data_folder_path=data_path,
            save_folder_path=save_path,
            y_value=[y],
            x_to_use=d4rl_x_axis_col_name,
            ymax=ymax,
            save_name_suffix=env_dataset_name,
            smooth=default_performance_smooth,
            offset_labels = ['DT+Synthetic', 'DT+Wiki'],
            offset_amount = [4, 16]
        )        


def plot_dt_loss_curves():
    labels = [
        'DT',
        'DT Wiki',
        'DT MC',
    ]
    base_names = [
        'dt-rerun-20seeds_dt',
        'chibiT-rerun-20seeds',
        'chibiT-syn-20seeds-steps-wo_step20000',
        ]

    y = d4rl_dt_loss_col_name
    ymax = None

    # aggregate
    aggregate_name = 'agg-dt'
    quick_plot_with_full_name(  # labels, folder name prefix, envs
        labels,
        get_full_names_with_envs(base_names, d4rl_9_datasets_envs),
        save_name_prefix=aggregate_name,
        base_data_folder_path=data_path,
        save_folder_path=save_path,
        y_value=[y],
        x_to_use=d4rl_x_axis_col_name,
        ymax=ymax,
        y_log_scale=True,
    )

    # separate
    for env_dataset_name in d4rl_9_datasets_envs:
        quick_plot_with_full_name(  # labels, folder name prefix, envs
            labels,
            get_full_names_with_envs(base_names, [env_dataset_name]),
            save_name_prefix='ind-dt',
            base_data_folder_path=data_path,
            save_folder_path=save_path,
            y_value=[y],
            x_to_use=d4rl_x_axis_col_name,
            ymax=ymax,
            save_name_suffix=env_dataset_name,
            y_log_scale=True,
        )


def plot_dt_loss_curves_offset():
    labels = [
        'DT',
        'DT+Wiki',
        'DT+Synthetic',
    ]
    base_names = [
        'dt-long-20seeds_dt_36',
        'chibiT-rerun-20seeds',
        'chibiT-syn-20seeds-steps-wo_step20000',
        ]

    y = d4rl_dt_loss_col_name
    ymax = None

    # aggregate
    aggregate_name = 'agg-dt-offset-new'
    quick_plot_with_full_name(  # labels, folder name prefix, envs
        labels,
        get_full_names_with_envs(base_names, d4rl_9_datasets_envs),
        save_name_prefix=aggregate_name,
        base_data_folder_path=data_path,
        save_folder_path=save_path,
        y_value=[y],
        x_to_use=d4rl_x_axis_col_name,
        ymax=ymax,
        y_log_scale=True,
        offset_labels = ['DT+Synthetic', 'DT+Wiki'],
        offset_amount = [4, 16]
    )

    # separate
    for env_dataset_name in d4rl_9_datasets_envs:
        quick_plot_with_full_name(  # labels, folder name prefix, envs
            labels,
            get_full_names_with_envs(base_names, [env_dataset_name]),
            save_name_prefix='ind-dt-offset-new',
            base_data_folder_path=data_path,
            save_folder_path=save_path,
            y_value=[y],
            x_to_use=d4rl_x_axis_col_name,
            ymax=ymax,
            save_name_suffix=env_dataset_name,
            y_log_scale=True,
            offset_labels = ['DT+Synthetic', 'DT+Wiki'],
            offset_amount = [4, 16]
        )










# def plot_july_new1():
#     labels = [
#         'CQL',
#         'CQL same task',
#     ]
#     base_names = [
#         cql_jul,
#         cql_jul_fd_pretrain,
#         ]
#
#     y = 'sac_combined_loss'
#     ymax = 100
#
#     save_path = '../../figures/july_test'
#
#     # aggregate
#     aggregate_name = 'agg-julnew'
#     quick_plot_with_full_name(  # labels, folder name prefix, envs
#         labels,
#         get_full_names_with_envs(base_names, d4rl_9_datasets_envs),
#         save_name_prefix=aggregate_name,
#         base_data_folder_path=data_path,
#         save_folder_path=save_path,
#         y_value=[y],
#         x_to_use=d4rl_x_axis_col_name,
#         ymax=ymax,
#         smooth=default_performance_smooth
#     )
#
#     # separate
#     ymax = None
#     for env_dataset_name in d4rl_9_datasets_envs:
#         quick_plot_with_full_name(  # labels, folder name prefix, envs
#             labels,
#             get_full_names_with_envs(base_names, [env_dataset_name]),
#             save_name_prefix='ind-julnew',
#             base_data_folder_path=data_path,
#             save_folder_path=save_path,
#             y_value=[y],
#             x_to_use=d4rl_x_axis_col_name,
#             ymax=ymax,
#             save_name_suffix=env_dataset_name,
#         smooth=default_performance_smooth
#         )
#
# def plot_july_new2():
#     labels = [
#         'CQL new 10seed',
#         'CQL same task old',
#         'CQL same new 10seed',
#     ]
#     base_names = [
#         cql_jul,
#         cql_fd_pretrain,
#         cql_jul_fd_pretrain,
#         ]
#
#     y = d4rl_test_performance_col_name
#
#     save_path = '../../figures/july_test'
#
#     # aggregate
#     ymax = None
#     aggregate_name = 'agg-julnew'
#     quick_plot_with_full_name(  # labels, folder name prefix, envs
#         labels,
#         get_full_names_with_envs(base_names, d4rl_9_datasets_envs),
#         save_name_prefix=aggregate_name,
#         base_data_folder_path=data_path,
#         save_folder_path=save_path,
#         y_value=[y],
#         x_to_use=d4rl_x_axis_col_name,
#         ymax=ymax,
#         smooth=default_performance_smooth
#     )
#
#     # separate
#     ymax = None
#     for env_dataset_name in d4rl_9_datasets_envs:
#         quick_plot_with_full_name(  # labels, folder name prefix, envs
#             labels,
#             get_full_names_with_envs(base_names, [env_dataset_name]),
#             save_name_prefix='ind-julnew',
#             base_data_folder_path=data_path,
#             save_folder_path=save_path,
#             y_value=[y],
#             x_to_use=d4rl_x_axis_col_name,
#             ymax=ymax,
#             save_name_suffix=env_dataset_name,
#         smooth=default_performance_smooth
#         )

# plot_july_new1()
# plot_july_new2()


def plot_dt_performance_curves_offset_new():
    labels = [
        'DT',
        'DT+Wiki',
        'DT+Synthetic',
    ]
    base_names = [
        'dt-long-20seeds_dt_36',
        'chibiT-rerun-20seeds',
        "chibiT-syn-long-20seeds-steps-wo_S100_32",
        ]

    y = d4rl_test_performance_col_name
    ymax = None

    # aggregate
    aggregate_name = 'agg-dt-offset-new-new'
    quick_plot_with_full_name(  # labels, folder name prefix, envs
        labels,
        get_full_names_with_envs(base_names, d4rl_9_datasets_envs),
        save_name_prefix=aggregate_name,
        base_data_folder_path=data_path,
        save_folder_path=save_path,
        y_value=[y],
        x_to_use=d4rl_x_axis_col_name,
        ymax=ymax,
        smooth=default_performance_smooth,
        xlabel="Total Number of Updates",
        offset_labels = ['DT+Synthetic', 'DT+Wiki'],
        offset_amount = [4, 16]
        
    )

    # separate
    for env_dataset_name in d4rl_9_datasets_envs:
        quick_plot_with_full_name(  # labels, folder name prefix, envs
            labels,
            get_full_names_with_envs(base_names, [env_dataset_name]),
            save_name_prefix='ind-dt-offset-new-new',
            base_data_folder_path=data_path,
            save_folder_path=save_path,
            y_value=[y],
            x_to_use=d4rl_x_axis_col_name,
            ymax=ymax,
            save_name_suffix=env_dataset_name,
            smooth=default_performance_smooth,
            xlabel="Total Number of Updates",

            offset_labels = ['DT+Synthetic', 'DT+Wiki'],
            offset_amount = [4, 16]
        )      

def plot_dt_loss_curves_offset_new():
    labels = [
        'DT',
        'DT+Wiki',
        'DT+Synthetic',
    ]
    base_names = [
        'dt-long-20seeds_dt_36',
        'chibiT-rerun-20seeds',
        "chibiT-syn-long-20seeds-steps-wo_S100_32",
        ]

    y = d4rl_dt_loss_col_name
    ymax = None

    # aggregate
    aggregate_name = 'agg-dt-offset-new-new'
    quick_plot_with_full_name(  # labels, folder name prefix, envs
        labels,
        get_full_names_with_envs(base_names, d4rl_9_datasets_envs),
        save_name_prefix=aggregate_name,
        base_data_folder_path=data_path,
        save_folder_path=save_path,
        y_value=[y],
        x_to_use=d4rl_x_axis_col_name,
        ymax=ymax,
        y_log_scale=True,
        xlabel="Total Number of Updates",

        offset_labels = ['DT+Synthetic', 'DT+Wiki'],
        offset_amount = [4, 16]
    )

    # separate
    for env_dataset_name in d4rl_9_datasets_envs:
        quick_plot_with_full_name(  # labels, folder name prefix, envs
            labels,
            get_full_names_with_envs(base_names, [env_dataset_name]),
            save_name_prefix='ind-dt-offset-new-new',
            base_data_folder_path=data_path,
            save_folder_path=save_path,
            y_value=[y],
            x_to_use=d4rl_x_axis_col_name,
            ymax=ymax,
            save_name_suffix=env_dataset_name,
            y_log_scale=True,
            xlabel="Total Number of Updates",

            offset_labels = ['DT+Synthetic', 'DT+Wiki'],
            offset_amount = [4, 16]
        )




def plot_dt_performance_curves_states_abl():
    labels = [
        'DT',
        'DT+Wiki',
        'S10',
        'S100',
        'S1000',
        'S10000',

    ]
    base_names = [
        'dt-rerun-20seeds_dt',
        'chibiT-rerun-20seeds',
        'chibiT-syn-20seeds-states-wo_S10',
        'chibiT-syn-20seeds-steps-wo_step20000',
        'chibiT-syn-20seeds-states-wo_S1000',
        'chibiT-syn-20seeds-states-wo_S10000',
        ]

    y = d4rl_test_performance_col_name
    ymax = None

    # aggregate
    aggregate_name = 'agg-dt-ftsteps-states'
    quick_plot_with_full_name_new(  # labels, folder name prefix, envs
        labels,
        get_full_names_with_envs(base_names, d4rl_9_datasets_envs),
        save_name_prefix=aggregate_name,
        base_data_folder_path=data_path,
        save_folder_path=save_path,
        y_value=[y],
        x_to_use=d4rl_x_axis_col_name,
        ymax=ymax,
        smooth=default_performance_smooth,
        xlabel="Number of Finetune Updates",
        take_every=4,
        dot=True

        
    )

    # separate
    for env_dataset_name in d4rl_9_datasets_envs:
        quick_plot_with_full_name_new(  # labels, folder name prefix, envs
            labels,
            get_full_names_with_envs(base_names, [env_dataset_name]),
            save_name_prefix='ind-dt-ftsteps-states',
            base_data_folder_path=data_path,
            save_folder_path=save_path,
            y_value=[y],
            x_to_use=d4rl_x_axis_col_name,
            ymax=ymax,
            save_name_suffix=env_dataset_name,
            xlabel="Number of Finetune Updates",

            smooth=default_performance_smooth,
            take_every=4,
            dot=True

        )      

def plot_dt_performance_curves_temp_abl():
    labels = [
        'DT',
        'DT+Wiki',
        'T0.1',
        'T1.0',
        'T10.0',
        'IID',

    ]
    base_names = [
        'dt-rerun-20seeds_dt',
        'chibiT-rerun-20seeds',
        'chibiT-syn-20seeds-temps-wo_T0.1',
        'chibiT-syn-20seeds-steps-wo_step20000',
        'chibiT-syn-20seeds-temps-wo_T10.0',
        'chibiT-iid_wo_step20000'
        ]

    y = d4rl_test_performance_col_name
    ymax = None

    # aggregate
    aggregate_name = 'agg-dt-ftsteps-temp'
    quick_plot_with_full_name_new(  # labels, folder name prefix, envs
        labels,
        get_full_names_with_envs(base_names, d4rl_9_datasets_envs),
        save_name_prefix=aggregate_name,
        base_data_folder_path=data_path,
        save_folder_path=save_path,
        y_value=[y],
        x_to_use=d4rl_x_axis_col_name,
        ymax=ymax,
        smooth=default_performance_smooth,
        take_every=4,
        xlabel="Number of Finetune Updates",
        dot=True

        
    )

    # separate
    for env_dataset_name in d4rl_9_datasets_envs:
        quick_plot_with_full_name_new(  # labels, folder name prefix, envs
            labels,
            get_full_names_with_envs(base_names, [env_dataset_name]),
            save_name_prefix='ind-dt-ftsteps-temp',
            base_data_folder_path=data_path,
            save_folder_path=save_path,
            y_value=[y],
            x_to_use=d4rl_x_axis_col_name,
            ymax=ymax,
            save_name_suffix=env_dataset_name,
            smooth=default_performance_smooth,
            take_every=4,
            xlabel="Number of Finetune Updates",

            dot=True

        )   


def plot_dt_performance_curves_step_abl():
    labels = [
        'DT',
        'DT+Wiki',
        '1-MC',
        '2-MC',
        '5-MC',

    ]
    base_names = [
        'dt-rerun-20seeds_dt',
        'chibiT-rerun-20seeds',
        'chibiT-syn-20seeds-steps-wo_step20000',
        'chibiT-syn-20seeds-steps-wo_MC-2',
        'chibiT-syn-20seeds-steps-wo_MC-5',
        ]

    y = d4rl_test_performance_col_name
    ymax = None

    # aggregate
    aggregate_name = 'agg-dt-ftsteps-step'
    quick_plot_with_full_name_new(  # labels, folder name prefix, envs
        labels,
        get_full_names_with_envs(base_names, d4rl_9_datasets_envs),
        save_name_prefix=aggregate_name,
        base_data_folder_path=data_path,
        save_folder_path=save_path,
        y_value=[y],
        x_to_use=d4rl_x_axis_col_name,
        ymax=ymax,
        smooth=default_performance_smooth,
        take_every=4,
        xlabel="Number of Finetune Updates",

        dot=True

        
    )

    # separate
    for env_dataset_name in d4rl_9_datasets_envs:
        quick_plot_with_full_name_new(  # labels, folder name prefix, envs
            labels,
            get_full_names_with_envs(base_names, [env_dataset_name]),
            save_name_prefix='ind-dt-ftsteps-step',
            base_data_folder_path=data_path,
            save_folder_path=save_path,
            y_value=[y],
            x_to_use=d4rl_x_axis_col_name,
            ymax=ymax,
            save_name_suffix=env_dataset_name,
            smooth=default_performance_smooth,
            take_every=4,
            xlabel="Number of Finetune Updates",

            dot=True

        ) 



def plot_dt_performance_curves_states_abl_data_ratio():
    labels = [
        'DT',
        'DT+Wiki',
        'S10',
        'S100',
        'S1000',
        'S10000',

    ]
    base_names = [
        ('dt-rerun-5seeds_data_size_dt_0.1',
            'dt-rerun-5seeds_data_size_dt_0.2',
            'dt-rerun-5seeds_data_size_dt_0.4',
            'dt-rerun-5seeds_data_size_dt_0.6',
            'dt-rerun-5seeds_data_size_dt_0.8',
            'dt-rerun-20seeds_dt',),
        ('chibiT-rerun-5seeds-ftratio_0.1',
            'chibiT-rerun-5seeds-ftratio_0.2',
            'chibiT-rerun-5seeds-ftratio_0.4',
            'chibiT-rerun-5seeds-ftratio_0.6',
            'chibiT-rerun-5seeds-ftratio_0.8',
            'chibiT-rerun-20seeds',),
        ('chibiT-syn-5seeds-states-ftratio-wo_S10_0.1',
            'chibiT-syn-5seeds-states-ftratio-wo_S10_0.2',
            'chibiT-syn-5seeds-states-ftratio-wo_S10_0.4',
            'chibiT-syn-5seeds-states-ftratio-wo_S10_0.6',
            'chibiT-syn-5seeds-states-ftratio-wo_S10_0.8',
            'chibiT-syn-20seeds-states-wo_S10',),
        ('chibiT-syn-5seeds-states-ftratio-wo_S100_0.1'
            'chibiT-syn-5seeds-states-ftratio-wo_S100_0.2',
            'chibiT-syn-5seeds-states-ftratio-wo_S100_0.4',
            'chibiT-syn-5seeds-states-ftratio-wo_S100_0.6',
            'chibiT-syn-5seeds-states-ftratio-wo_S100_0.8',
            'chibiT-syn-20seeds-steps-wo_step20000',),
        ('chibiT-syn-5seeds-states-ftratio-wo_S1000_0.1',
            'chibiT-syn-5seeds-states-ftratio-wo_S1000_0.2',
            'chibiT-syn-5seeds-states-ftratio-wo_S1000_0.4',
            'chibiT-syn-5seeds-states-ftratio-wo_S1000_0.6',
            'chibiT-syn-5seeds-states-ftratio-wo_S1000_0.8',
            'chibiT-syn-20seeds-states-wo_S1000',),
        ('chibiT-syn-5seeds-states-ftratio-wo_S10000_0.1',
            'chibiT-syn-5seeds-states-ftratio-wo_S10000_0.2',
            'chibiT-syn-5seeds-states-ftratio-wo_S10000_0.4',
            'chibiT-syn-5seeds-states-ftratio-wo_S10000_0.6',
            'chibiT-syn-5seeds-states-ftratio-wo_S10000_0.8',
            'chibiT-syn-20seeds-states-wo_S10000',),
        ]

    y = d4rl_test_performance_col_name
    ymax = None

    # aggregate
    aggregate_name = 'agg-dt-ftratio-states'
    quick_plot_with_full_name_data_ratio(  # labels, folder name prefix, envs
        labels,
        get_full_names_with_envs_data_ratio(base_names, d4rl_9_datasets_envs),
        save_name_prefix=aggregate_name,
        base_data_folder_path=data_path,
        save_folder_path=save_path,
        y_value=[y],
        x_to_use=d4rl_x_axis_col_name,
        ymax=ymax,
        smooth=default_performance_smooth,
        xlabel="Finetune Data Ratio",
        take_every=4,
        dot=True

        
    )

    # separate
    for env_dataset_name in d4rl_9_datasets_envs:
        quick_plot_with_full_name_data_ratio(  # labels, folder name prefix, envs
            labels,
            get_full_names_with_envs_data_ratio(base_names, [env_dataset_name]),
            save_name_prefix='ind-dt-ftratio-states',
            base_data_folder_path=data_path,
            save_folder_path=save_path,
            y_value=[y],
            x_to_use=d4rl_x_axis_col_name,
            ymax=ymax,
            save_name_suffix=env_dataset_name,
            xlabel="Finetune Data Ratio",

            smooth=default_performance_smooth,
            take_every=4,
            dot=True

        )      


def plot_dt_performance_curves_step_abl_data_ratio():
    labels = [
        'DT',
        'DT+Wiki',
        '1-MC',
        '2-MC',
        '5-MC',

    ]
    base_names = [
        ('dt-rerun-5seeds_data_size_dt_0.1',
            'dt-rerun-5seeds_data_size_dt_0.2',
            'dt-rerun-5seeds_data_size_dt_0.4',
            'dt-rerun-5seeds_data_size_dt_0.6',
            'dt-rerun-5seeds_data_size_dt_0.8',
            'dt-rerun-20seeds_dt',),
        ('chibiT-rerun-5seeds-ftratio_0.1',
            'chibiT-rerun-5seeds-ftratio_0.2',
            'chibiT-rerun-5seeds-ftratio_0.4',
            'chibiT-rerun-5seeds-ftratio_0.6',
            'chibiT-rerun-5seeds-ftratio_0.8',
            'chibiT-rerun-20seeds',),
        ('chibiT-syn-5seeds-states-ftratio-wo_MC-1_0.1',
            'chibiT-syn-5seeds-states-ftratio-wo_MC-1_0.2',
            'chibiT-syn-5seeds-states-ftratio-wo_MC-1_0.4',
            'chibiT-syn-5seeds-states-ftratio-wo_MC-1_0.6',
            'chibiT-syn-5seeds-states-ftratio-wo_MC-1_0.8',
            'chibiT-syn-20seeds-steps-wo_step20000',),
        ('chibiT-syn-5seeds-states-ftratio-wo_MC-2_0.1'
            'chibiT-syn-5seeds-states-ftratio-wo_MC-2_0.2',
            'chibiT-syn-5seeds-states-ftratio-wo_MC-2_0.4',
            'chibiT-syn-5seeds-states-ftratio-wo_MC-2_0.6',
            'chibiT-syn-5seeds-states-ftratio-wo_MC-2_0.8',
            'chibiT-syn-20seeds-steps-wo_MC-2',),
        ('chibiT-syn-5seeds-states-ftratio-wo_MC-5_0.1',
            'chibiT-syn-5seeds-states-ftratio-wo_MC-5_0.2',
            'chibiT-syn-5seeds-states-ftratio-wo_MC-5_0.4',
            'chibiT-syn-5seeds-states-ftratio-wo_MC-5_0.6',
            'chibiT-syn-5seeds-states-ftratio-wo_MC-5_0.8',
            'chibiT-syn-20seeds-steps-wo_MC-5',),
        ]

    y = d4rl_test_performance_col_name
    ymax = None

    # aggregate
    aggregate_name = 'agg-dt-ftratio-step'
    quick_plot_with_full_name_data_ratio(  # labels, folder name prefix, envs
        labels,
        get_full_names_with_envs_data_ratio(base_names, d4rl_9_datasets_envs),
        save_name_prefix=aggregate_name,
        base_data_folder_path=data_path,
        save_folder_path=save_path,
        y_value=[y],
        x_to_use=d4rl_x_axis_col_name,
        ymax=ymax,
        smooth=default_performance_smooth,
        xlabel="Finetune Data Ratio",
        take_every=4,
        dot=True

        
    )

    # separate
    for env_dataset_name in d4rl_9_datasets_envs:
        quick_plot_with_full_name_data_ratio(  # labels, folder name prefix, envs
            labels,
            get_full_names_with_envs_data_ratio(base_names, [env_dataset_name]),
            save_name_prefix='ind-dt-ftratio-step',
            base_data_folder_path=data_path,
            save_folder_path=save_path,
            y_value=[y],
            x_to_use=d4rl_x_axis_col_name,
            ymax=ymax,
            save_name_suffix=env_dataset_name,
            xlabel="Finetune Data Ratio",

            smooth=default_performance_smooth,
            take_every=4,
            dot=True

        )      



def plot_dt_performance_curves_temp_abl_data_ratio():
    labels = [
        'DT',
        'DT+Wiki',
        'T0.1',
        'T1.0',
        'T10.0',
        'IID'

    ]
    base_names = [
        ('dt-rerun-5seeds_data_size_dt_0.1',
            'dt-rerun-5seeds_data_size_dt_0.2',
            'dt-rerun-5seeds_data_size_dt_0.4',
            'dt-rerun-5seeds_data_size_dt_0.6',
            'dt-rerun-5seeds_data_size_dt_0.8',
            'dt-rerun-20seeds_dt',),
        ('chibiT-rerun-5seeds-ftratio_0.1',
            'chibiT-rerun-5seeds-ftratio_0.2',
            'chibiT-rerun-5seeds-ftratio_0.4',
            'chibiT-rerun-5seeds-ftratio_0.6',
            'chibiT-rerun-5seeds-ftratio_0.8',
            'chibiT-rerun-20seeds',),
        ('chibiT-syn-20seeds-temps-ftratio-wo_T0.1_0.1',
            'chibiT-syn-20seeds-temps-ftratio-wo_T0.1_0.2',
            'chibiT-syn-20seeds-temps-ftratio-wo_T0.1_0.4',
            'chibiT-syn-20seeds-temps-ftratio-wo_T0.1_0.6',
            'chibiT-syn-20seeds-temps-ftratio-wo_T0.1_0.8',
            'chibiT-syn-20seeds-temps-wo_T0.1',),
        ('chibiT-syn-20seeds-temps-ftratio-wo_T1.0_0.1'
            'chibiT-syn-20seeds-temps-ftratio-wo_T1.0_0.2',
            'chibiT-syn-20seeds-temps-ftratio-wo_T1.0_0.4',
            'chibiT-syn-20seeds-temps-ftratio-wo_T1.0_0.6',
            'chibiT-syn-20seeds-temps-ftratio-wo_T1.0_0.8',
            'chibiT-syn-20seeds-steps-wo_step20000',),
        ('chibiT-syn-20seeds-temps-ftratio-wo_T10.0_0.1',
            'chibiT-syn-20seeds-temps-ftratio-wo_T10.0_0.2',
            'chibiT-syn-20seeds-temps-ftratio-wo_T10.0_0.4',
            'chibiT-syn-20seeds-temps-ftratio-wo_T10.0_0.6',
            'chibiT-syn-20seeds-temps-ftratio-wo_T10.0_0.8',
            'chibiT-syn-20seeds-temps-wo_T10.0',),
        ('chibiT-syn-20seeds-temps-ftratio-wo_TIID_0.1',
            'chibiT-syn-20seeds-temps-ftratio-wo_TIID_0.2',
            'chibiT-syn-20seeds-temps-ftratio-wo_TIID_0.4',
            'chibiT-syn-20seeds-temps-ftratio-wo_TIID_0.6',
            'chibiT-syn-20seeds-temps-ftratio-wo_TIID_0.8',
            'chibiT-iid_wo_step20000',),
        ]

    y = d4rl_test_performance_col_name
    ymax = None

    # aggregate
    aggregate_name = 'agg-dt-ftratio-temp'
    quick_plot_with_full_name_data_ratio(  # labels, folder name prefix, envs
        labels,
        get_full_names_with_envs_data_ratio(base_names, d4rl_9_datasets_envs),
        save_name_prefix=aggregate_name,
        base_data_folder_path=data_path,
        save_folder_path=save_path,
        y_value=[y],
        x_to_use=d4rl_x_axis_col_name,
        ymax=ymax,
        smooth=default_performance_smooth,
        xlabel="Finetune Data Ratio",
        take_every=4,
        dot=True

        
    )

    # separate
    for env_dataset_name in d4rl_9_datasets_envs:
        quick_plot_with_full_name_data_ratio(  # labels, folder name prefix, envs
            labels,
            get_full_names_with_envs_data_ratio(base_names, [env_dataset_name]),
            save_name_prefix='ind-dt-ftratio-temp',
            base_data_folder_path=data_path,
            save_folder_path=save_path,
            y_value=[y],
            x_to_use=d4rl_x_axis_col_name,
            ymax=ymax,
            save_name_suffix=env_dataset_name,
            xlabel="Finetune Data Ratio",

            smooth=default_performance_smooth,
            take_every=4,
            dot=True

        )  


# plot_cql_performance_curves()
# plot_cql_q_loss_curves()
# plot_cql_combined_loss_curves()
#plot_dt_performance_curves()
#plot_dt_performance_curves_offset()
#plot_dt_loss_curves()
#plot_dt_loss_curves_offset()
#plot_dt_loss_curves_offset_new()
#plot_dt_performance_curves_offset_new()
#plot_dt_performance_curves_states_abl()
#plot_dt_performance_curves_temp_abl()
#plot_dt_performance_curves_step_abl()
plot_dt_performance_curves_states_abl_data_ratio()
plot_dt_performance_curves_step_abl_data_ratio()
plot_dt_performance_curves_temp_abl_data_ratio()