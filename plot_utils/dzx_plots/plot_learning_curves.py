from plot_utils.quick_plot_helper import quick_plot, quick_plot_with_full_name
from plot_utils.log_alias import *


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


MUJOCO_3_ENVS = ['hopper', 'walker2d', 'halfcheetah', ]
MUJOCO_3_DATASETS = ['medium', 'medium-replay', 'medium-expert', ]
d4rl_9_datasets_envs = []
for e in MUJOCO_3_ENVS:
    for d in MUJOCO_3_DATASETS:
        d4rl_9_datasets_envs.append('%s_%s' % (e, d))

MUJOCO_4_ENVS = ['hopper', 'walker2d', 'halfcheetah', 'ant']
MUJOCO_3_DATASETS = ['medium', 'medium-replay', 'medium-expert', ]
d4rl_12_datasets_envs = []
for e in MUJOCO_4_ENVS:
    for d in MUJOCO_3_DATASETS:
        d4rl_12_datasets_envs.append('%s_%s' % (e, d))

d4rl_q_loss_maxs = [60, 60, 60, 120, 120, 120, 175, 175, 175, 200, 200, 200]
d4rl_combined_loss_maxs = [None for _ in range(12)]
d4rl_combined_loss_mins = [None for _ in range(12)]
d4rl_combined_loss_mins[7] = 0
d4rl_combined_loss_maxs[7] = 150

online_mujoco_5 = ['Hopper-v2', 'HalfCheetah-v2', 'Walker2d-v2', 'Ant-v2', 'Humanoid-v2']

twocolordoulbe = ['tab:blue', 'tab:orange', 'tab:blue', 'tab:orange', ]
twosoliddashed = ['dashed', 'dashed', 'solid', 'solid', ]
threecolordoulbe = ['tab:blue', 'tab:orange', 'tab:red', 'tab:blue', 'tab:orange', 'tab:red']
threesoliddashed = ['dashed', 'dashed', 'dashed', 'solid', 'solid', 'solid', ]
standard_6_colors = ('tab:red', 'tab:orange', 'tab:blue', 'tab:brown', 'tab:pink', 'tab:grey')

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


def plot_cql_performance_curves(labels, base_names):
    # labels = [
    #     'CQL',
    #     'CQL same data',
    #     'CQL MDP',
    # ]
    # base_names = [
    #     cql_jul,
    #     cql_jul_fd_pretrain,
    #     cql_jul_mdp_noproj_s100_t1,
    #     ]

    y = d4rl_test_performance_col_name
    ymax = None

    # aggregate
    aggregate_name = 'agg-cql'
    quick_plot_with_full_name(  # labels, folder name prefix, envs
        labels,
        get_full_names_with_envs(base_names, d4rl_12_datasets_envs),
        save_name_prefix=aggregate_name,
        base_data_folder_path=data_path,
        save_folder_path=save_path,
        y_value=[y],
        x_to_use=d4rl_x_axis_col_name,
        ymax=ymax,
        smooth=default_performance_smooth
    )

    # separate
    for env_dataset_name in d4rl_12_datasets_envs:
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


def plot_cql_q_loss_curves(labels, base_names):
    # labels = [
    #     'CQL',
    #     'CQL same data',
    #     'CQL MDP',
    # ]
    # base_names = [
    #     cql_jul,
    #     cql_jul_fd_pretrain,
    #     cql_jul_mdp_noproj_s100_t1,
    #     ]

    y = d4rl_q_loss_col_name
    ymax = 80
    ymin = 0
    # aggregate
    aggregate_name = 'agg-cql'
    quick_plot_with_full_name(  # labels, folder name prefix, envs
        labels,
        get_full_names_with_envs(base_names, d4rl_12_datasets_envs),
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
    for env_dataset_name, ymax in zip(d4rl_12_datasets_envs, d4rl_q_loss_maxs):
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


def plot_cql_combined_loss_curves(labels, base_names):
    # labels = [
    #     'CQL',
    #     'CQL same task',
    #     'CQL MDP',
    # ]
    # base_names = [
    #     cql_jul,
    #     cql_jul_fd_pretrain,
    #     cql_jul_mdp_noproj_s100_t1,
    #     ]

    y = d4rl_combined_loss_col_name
    ymax = None
    ymin = None
    # aggregate
    aggregate_name = 'agg-cql'
    quick_plot_with_full_name(  # labels, folder name prefix, envs
        labels,
        get_full_names_with_envs(base_names, d4rl_12_datasets_envs),
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
    for env_dataset_name, ymax, ymin in zip(d4rl_12_datasets_envs, d4rl_combined_loss_maxs, d4rl_combined_loss_mins):
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
        'DT same data'
    ]
    base_names = [
        dt,
        chibiT,
        dt_mc_1step_vocab100,
        dt_same_data,
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
        smooth=default_performance_smooth
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


def plot_dt_loss_curves():
    labels = [
        'DT',
        'DT Wiki',
        'DT MC',
        'DT same data'
    ]
    base_names = [
        dt,
        chibiT,
        dt_mc_1step_vocab100,
        dt_same_data,
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


# CQL Main Results
# labels = [
#     'CQL_Baseline',
#     'CQL_Same',
#     'CQL_MDP',
# ]
#
# base_names = [
#     cql_2x,
#     cql_same,
#     cql_mdp_t1
# ]

# CQL MDP Temp Ablation
# labels = [
#     'CQL_Baseline',
#     'CQL_Same',
#     'CQL_MDP0.001',
#     'CQL_MDP1',
#     'CQL_MDP1000',
#     'CQL_MDP_IID'
# ]
#
# base_names = [
#     cql_1x,
#     cql_same,
#     cql_mdp_t0001,
#     cql_mdp_t1,
#     cql_mdp_t1000,
#     cql_mdp_tinf2
# ]

# CQL Fix-prediction
# labels = [
#     'CQL_MDP_IID',
#     'CQL_FixTarg',
#     'CQL_MeanTarg'
# ]
# base_names = [
#     cql_mdp_tinf2,
#     cql_mdp_tfix,
#     cql_mdp_tmean
# ]

# CQL Cluster-Prediction
# labels = [
#     'CQL_IID',
#     'CQL_FixTarg',
#     'CQL_001N',
#     'CQL_01N',
#     'CQL_1N',
#     'CQL_2N'
# ]
#
# base_names = [
#     cql_mdp_tinf2,
#     cql_mdp_tfix,
#     cql_mdp_sigma001N,
#     cql_mdp_sigma01N,
#     cql_mdp_sigma1N,
#     cql_mdp_sigma2N,
# ]

# labels = [
#     'CQL_IID',
#     'CQL_FixTarg',
#     'CQL_001S',
#     'CQL_01S',
#     'CQL_1S',
#     'CQL_2S'
# ]
#
# base_names = [
#     cql_mdp_tinf2,
#     cql_mdp_tfix,
#     cql_mdp_sigma001S,
#     cql_mdp_sigma01S,
#     cql_mdp_sigma1S,
#     cql_mdp_sigma2S,
# ]

# labels = [
#     'CQL_FixTarg',
#     'CQL_001S',
#     'CQL_01S',
#     'CQL_1S',
#     'CQL_001N',
#     'CQL_01N',
#     'CQL_1N',
#
# ]
#
# base_names = [
#     cql_mdp_tfix,
#     cql_mdp_sigma001S,
#     cql_mdp_sigma01S,
#     cql_mdp_sigma1S,
#     cql_mdp_sigma001N,
#     cql_mdp_sigma01N,
#     cql_mdp_sigma1N,
#
# ]

# CQL MDP 2x:
# labels = [
#     'CQL_2x',
#     'CQL_MDP_2x'
# ]
#
# base_names = [
#     cql_2x,
#     cql_mdp_2x
# ]

# # CQL Slow Finetune:
# labels = [
#     'CQL_MDP_baseline',
#     'CQL_MDP_Slow0.67',
#     'CQL_MDP_Slow0.33',
#     'CQL_MDP_Slow0.1',
#     'CQL_MDP_Slow0.01',
# ]
# base_names = [
#     cql_mdp_t1,
#     cql_finetune_slow067,
#     cql_finetune_slow033,
#     cql_finetune_slow01,
#     cql_finetune_slow001
# ]

# labels = [
#     'CQL_2x',
#     'CQL_MDP'
# ]
# base_names = [
#     cql_2x,
#     cql_mdp_t1
# ]

# labels = [
#     'CQL',
#     'CQL_MDP_Base',
#     'CQL_Layer',
#     'CQL_Model',
#     'CQL_Whole'
# ]
# base_names = [
#     cql_1x,
#     cql_mdp_t1,
#     cql_init_layerG,
#     cql_init_modelG,
#     cql_init_wholeG
# ]

labels = [
    'CQL',
    'CQL+MDP',
    'CQL+IID'
]

base_names = [
    iclr_cql,
    iclr_cql_mdp_t1,
    iclr_cql_iid_preT100k
]


data_path = '../../code/checkpoints/final'
save_path = '../../figures/'
plot_cql_performance_curves(labels, base_names)
# plot_cql_q_loss_curves(labels, base_names)
plot_cql_combined_loss_curves(labels, base_names)
# plot_dt_performance_curves()
# plot_dt_loss_curves()
