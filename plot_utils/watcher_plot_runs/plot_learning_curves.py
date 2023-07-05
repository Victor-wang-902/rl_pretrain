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

MUJOCO_3_ENVS = ['hopper', 'walker2d', 'halfcheetah',  ]
MUJOCO_3_DATASETS = ['medium','medium-replay','medium-expert',]
d4rl_9_datasets_envs = []
for e in MUJOCO_3_ENVS:
    for d in MUJOCO_3_DATASETS:
        d4rl_9_datasets_envs.append('%s_%s' % (e, d))

d4rl_q_loss_maxs = [60, 60, 60, 120, 120, 120, 150, 150, 150]

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

d4rl_dt_loss_col_name = 'current_itr_train_loss_mean'

default_performance_smooth = 5

def plot_cql_performance_curves():
    labels = [
        'CQL',
        'CQL same task',
        'CQL MDP',
    ]
    base_names = [
        cql_base,
        cql_fd_pretrain,
        cql_mdp_pretrain_nstate100,
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
        cql_base,
        cql_fd_pretrain,
        cql_mdp_pretrain_nstate100,
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
        )

def plot_dt_performance_curves():
    labels = [
        'DT',
        'DT Wiki',
        'DT MC',
    ]
    base_names = [
        dt,
        chibiT,
        dt_mc_1step_vocab100,
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
    ]
    base_names = [
        dt,
        chibiT,
        dt_mc_1step_vocab100,
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


# plot_cql_performance_curves()
# plot_cql_q_loss_curves()
# plot_dt_performance_curves()
# plot_dt_loss_curves()






def plot_july_new1():
    labels = [
        'CQL',
        'CQL same task',
    ]
    base_names = [
        cql_jul_new,
        cql_fd_pretrain_jul_new,
        ]

    y = 'sac_combined_loss'
    ymax = 100

    save_path = '../../figures/july_test'

    # aggregate
    aggregate_name = 'agg-julnew'
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
    ymax = None
    for env_dataset_name in d4rl_9_datasets_envs:
        quick_plot_with_full_name(  # labels, folder name prefix, envs
            labels,
            get_full_names_with_envs(base_names, [env_dataset_name]),
            save_name_prefix='ind-julnew',
            base_data_folder_path=data_path,
            save_folder_path=save_path,
            y_value=[y],
            x_to_use=d4rl_x_axis_col_name,
            ymax=ymax,
            save_name_suffix=env_dataset_name,
        smooth=default_performance_smooth
        )

def plot_july_new2():
    labels = [
        'CQL new 10seed',
        'CQL same task old',
        'CQL same new 10seed',
    ]
    base_names = [
        cql_jul_new,
        cql_fd_pretrain,
        cql_fd_pretrain_jul_new,
        ]

    y = d4rl_test_performance_col_name

    save_path = '../../figures/july_test'

    # aggregate
    ymax = None
    aggregate_name = 'agg-julnew'
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
    ymax = None
    for env_dataset_name in d4rl_9_datasets_envs:
        quick_plot_with_full_name(  # labels, folder name prefix, envs
            labels,
            get_full_names_with_envs(base_names, [env_dataset_name]),
            save_name_prefix='ind-julnew',
            base_data_folder_path=data_path,
            save_folder_path=save_path,
            y_value=[y],
            x_to_use=d4rl_x_axis_col_name,
            ymax=ymax,
            save_name_suffix=env_dataset_name,
        smooth=default_performance_smooth
        )

# plot_july_new1()
plot_july_new2()
