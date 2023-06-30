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

d4rl_test_performance_col_name = 'TestEpRet'
d4rl_q_loss_col_name = 'sac_qf1_loss'
d4rl_x_axis_col_name = 'Steps'
d4rl_q_value_col_name = 'sac_average_qf1'


labels =         [
    'cql',
    'cql pretrain',
    'cql mdp',
    'target up'
]
base_names = [
        cql_base,
        cql_fd_pretrain,
        cql_mdp_pretrain_temperature1,
    cql_mdp_with_target_hard_update,
    ]
quick_plot_with_full_name( # labels, folder name prefix, envs
    labels,
    get_full_names_with_envs(base_names, d4rl_9_datasets_envs),
    save_name='cql',
    base_data_folder_path=data_path,
    save_folder_path=save_path,
    y_value=[d4rl_test_performance_col_name, d4rl_q_loss_col_name, d4rl_q_value_col_name],
    x_to_use=d4rl_x_axis_col_name,
    ymin=0,
    ymax=500,
)
