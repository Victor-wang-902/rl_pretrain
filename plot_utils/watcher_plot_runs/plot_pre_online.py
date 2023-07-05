from plot_utils.quick_plot_helper import quick_plot, quick_plot_with_full_name
# standard plotting, might not be useful for us...

twocolordoulbe = ['tab:blue', 'tab:orange', 'tab:blue', 'tab:orange',]
twosoliddashed = ['dashed', 'dashed',  'solid', 'solid', ]
threecolordoulbe = ['tab:blue', 'tab:orange', 'tab:red', 'tab:blue', 'tab:orange', 'tab:red']
threesoliddashed = ['dashed', 'dashed', 'dashed', 'solid', 'solid', 'solid', ]
standard_6_colors = ('tab:red', 'tab:orange', 'tab:blue', 'tab:brown', 'tab:pink','tab:grey')

MUJOCO_3_ENVS = ['hopper', 'walker2d', 'halfcheetah',  ]
MUJOCO_3_DATASETS = ['medium','medium-replay','medium-expert',]
envs = ['Hopper-v2', 'HalfCheetah-v2', 'Walker2d-v2', 'Ant-v2', 'Humanoid-v2']

envs = ['HalfCheetah-v2']
envs = ['Hopper-v2',]
envs = ['HalfCheetah-v2']
envs = ['Walker2d-v2',]

data_path = '../../code/checkpoints/'
save_path = '../../figures/'

standard_ys = ['TestEpRet', 'weight_diff_last_iter',
               'feature_diff_last_iter',
               'weight_sim_last_iter', 'feature_sim_last_iter']
standard_ys = ['AverageTestEpRet']
do_sac_mdp = True

def get_full_names_with_envs(base_names):
    n = len(base_names)
    to_return = []
    for i in range(n):
        new_list = []
        for env in envs:
            full_name = base_names[i] + '_' + env
            new_list.append(full_name)
        to_return.append(new_list)
    return to_return

if do_sac_mdp:
    labels =         [
        'sac',
        'sac mdp pretrain',
    ]
    base_names = [
            'sacpre_ep3000_nq2_utd1_mdpFalse',
            'sacpre_ep3000_nq2_utd1_mdpTrue',
        ]
    quick_plot_with_full_name( # labels, folder name prefix, envs
        labels,
        get_full_names_with_envs(base_names),
        save_name_prefix='agg_sac',
        base_data_folder_path=data_path,
        save_folder_path=save_path,
        y_value=standard_ys,
        x_to_use='TotalEnvInteracts'
    )
