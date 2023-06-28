from plot_utils.quick_plot_helper import quick_plot, quick_plot_with_full_name
# standard plotting, might not be useful for us...

twocolordoulbe = ['tab:blue', 'tab:orange', 'tab:blue', 'tab:orange',]
twosoliddashed = ['dashed', 'dashed',  'solid', 'solid', ]
threecolordoulbe = ['tab:blue', 'tab:orange', 'tab:red', 'tab:blue', 'tab:orange', 'tab:red']
threesoliddashed = ['dashed', 'dashed', 'dashed', 'solid', 'solid', 'solid', ]
standard_6_colors = ('tab:red', 'tab:orange', 'tab:blue', 'tab:brown', 'tab:pink','tab:grey')

MUJOCO_3_ENVS = ['hopper', 'walker2d', 'halfcheetah',  ]
MUJOCO_3_DATASETS = ['medium','medium-replay','medium-expert',]
envs = []

for e in MUJOCO_3_ENVS:
    for dataset in MUJOCO_3_DATASETS:
        envs.append('%s_%s' % (e, dataset))

data_path = '../../code/checkpoints/'
save_path = '../../figures/'

standard_ys = ['TestEpRet', 'TestEpNormRet', 'total_time']

do_cql_arch = False
do_cql_pretrain_arch = True

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

if do_cql_arch:
    labels = [
            'cql l2',
            'cql l4',
            'cql l6',
            'cql l8',
        ]
    base_names = [
            'cql_prenone',
            'cql_prenone_pe200_layer4',
            'cql_prenone_pe200_layer6',
            'cql_prenone_pe200_layer8',
        ]
    quick_plot_with_full_name( # labels, folder name prefix, envs
        labels,
        get_full_names_with_envs(base_names),
        save_name='agg_cql_layers',
        base_data_folder_path=data_path,
        save_folder_path=save_path,
        y_value=standard_ys
    )

if do_cql_pretrain_arch:
    labels =         [
            'cql pre l2',
            'cql pre l4',
            'cql pre l6',
            'cql pre l8',
        ]
    base_names = [
            'cql_preq_sprime',
            'cql_preq_sprime_pe200_layer4',
            'cql_preq_sprime_pe200_layer6',
            'cql_preq_sprime_pe200_layer8',
        ]
    quick_plot_with_full_name( # labels, folder name prefix, envs
        labels,
        get_full_names_with_envs(base_names),
        save_name='agg_cql_pretrain_layers',
        base_data_folder_path=data_path,
        save_folder_path=save_path,
        y_value=standard_ys
    )
