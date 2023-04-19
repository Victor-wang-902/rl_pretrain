from plot_utils.quick_plot_helper import quick_plot

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

quick_plot( # labels, folder name prefix, envs
    [
        'dt_baseline',
        'il',
        'cql',
    ],
    [
        'cpubase_dt',
        'rl_il',
        'rl_cql',
    ],
    envs=envs,
    save_name='debug',
    base_data_folder_path=data_path,
    save_folder_path=save_path,
    y_value=standard_ys
)

