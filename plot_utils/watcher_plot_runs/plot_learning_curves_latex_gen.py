
# plot_measures = ['best_return_normalized', 'best_weight_diff', 'best_feature_diff',
#                 'final_test_normalized_returns', 'final_weight_diff', 'final_feature_diff',
#                 'best_return_normalized_std', 'final_test_normalized_returns_std', 'convergence_iter',
#                  ]
# plot_y_labels = ['Best Normalized Score', 'Best Weight l2 Diff', 'Best Feature L2 Diff',
#                  'Final Normalized Score', 'Final Weight l2 Diff', 'Final Feature L2 Diff',
#                  'Best Normalized Score Std', 'Final Normalized Score Std', 'Convergence Iter',
#                  ]
plot_measures = ['best_return_normalized', 'best_return_normalized_std', 'convergence_iter',
                 'best_feature_diff', 'best_weight_diff', "best_0_weight_diff", "best_1_weight_diff",
                 'best_feature_sim', 'best_weight_sim', "best_0_weight_sim", "best_1_weight_sim",
                 ]
plot_y_labels = ['Best Normalized Score', 'Best Normalized Score Std', 'Convergence Iter',
                'Best Feature Diff', 'Best Weight Diff', 'Best Weight Diff L0',  'Best Weight Diff L1',
                 'Best Feature Sim', 'Best Weight Sim', 'Best Weight Sim L0', 'Best Weight Sim L1',
                 ]

prefix_list = ['cql_pretrain_epochs',
               # 'cql_layers',
               'rl_datasize',
               'dt_modelsize',
               'dt_pretrain_perturb']
caption_list = ['CQL with different forward dynamics pretraining epochs.',
                # 'CQL layers comparison. Y-axis in each figure shows an average measure across 9 MuJoCo datasets.',
                'with different offline dataset ratio. ',
                'DT with different model sizes.',
                'DT with pretraining, and different perturbation noise std into the pretrained mdoel. ',
                ]



def print_figures_latex(figure_folder, figure_names, sub_figure_captions, caption='', ref_label=''):
    # 9 subfigures, each for one plot measure
    print("\\begin{figure}[htb]")
    print("\\centering")
    for i in range(len(figure_names)):
        figure_name = figure_names[i]
        sub_figure_caption = sub_figure_captions[i]

        print('\\begin{subfigure}[t]{.3\\linewidth}')
        print('\\centering')
        print('\\includegraphics[width=\\linewidth]{%s/%s}' % (figure_folder, figure_name))
        print('\\caption{%s}' % sub_figure_caption)
        print('\\end{subfigure}')
    print('\\caption{%s}' % caption)
    print('\\label{%s}' % ref_label)
    print('\\end{figure}')
    print()


MUJOCO_3_ENVS = ['hopper', 'walker2d', 'halfcheetah', "ant" ]
MUJOCO_3_DATASETS = ['medium','medium-replay','medium-expert',]
d4rl_9_datasets_envs = []
for e in MUJOCO_3_ENVS:
    for d in MUJOCO_3_DATASETS:
        d4rl_9_datasets_envs.append('%s_%s' % (e, d))

MUJOCO_3_ENVS_captions = ['Hopper', 'Walker', 'HalfCheetah', "ant" ]
MUJOCO_3_DATASETS_captions = ['medium','medium-replay','medium-expert',]
d4rl_9_datasets_envs_captions = []
for e in MUJOCO_3_ENVS_captions:
    for d in MUJOCO_3_DATASETS_captions:
        d4rl_9_datasets_envs_captions.append('%s-%s' % (e, d))


def gen_cql_curves():
    figure_folder=  'figure-curves'
    figure_names = []
    subfigure_captions = d4rl_9_datasets_envs_captions

    # performance
    for e in d4rl_9_datasets_envs:
        figure_names.append('ind-cql_TestEpNormRet_%s.png' % e)

    caption = 'Learning curves for CQL, CQL with same task RL data pretraining, and CQL with MDP pretraining.'
    ref_label = 'fig:cql-performance-curves'
    print_figures_latex(
        figure_folder,
        figure_names,
        subfigure_captions,
        caption,
        ref_label,
    )

    # q loss
    figure_names = []
    for e in d4rl_9_datasets_envs:
        figure_names.append('ind-cql_sac_qf1_loss_%s.png' % e)

    caption = 'Standard Q loss for CQL, CQL with same task RL data pretraining, and CQL with MDP pretraining.'
    ref_label = 'fig:cql-q-loss-curves'
    print_figures_latex(
        figure_folder,
        figure_names,
        subfigure_captions,
        caption,
        ref_label,
    )

    # combined loss
    figure_names = []
    for e in d4rl_9_datasets_envs:
        figure_names.append('ind-cql_sac_combined_loss_%s.png' % e)

    caption = 'Combined loss for CQL, CQL with same task RL data pretraining, and CQL with MDP pretraining.'
    ref_label = 'fig:cql-combined-loss-curves'
    print_figures_latex(
        figure_folder,
        figure_names,
        subfigure_captions,
        caption,
        ref_label,
    )


def gen_dt_curves():
    figure_folder=  'figure-curves'
    figure_names = []
    subfigure_captions = d4rl_9_datasets_envs_captions

    # performance
    for e in d4rl_9_datasets_envs:
        figure_names.append('ind-dt_TestEpNormRet_%s.png' % e)

    caption = 'Learning curves for DT, DT with Wikipedia pretraining, and DT with MC pretraining.'
    ref_label = 'fig:dt-performance-curves'
    print_figures_latex(
        figure_folder,
        figure_names,
        subfigure_captions,
        caption,
        ref_label,
    )

    # q loss
    figure_names = []
    for e in d4rl_9_datasets_envs:
        figure_names.append('ind-dt_current_itr_train_loss_mean_%s.png' % e)

    caption = 'Training loss curves for DT, DT with Wikipedia pretraining, and DT with MC pretraining.'
    ref_label = 'fig:dt-train-loss-curves'
    print_figures_latex(
        figure_folder,
        figure_names,
        subfigure_captions,
        caption,
        ref_label,
    )

def gen_dt_curves_offset():
    figure_folder=  'figure-curves'
    figure_names = []
    subfigure_captions = d4rl_9_datasets_envs_captions

    # performance
    for e in d4rl_9_datasets_envs:
        figure_names.append('ind-dt-offset_TestEpNormRet_%s.png' % e)

    caption = 'Learning curves for DT, DT with Wikipedia pretraining, and DT with MC pretraining.'
    ref_label = 'fig:dt-performance-curves-offset'
    print_figures_latex(
        figure_folder,
        figure_names,
        subfigure_captions,
        caption,
        ref_label,
    )

    # q loss
    figure_names = []
    for e in d4rl_9_datasets_envs:
        figure_names.append('ind-dt-offset_current_itr_train_loss_mean_%s.png' % e)

    caption = 'Training loss curves for DT, DT with Wikipedia pretraining, and DT with MC pretraining.'
    ref_label = 'fig:dt-train-loss-curves-offset'
    print_figures_latex(
        figure_folder,
        figure_names,
        subfigure_captions,
        caption,
        ref_label,
    )


#gen_cql_curves()
gen_dt_curves_offset()
