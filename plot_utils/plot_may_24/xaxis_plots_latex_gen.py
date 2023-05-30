
plot_measures = ['best_return_normalized', 'best_return_normalized_std', 'convergence_iter',
                 'best_feature_diff', 'best_weight_diff', "best_0_weight_diff", "best_1_weight_diff",
                 'best_feature_sim', 'best_weight_sim', "best_0_weight_sim", "best_1_weight_sim",
                 ]
plot_y_labels = ['Best Normalized Score', 'Best Normalized Score Std', 'Convergence Iter',
                'Best Feature Diff', 'Best Weight Diff', 'Best Weight Diff L0',  'Best Weight Diff L1',
                 'Best Feature Sim', 'Best Weight Sim', 'Best Weight Sim L0', 'Best Weight Sim L1',
                 ]

prefix_list = [
    'cql_q_distill_weight',
    'cql_q_distill_pre_epoch',
    'dt_pretrain_perturb'
]
caption_list = [
    'CQL with different weight for Q distillation auxiliary loss.',
    'CQL with different epochs of Q distillation pretraining (but no distillation during CQL training)',
    'DT with Pretraining, with different level of perturbation on the pretrained network weights.'
]


def print_figures_latex(prefix, caption='', ref_label=''):
    # 9 subfigures, each for one plot measure
    print("\\begin{figure}[htb]")
    print("\\centering")
    for i in range(len(plot_measures)):
        suffix = plot_measures[i]
        label = plot_y_labels[i]
        print('\\begin{subfigure}[t]{.325\\linewidth}')
        print('\\centering')
        print('\\includegraphics[width=\\linewidth]{figures2/%s_%s.png}' % (prefix, suffix))
        print('\\caption{%s}' % label)
        print('\\end{subfigure}')
    print('\\caption{%s}' % caption)
    print('\\label{%s}' % ref_label)
    print('\\end{figure}')
    print()

for prefix, caption in zip(prefix_list, caption_list):
    print_figures_latex(prefix, caption=caption)




