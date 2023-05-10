
plot_measures = ['best_return_normalized', 'best_weight_diff', 'best_feature_diff',
                'final_test_normalized_returns', 'final_weight_diff', 'final_feature_diff',
                'best_return_normalized_std', 'final_test_normalized_returns_std', 'convergence_iter',
                 ]
plot_y_labels = ['Best Normalized Score', 'Best Weight l2 Diff', 'Best Feature L2 Diff',
                 'Final Normalized Score', 'Final Weight l2 Diff', 'Final Feature L2 Diff',
                 'Best Normalized Score Std', 'Final Normalized Score Std', 'Convergence Iter',
                 ]

prefix_list = ['cql_pretrain_epochs',
               'cql_layers',
               'dt_rl_datasize',
               'dt_modelsize',
               'dt_pretrain_perturb']
caption_list = ['CQL with different forward dynamics pretraining epochs.',
                'CQL layers comparison. Y-axis in each figure shows an average measure across 9 MuJoCo datasets.',
                'DT and DT with pretraining, with different offline dataset ratio. ',
                'DT with different model sizes.',
                'DT with pretraining, and different perturbation noise std into the pretrained mdoel. ',
                ]


def print_figures_latex(prefix, caption='', ref_label=''):
    # 9 subfigures, each for one plot measure
    print("\\begin{figure}[htb]")
    print("\\centering")
    for i in range(9):
        suffix = plot_measures[i]
        label = plot_y_labels[i]
        print('\\begin{subfigure}[t]{.325\\linewidth}')
        print('\\centering')
        print('\\includegraphics[width=\\linewidth]{figures/%s_%s.png}' % (prefix, suffix))
        print('\\caption{%s}' % label)
        print('\\end{subfigure}')
    print('\\caption{%s}' % caption)
    print('\\label{%s}' % ref_label)
    print('\\end{figure}')
    print()

for prefix, caption in zip(prefix_list, caption_list):
    print_figures_latex(prefix, caption=caption)




