"""Use this to help us know where the training logs are"""

"""
DT related
"""





"""
CQL related
"""

# CQL baselines
cql_base = 'cqlr3_prenone_l2_qflrs1'
cql_fd_pretrain = 'cqlr3_preq_sprime_l2_qflrs1'
cql_random_pretrain = 'cqlr3_prerand_q_sprime_l2'

# CQL data size
cql_no_pretrain_0_1_data = 'cqlr3_prenone_l2_dr0.1'
cql_no_pretrain_0_25_data = 'cqlr3_prenone_l2_dr0.25'
cql_no_pretrain_0_5_data = 'cqlr3_prenone_l2_dr0.5'
cql_no_pretrain_0_75_data = 'cqlr3_prenone_l2_dr0.75'
cql_no_pretrain_1_data = cql_base


cql_pretrain_0_1_data = 'cqlr3_preq_sprime_l2_dr0.1'
cql_pretrain_0_25_data = 'cqlr3_preq_sprime_l2_dr0.25'
cql_pretrain_0_5_data = 'cqlr3_preq_sprime_l2_dr0.5'
cql_pretrain_0_75_data = 'cqlr3_preq_sprime_l2_dr0.75'
cql_pretrain_1_data = cql_fd_pretrain



# CQL pretrain epoch

# CQL 3x data
# 'q_noact_sprime', 'q_sprime_3x', 'proj0_q_sprime_3x', 'proj1_q_sprime_3x'
cql_fd_3x_data = 'cqlr3_preq_sprime_3x_l2'
cql_fd_3x_data_with_projection = 'cqlr3_preproj0_q_sprime_3x_l2'
cql_fd_3x_data_cross_task = 'cqlr3_preproj1_q_sprime_3x_l2'

# CQL alternative pretrain scheme
cql_no_action_predict_next_state = 'cqlr3_preq_noact_sprime_l2'


# CQL cross domain
cql_fd_pretrain_same_task_with_projection = 'cqlr3_preproj0_q_sprime_l2'
cql_fd_pretrain_cross_task1 = 'cqlr3_preproj1_q_sprime_l2'
cql_fd_pretrain_cross_task2 = 'cqlr3_preproj2_q_sprime_l2'

# CQL MDP
cql_mdp_pretrain_nstate_base = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue'

cql_mdp_pretrain_nstate1 = 'cqlr3_premdp_q_sprime_l2_ns1_pt1_sd20_sameTrue'
cql_mdp_pretrain_nstate10 = 'cqlr3_premdp_q_sprime_l2_ns10_pt1_sd20_sameTrue'
cql_mdp_pretrain_nstate100 = 'cqlr3_premdp_q_sprime_l2_ns100_pt1_sd20_sameTrue'
cql_mdp_pretrain_nstate1000 = cql_mdp_pretrain_nstate_base
cql_mdp_pretrain_nstate10000= 'cqlr3_premdp_q_sprime_l2_ns10000_pt1_sd20_sameTrue'
cql_mdp_pretrain_nstate50257= 'cqlr3_premdp_q_sprime_l2_ns50257_pt1_sd20_sameTrue'

cql_mdp_pretrain_temperature0_01 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt0.01_sd20_sameTrue'
cql_mdp_pretrain_temperature0_1 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt0.1_sd20_sameTrue'
cql_mdp_pretrain_temperature1 = cql_mdp_pretrain_nstate_base
cql_mdp_pretrain_temperature10 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt10_sd20_sameTrue'
cql_mdp_pretrain_temperature100 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt100_sd20_sameTrue'

cql_mdp_pretrain_state_action_dim1 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd1_sameTrue'
cql_mdp_pretrain_state_action_dim5 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd5_sameTrue'
cql_mdp_pretrain_state_action_dim20 = cql_mdp_pretrain_nstate_base
cql_mdp_pretrain_state_action_dim50 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd50_sameTrue'
cql_mdp_pretrain_state_action_dim200 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd200_sameTrue'
cql_mdp_pretrain_state_action_dim1000 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd1000_sameTrue'

cql_mdp_pretrain_same_dim_no_projection = 'cqlr3_premdp_same_noproj_l2_ns1000_pt1_sameTrue'
cql_mdp_pretrain_same_dim_with_projection = 'cqlr3_premdp_same_proj_l2_ns1000_pt1_sameTrue'


