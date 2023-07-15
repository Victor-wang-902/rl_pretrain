"""Use this to help us know where the training logs are"""

"""
DT related
"""
# DT baselines
dt = 'dt-rerun-data_size_dt_1.0'
chibiT = 'chibiT-rerun'

# DT data size experiment
dt_finetune_data_size_0_1 = 'dt-rerun-data_size_dt_0.1'
dt_finetune_data_size_0_25 = 'dt-rerun-data_size_dt_0.25'
dt_finetune_data_size_0_5 = 'dt-rerun-data_size_dt_0.5'
dt_finetune_data_size_0_75 = 'dt-rerun-data_size_dt_0.75'
dt_finetune_data_size_1 = dt

chibiT_finetune_data_size_0_1 = 'chibiT-rerun-data_size_dt_0.1'
chibiT_finetune_data_size_0_25 = 'chibiT-rerun-data_size_dt_0.25'
chibiT_finetune_data_size_0_5 = 'chibiT-rerun-data_size_dt_0.5'
chibiT_finetune_data_size_0_75 = 'chibiT-rerun-data_size_dt_0.75'
chibiT_finetune_data_size_1 = chibiT

# DT with different model sizes
dt_model_size_default = dt
dt_model_size_4layer_256 = 'dt_embed_dim256_n_layer4_n_head4'
dt_model_size_6layer_512 = 'dt_embed_dim512_n_layer6_n_head8'
dt_model_size_12layer_768 = 'dt_embed_dim768_n_layer12_n_head12'

chibiT_model_size_default = chibiT
chibiT_model_size_4layer_256 = 'chibiT_embed_dim256_n_layer4_n_head4'
chibiT_model_size_6layer_512 = 'chibiT_embed_dim512_n_layer6_n_head8'
chibiT_model_size_12layer_768 = 'chibiT_embed_dim768_n_layer12_n_head12'


# DT with markov chain pretraining
dt_mc_1step_vocab10 = 'chibiT-rerun-syn_ngram1_nvocab10_temperature1.0'
dt_mc_1step_vocab100 = 'chibiT-rerun-syn_ngram1_nvocab100_temperature1.0'
dt_mc_1step_vocab1000 = 'chibiT-rerun-syn_ngram1_nvocab1000_temperature1.0'
dt_mc_1step_vocab10000 = 'chibiT-rerun-syn_ngram1_nvocab10000_temperature1.0'
dt_mc_1step_vocab50257 = 'chibiT-rerun-syn_ngram1_nvocab50257_temperature1.0'
dt_mc_1step_vocab100000 = 'chibiT-rerun-syn_ngram1_nvocab100000_temperature1.0'

dt_mc_2step_vocab50257 = 'chibiT-rerun-syn_ngram2_nvocab50257_temperature1.0'
dt_mc_3step_vocab50257 = 'chibiT-rerun-syn_ngram3_nvocab50257_temperature1.0'
dt_mc_4step_vocab50257 = 'chibiT-rerun-syn_ngram4_nvocab50257_temperature1.0'
dt_mc_5step_vocab50257 = 'chibiT-rerun-syn_ngram5_nvocab50257_temperature1.0'

dt_mc_temp0_1_vocab50257 = 'chibiT-rerun-syn_ngram1_nvocab50257_temperature0.1'
dt_mc_temp0_2_vocab50257 = 'chibiT-rerun-syn_ngram1_nvocab50257_temperature0.2'
dt_mc_temp0_4_vocab50257 = 'chibiT-rerun-syn_ngram1_nvocab50257_temperature0.4'
dt_mc_temp0_8_vocab50257 = 'chibiT-rerun-syn_ngram1_nvocab50257_temperature0.8'
dt_mc_temp1_0_vocab50257 = 'chibiT-rerun-syn_ngram1_nvocab50257_temperature1.0'
dt_mc_temp10_0_vocab50257 = 'chibiT-rerun-syn_ngram1_nvocab50257_temperature10.0'


dt_mc_2step_vocab100 = 'chibiT-rerun-syn_ngram2_nvocab100_temperature1.0'
dt_mc_3step_vocab100 = 'chibiT-rerun-syn_ngram3_nvocab100_temperature1.0'
dt_mc_4step_vocab100 = 'chibiT-rerun-syn_ngram4_nvocab100_temperature1.0'
dt_mc_5step_vocab100 = 'chibiT-rerun-syn_ngram5_nvocab100_temperature1.0'

dt_mc_temp0_1_vocab100 = 'chibiT-rerun-syn_ngram1_nvocab100_temperature0.1'
dt_mc_temp0_2_vocab100 = 'chibiT-rerun-syn_ngram1_nvocab100_temperature0.2'
dt_mc_temp0_4_vocab100 = 'chibiT-rerun-syn_ngram1_nvocab100_temperature0.4'
dt_mc_temp0_8_vocab100 = 'chibiT-rerun-syn_ngram1_nvocab100_temperature0.8'
dt_mc_temp10_0_vocab100 = 'chibiT-rerun-syn_ngram1_nvocab100_temperature10.0'


dt_same_data = 'same_new_ft'


"""
CQL related
"""

# CQL baselines
cql_base = 'cqlr3_prenone_l2_qflrs1'
cql_fd_pretrain = 'cqlr3_preq_sprime_l2_qflrs1'
cql_random_pretrain = 'cqlr3_prerand_q_sprime_l2'
cql_random_1000_state = 'cqlr3_prerandom_fd_1000_state_l2'

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

# with target network update



# CQL pretrain epoch

# CQL pretrain and offline data size variants
cql_fd_pretrain_data_ratio_0_01 = 'cqlr3_preq_sprime_l2_pdr0.01'
cql_fd_pretrain_data_ratio_0_1 = 'cqlr3_preq_sprime_l2_pdr0.1'
cql_fd_pretrain_data_ratio_0_25 = 'cqlr3_preq_sprime_l2_pdr0.25'
cql_fd_pretrain_data_ratio_0_5 = 'cqlr3_preq_sprime_l2_pdr0.5'
cql_fd_pretrain_data_ratio_0_75 = 'cqlr3_preq_sprime_l2_pdr0.75'
cql_fd_pretrain_data_ratio_1 = cql_fd_pretrain


cql_mdp_pretrain_data_ratio_0_01 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr0.01'
cql_mdp_pretrain_data_ratio_0_1 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr0.1'
cql_mdp_pretrain_data_ratio_0_25 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr0.25'
cql_mdp_pretrain_data_ratio_0_5 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr0.5'
cql_mdp_pretrain_data_ratio_0_75 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr0.75'
cql_mdp_pretrain_data_ratio_1 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue'


cql_fd_finetune_data_ratio_0_01 = 'cqlr3_preq_sprime_l2_dr0.01'
cql_fd_finetune_data_ratio_0_1 = 'cqlr3_preq_sprime_l2_dr0.1'
cql_fd_finetune_data_ratio_0_25 = 'cqlr3_preq_sprime_l2_dr0.25'
cql_fd_finetune_data_ratio_0_5 = 'cqlr3_preq_sprime_l2_dr0.5'
cql_fd_finetune_data_ratio_0_75 = 'cqlr3_preq_sprime_l2_dr0.75'
cql_fd_finetune_data_ratio_1 = cql_fd_pretrain

cql_mdp_finetune_data_ratio_0_01 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr1_dr0.01'
cql_mdp_finetune_data_ratio_0_1 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr1_dr0.1'
cql_mdp_finetune_data_ratio_0_25 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr1_dr0.25'
cql_mdp_finetune_data_ratio_0_5 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr1_dr0.5'
cql_mdp_finetune_data_ratio_0_75 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr1_dr0.75'
cql_mdp_finetune_data_ratio_1 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue'


cql_fd_finetune_both_ratio_0_01 = 'cqlr3_preq_sprime_l2_bothdr0.01'
cql_fd_finetune_both_ratio_0_1 = 'cqlr3_preq_sprime_l2_bothdr0.1'
cql_fd_finetune_both_ratio_0_25 = 'cqlr3_preq_sprime_l2_bothdr0.25'
cql_fd_finetune_both_ratio_0_5 = 'cqlr3_preq_sprime_l2_bothdr0.5'
cql_fd_finetune_both_ratio_0_75 = 'cqlr3_preq_sprime_l2_bothdr0.75'
cql_fd_finetune_both_ratio_1 = cql_fd_pretrain


cql_mdp_with_target_hard_update = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr1_dr1_True'
cql_mdp_with_target_hard_update_1 = cql_mdp_with_target_hard_update
cql_mdp_with_target_hard_update_0_5 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr1_dr0.5_True'
cql_mdp_with_target_hard_update_0_1 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr1_dr0.1_True'
cql_mdp_with_target_hard_update_0_25 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr1_dr0.25_True'
cql_mdp_with_target_hard_update_0_75 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr1_dr0.75_True'
cql_mdp_with_target_hard_update_0_01 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr1_dr0.01_True'


cql_rl_with_target_hard_update = 'cqlr3_preq_sprime_l2_pdr1_dr1_True'


cql_rl_with_target_hard_update_0_01 = 'cqlr3_preq_sprime_l2_pdr1_dr0.01_True'
cql_rl_with_target_hard_update_0_1 = 'cqlr3_preq_sprime_l2_pdr1_dr0.1_True'
cql_rl_with_target_hard_update_0_25 = 'cqlr3_preq_sprime_l2_pdr1_dr0.25_True'
cql_rl_with_target_hard_update_0_5 = 'cqlr3_preq_sprime_l2_pdr1_dr0.5_True'
cql_rl_with_target_hard_update_0_75 = 'cqlr3_preq_sprime_l2_pdr1_dr0.75_True'
cql_rl_with_target_hard_update_1 = cql_rl_with_target_hard_update


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


# new ones
cql_jul = 'cqlr3n_prenone_l2'
cql_jul_fd_pretrain = 'cqlr3n_preq_sprime_l2'
cql_jul_mdp_noproj_s1_t1 = 'cqlr3n_premdp_same_noproj_l2_ns1_pt1_sameTrue'
cql_jul_mdp_noproj_s10_t1 = 'cqlr3n_premdp_same_noproj_l2_ns10_pt1_sameTrue'
cql_jul_mdp_noproj_s100_t1 = 'cqlr3n_premdp_same_noproj_l2_ns100_pt1_sameTrue'
cql_jul_mdp_noproj_s1000_t1 = 'cqlr3n_premdp_same_noproj_l2_ns1000_pt1_sameTrue'
cql_jul_mdp_noproj_s10000_t1 = 'cqlr3n_premdp_same_noproj_l2_ns10000_pt1_sameTrue'
cql_jul_mdp_noproj_s100000_t1 = 'cqlr3n_premdp_same_noproj_l2_ns100000_pt1_sameTrue'

cql_jul_mdp_noproj_s100_t0_0001 = 'cqlr3n_premdp_same_noproj_l2_ns100_pt0.0001_sameTrue'
cql_jul_mdp_noproj_s100_t0_001 = 'cqlr3n_premdp_same_noproj_l2_ns100_pt0.001_sameTrue'
cql_jul_mdp_noproj_s100_t0_01 = 'cqlr3n_premdp_same_noproj_l2_ns100_pt0.01_sameTrue'
cql_jul_mdp_noproj_s100_t0_1 = 'cqlr3n_premdp_same_noproj_l2_ns100_pt0.1_sameTrue'
cql_jul_mdp_noproj_s100_t10 = 'cqlr3n_premdp_same_noproj_l2_ns100_pt10_sameTrue'
cql_jul_mdp_noproj_s100_t100 = 'cqlr3n_premdp_same_noproj_l2_ns100_pt100_sameTrue'
cql_jul_mdp_noproj_s100_t1000 = 'cqlr3n_premdp_same_noproj_l2_ns100_pt1000_sameTrue'

cql_jul_mdp_noproj_s100000_t0_0001 = 'cqlr3n_premdp_same_noproj_l2_ns100000_pt0.0001_sameTrue'
cql_jul_mdp_noproj_s100000_t1000 = 'cqlr3n_premdp_same_noproj_l2_ns100000_pt1000_sameTrue'
cql_jul_mdp_noproj_datainf_sinf_iid = 'cqlr3n_premdp_same_noproj_l2_ns42_pt42_sameTrue'
cql_jul_mdp_noproj_s100_iid = 'cqlr3n_premdp_same_noproj_l2_ns100_pt9999999_sameTrue'
cql_jul_mdp_noproj_s100000_iid = 'cqlr3n_premdp_same_noproj_l2_ns100000_pt9999999_sameTrue'
cql_jul_mdp_noproj_s10000000_iid = 'cqlr3n_premdp_same_noproj_l2_ns10000000_pt9999999_sameTrue'

cql_jul_mdp_noproj_1mdata_sinf_iid = 'cqlr3n_premdp_same_noproj_l2_ns12345678_pt9999999_sameTrue'
cql_jul_mdp_noproj_10mdata_sinf_iid = 'cqlr3n_premdp_same_noproj_l2_nt10000_ns40000000_pt9999999_sameTrue'
cql_jul_mdp_noproj_20mdata_sinf_iid = 'cqlr3n_premdp_same_noproj_l2_nt20000_ns40000000_pt9999999_sameTrue'




cql_jul_pdr0_001_dr0_001 = 'cqlr3n_prenone_l2_pdr0.001_dr0.001'
cql_jul_pdr0_001_dr1 = 'cqlr3n_prenone_l2_pdr0.001_dr1'
cql_jul_pdr1_dr0_001 = 'cqlr3n_prenone_l2_pdr1_dr0.001'
cql_jul_pdr1_dr1 = cql_jul

cql_jul_same_data_pdr0_001_dr0_001 = 'cqlr3n_preq_sprime_l2_pdr0.001_dr0.001'
cql_jul_same_data_pdr0_001_dr1 = 'cqlr3n_preq_sprime_l2_pdr0.001_dr1'
cql_jul_same_data_pdr1_dr0_001 = 'cqlr3n_preq_sprime_l2_pdr1_dr0.001'
cql_jul_same_data_pdr1_dr1 = cql_jul_fd_pretrain

cql_jul_mdp_pdr0_001_dr0_001 = 'cqlr3n_premdp_same_noproj_l2_ns100_pt1_sameTrue_pdr0.001_dr0.001'
cql_jul_mdp_pdr0_001_dr1 = 'cqlr3n_premdp_same_noproj_l2_ns100_pt1_sameTrue_pdr0.001_dr1'
cql_jul_mdp_pdr1_dr0_001 = 'cqlr3n_premdp_same_noproj_l2_ns100_pt1_sameTrue_pdr1_dr0.001'
cql_jul_mdp_pdr1_dr1 = cql_jul_mdp_noproj_s100_t1

