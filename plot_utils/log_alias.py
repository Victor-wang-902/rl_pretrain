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


