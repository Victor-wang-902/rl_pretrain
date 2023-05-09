python experiment.py --env hopper --dataset medium --model_type dt --seed 666  --pretrained_lm chibiT  --outdir "checkpoints/cibiT_kmeans_medium_positions_hopper_perturb_new_mlp_8e0_rerun_666" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only --perturb_mlp_only
python experiment.py --env hopper --dataset medium --model_type dt --seed 42  --pretrained_lm chibiT  --outdir "checkpoints/cibiT_kmeans_medium_positions_hopper_perturb_new_mlp_8e0_rerun_666" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only --perturb_mlp_only
python experiment.py --env hopper --dataset medium --model_type dt --seed 1024  --pretrained_lm chibiT  --outdir "checkpoints/cibiT_kmeans_medium_positions_hopper_perturb_new_mlp_8e0_rerun_666" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only --perturb_mlp_only


python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 666 --data_size 0.1 --outdir "checkpoints/dt_halfcheetah_medium-expert_finetune_10_666" &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 24 --data_size 0.1 --outdir "checkpoints/dt_halfcheetah_medium-expert_finetune_10_24" &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 1024 --data_size 0.1 --outdir "checkpoints/dt_halfcheetah_medium-expert_finetune_10_1024" &
wait
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 666 --data_size 0.1 --outdir "checkpoints/dt_halfcheetah_medium_finetune_10_666" &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 24 --data_size 0.1 --outdir "checkpoints/dt_halfcheetah_medium_finetune_10_24" &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 1024 --data_size 0.1 --outdir "checkpoints/dt_halfcheetah_medium_finetune_10_1024" &
wait
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 666 --data_size 0.1 --outdir "checkpoints/dt_halfcheetah_medium_replay_finetune_10_666" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 24 --data_size 0.1 --outdir "checkpoints/dt_halfcheetah_medium_replay_finetune_10_24" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 1024 --data_size 0.1 --outdir "checkpoints/dt_halfcheetah_medium_replay_finetune_10_1024" &
wait
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 666 --data_size 0.1 --outdir "checkpoints/dt_hopper_medium-expert_finetune_10_666" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 24 --data_size 0.1 --outdir "checkpoints/dt_hopper_medium-expert_finetune_10_24" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 1024 --data_size 0.1 --outdir "checkpoints/dt_hopper_medium-expert_finetune_10_1024" &
wait
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 666 --data_size 0.1 --outdir "checkpoints/dt_hopper_medium_finetune_10_666" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 24 --data_size 0.1 --outdir "checkpoints/dt_hopper_medium_finetune_10_24" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 1024 --data_size 0.1 --outdir "checkpoints/dt_hopper_medium_finetune_10_1024" &
wait
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 666 --data_size 0.1 --outdir "checkpoints/dt_hopper_medium_replay_finetune_10_666" &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 24 --data_size 0.1 --outdir "checkpoints/dt_hopper_medium_replay_finetune_10_24" &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 1024 --data_size 0.1 --outdir "checkpoints/dt_hopper_medium_replay_finetune_10_1024" &
wait
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 666 --data_size 0.1 --outdir "checkpoints/dt_walker_medium-expert_finetune_10_666" &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 24 --data_size 0.1 --outdir "checkpoints/dt_walker_medium-expert_finetune_10_24" &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 1024 --data_size 0.1 --outdir "checkpoints/dt_walker_medium-expert_finetune_10_1024" &
wait
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 666 --data_size 0.1 --outdir "checkpoints/dt_walker_medium_finetune_10_666" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 24 --data_size 0.1 --outdir "checkpoints/dt_walker_medium_finetune_10_24" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 1024 --data_size 0.1 --outdir "checkpoints/dt_walker_medium_finetune_10_1024" &
wait
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 666 --data_size 0.1 --outdir "checkpoints/dt_walker_medium_replay_finetune_10_666" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 24 --data_size 0.1 --outdir "checkpoints/dt_walker_medium_replay_finetune_10_24" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 1024 --data_size 0.1 --outdir "checkpoints/dt_walker_medium_replay_finetune_10_1024" &
wait
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 666 --data_size 0.25 --outdir "checkpoints/dt_halfcheetah_medium-expert_finetune_25_666" &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 24 --data_size 0.25 --outdir "checkpoints/dt_halfcheetah_medium-expert_finetune_25_24" &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 1024 --data_size 0.25 --outdir "checkpoints/dt_halfcheetah_medium-expert_finetune_25_1024" &
wait
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 666 --data_size 0.25 --outdir "checkpoints/dt_halfcheetah_medium_finetune_25_666" &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 24 --data_size 0.25 --outdir "checkpoints/dt_halfcheetah_medium_finetune_25_24" &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 1024 --data_size 0.25 --outdir "checkpoints/dt_halfcheetah_medium_finetune_25_1024" &
wait
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 666 --data_size 0.25 --outdir "checkpoints/dt_halfcheetah_medium_replay_finetune_25_666" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 24 --data_size 0.25 --outdir "checkpoints/dt_halfcheetah_medium_replay_finetune_25_24" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 1024 --data_size 0.25 --outdir "checkpoints/dt_halfcheetah_medium_replay_finetune_25_1024" &
wait
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 666 --data_size 0.25 --outdir "checkpoints/dt_hopper_medium-expert_finetune_25_666" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 24 --data_size 0.25 --outdir "checkpoints/dt_hopper_medium-expert_finetune_25_24" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 1024 --data_size 0.25 --outdir "checkpoints/dt_hopper_medium-expert_finetune_25_1024" &
wait
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 666 --data_size 0.25 --outdir "checkpoints/dt_hopper_medium_finetune_25_666" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 24 --data_size 0.25 --outdir "checkpoints/dt_hopper_medium_finetune_25_24" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 1024 --data_size 0.25 --outdir "checkpoints/dt_hopper_medium_finetune_25_1024" &
wait
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 666 --data_size 0.25 --outdir "checkpoints/dt_hopper_medium_replay_finetune_25_666" &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 24 --data_size 0.25 --outdir "checkpoints/dt_hopper_medium_replay_finetune_25_24" &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 1024 --data_size 0.25 --outdir "checkpoints/dt_hopper_medium_replay_finetune_25_1024" &
wait
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 666 --data_size 0.25 --outdir "checkpoints/dt_walker_medium-expert_finetune_25_666" &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 24 --data_size 0.25 --outdir "checkpoints/dt_walker_medium-expert_finetune_25_24" &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 1024 --data_size 0.25 --outdir "checkpoints/dt_walker_medium-expert_finetune_25_1024" &
wait
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 666 --data_size 0.25 --outdir "checkpoints/dt_walker_medium_finetune_25_666" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 24 --data_size 0.25 --outdir "checkpoints/dt_walker_medium_finetune_25_24" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 1024 --data_size 0.25 --outdir "checkpoints/dt_walker_medium_finetune_25_1024" &
wait
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 666 --data_size 0.25 --outdir "checkpoints/dt_walker_medium_replay_finetune_25_666" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 24 --data_size 0.25 --outdir "checkpoints/dt_walker_medium_replay_finetune_25_24" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 1024 --data_size 0.25 --outdir "checkpoints/dt_walker_medium_replay_finetune_25_1024" &
wait
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 666 --data_size 0.5 --outdir "checkpoints/dt_halfcheetah_medium-expert_finetune_50_666" &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 24 --data_size 0.5 --outdir "checkpoints/dt_halfcheetah_medium-expert_finetune_50_24" &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 1024 --data_size 0.5 --outdir "checkpoints/dt_halfcheetah_medium-expert_finetune_50_1024" &
wait
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 666 --data_size 0.5 --outdir "checkpoints/dt_halfcheetah_medium_finetune_50_666" &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 24 --data_size 0.5 --outdir "checkpoints/dt_halfcheetah_medium_finetune_50_24" &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 1024 --data_size 0.5 --outdir "checkpoints/dt_halfcheetah_medium_finetune_50_1024" &
wait
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 666 --data_size 0.5 --outdir "checkpoints/dt_halfcheetah_medium_replay_finetune_50_666" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 24 --data_size 0.5 --outdir "checkpoints/dt_halfcheetah_medium_replay_finetune_50_24" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 1024 --data_size 0.5 --outdir "checkpoints/dt_halfcheetah_medium_replay_finetune_50_1024" &
wait
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 666 --data_size 0.5 --outdir "checkpoints/dt_hopper_medium-expert_finetune_50_666" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 24 --data_size 0.5 --outdir "checkpoints/dt_hopper_medium-expert_finetune_50_24" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 1024 --data_size 0.5 --outdir "checkpoints/dt_hopper_medium-expert_finetune_50_1024" &
wait
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 666 --data_size 0.5 --outdir "checkpoints/dt_hopper_medium_finetune_50_666" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 24 --data_size 0.5 --outdir "checkpoints/dt_hopper_medium_finetune_50_24" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 1024 --data_size 0.5 --outdir "checkpoints/dt_hopper_medium_finetune_50_1024" &
wait
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 666 --data_size 0.5 --outdir "checkpoints/dt_hopper_medium_replay_finetune_50_666" &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 24 --data_size 0.5 --outdir "checkpoints/dt_hopper_medium_replay_finetune_50_24" &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 1024 --data_size 0.5 --outdir "checkpoints/dt_hopper_medium_replay_finetune_50_1024" &
wait
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 666 --data_size 0.5 --outdir "checkpoints/dt_walker_medium-expert_finetune_50_666" &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 24 --data_size 0.5 --outdir "checkpoints/dt_walker_medium-expert_finetune_50_24" &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 1024 --data_size 0.5 --outdir "checkpoints/dt_walker_medium-expert_finetune_50_1024" &
wait
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 666 --data_size 0.5 --outdir "checkpoints/dt_walker_medium_finetune_50_666" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 24 --data_size 0.5 --outdir "checkpoints/dt_walker_medium_finetune_50_24" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 1024 --data_size 0.5 --outdir "checkpoints/dt_walker_medium_finetune_50_1024" &
wait
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 666 --data_size 0.5 --outdir "checkpoints/dt_walker_medium_replay_finetune_50_666" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 24 --data_size 0.5 --outdir "checkpoints/dt_walker_medium_replay_finetune_50_24" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 1024 --data_size 0.5 --outdir "checkpoints/dt_walker_medium_replay_finetune_50_1024" &
wait
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 666 --data_size 0.75 --outdir "checkpoints/dt_halfcheetah_medium-expert_finetune_75_666" &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 24 --data_size 0.75 --outdir "checkpoints/dt_halfcheetah_medium-expert_finetune_75_24" &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 1024 --data_size 0.75 --outdir "checkpoints/dt_halfcheetah_medium-expert_finetune_75_1024" &
wait
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 666 --data_size 0.75 --outdir "checkpoints/dt_halfcheetah_medium_finetune_75_666" &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 24 --data_size 0.75 --outdir "checkpoints/dt_halfcheetah_medium_finetune_75_24" &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 1024 --data_size 0.75 --outdir "checkpoints/dt_halfcheetah_medium_finetune_75_1024" &
wait
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 666 --data_size 0.75 --outdir "checkpoints/dt_halfcheetah_medium_replay_finetune_75_666" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 24 --data_size 0.75 --outdir "checkpoints/dt_halfcheetah_medium_replay_finetune_75_24" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 1024 --data_size 0.75 --outdir "checkpoints/dt_halfcheetah_medium_replay_finetune_75_1024" &
wait
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 666 --data_size 0.75 --outdir "checkpoints/dt_hopper_medium-expert_finetune_75_666" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 24 --data_size 0.75 --outdir "checkpoints/dt_hopper_medium-expert_finetune_75_24" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 1024 --data_size 0.75 --outdir "checkpoints/dt_hopper_medium-expert_finetune_75_1024" &
wait
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 666 --data_size 0.75 --outdir "checkpoints/dt_hopper_medium_finetune_75_666" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 24 --data_size 0.75 --outdir "checkpoints/dt_hopper_medium_finetune_75_24" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 1024 --data_size 0.75 --outdir "checkpoints/dt_hopper_medium_finetune_75_1024" &
wait
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 666 --data_size 0.75 --outdir "checkpoints/dt_hopper_medium_replay_finetune_75_666" &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 24 --data_size 0.75 --outdir "checkpoints/dt_hopper_medium_replay_finetune_75_24" &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 1024 --data_size 0.75 --outdir "checkpoints/dt_hopper_medium_replay_finetune_75_1024" &
wait
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 666 --data_size 0.75 --outdir "checkpoints/dt_walker_medium-expert_finetune_75_666" &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 24 --data_size 0.75 --outdir "checkpoints/dt_walker_medium-expert_finetune_75_24" &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 1024 --data_size 0.75 --outdir "checkpoints/dt_walker_medium-expert_finetune_75_1024" &
wait
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 666 --data_size 0.75 --outdir "checkpoints/dt_walker_medium_finetune_75_666" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 24 --data_size 0.75 --outdir "checkpoints/dt_walker_medium_finetune_75_24" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 1024 --data_size 0.75 --outdir "checkpoints/dt_walker_medium_finetune_75_1024" &
wait
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 666 --data_size 0.75 --outdir "checkpoints/dt_walker_medium_replay_finetune_75_666" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 24 --data_size 0.75 --outdir "checkpoints/dt_walker_medium_replay_finetune_75_24" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 1024 --data_size 0.75 --outdir "checkpoints/dt_walker_medium_replay_finetune_75_1024" &
wait

## These three has WRONG seed name!!! Fix after finish
python experiment.py --env halfcheetah --dataset medium-expert --model_type dt --seed 666 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_halfcheetah_medium-expert_size_3_666" &
python experiment.py --env halfcheetah --dataset medium-expert --model_type dt --seed 42 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_halfcheetah_medium-expert_size_3_24" &
python experiment.py --env halfcheetah --dataset medium-expert --model_type dt --seed 1024 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_halfcheetah_medium-expert_size_3_1024" &
wait
python experiment.py --env halfcheetah --dataset medium --model_type dt --seed 666 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_halfcheetah_medium_size_3_666" &
python experiment.py --env halfcheetah --dataset medium --model_type dt --seed 42 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_halfcheetah_medium_size_3_24" &
python experiment.py --env halfcheetah --dataset medium --model_type dt --seed 1024 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_halfcheetah_medium_size_3_1024" &
wait
python experiment.py --env hopper --dataset medium-expert --model_type dt --seed 666 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_hopper_medium-expert_size_3_666" &
python experiment.py --env hopper --dataset medium-expert --model_type dt --seed 42 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_hopper_medium-expert_size_3_24" &
python experiment.py --env hopper --dataset medium-expert --model_type dt --seed 1024 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_hopper_medium-expert_size_3_1024" &
wait



python experiment.py --env hopper --dataset medium-replay --model_type dt --seed 666 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_hopper_medium-replay_size_3_666" &
python experiment.py --env hopper --dataset medium-replay --model_type dt --seed 42 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_hopper_medium-replay_size_3_24" &
python experiment.py --env hopper --dataset medium-replay --model_type dt --seed 1024 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_hopper_medium-replay_size_3_1024" &
wait
python experiment.py --env hopper --dataset medium --model_type dt --seed 666 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_hopper_medium_size_3_666" &
python experiment.py --env hopper --dataset medium --model_type dt --seed 42 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_hopper_medium_size_3_24" &
python experiment.py --env hopper --dataset medium --model_type dt --seed 1024 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_hopper_medium_size_3_1024" &
wait
python experiment.py --env walker2d --dataset medium-expert --model_type dt --seed 666 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_walker_medium-expert_size_3_666" &
python experiment.py --env walker2d --dataset medium-expert --model_type dt --seed 42 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_walker_medium-expert_size_3_24" &
python experiment.py --env walker2d --dataset medium-expert --model_type dt --seed 1024 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_walker_medium-expert_size_3_1024" &
wait
python experiment.py --env walker2d --dataset medium-replay --model_type dt --seed 666 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_walker_medium-replay_size_3_666" &
python experiment.py --env walker2d --dataset medium-replay --model_type dt --seed 42 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_walker_medium-replay_size_3_24" &
python experiment.py --env walker2d --dataset medium-replay --model_type dt --seed 1024 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_walker_medium-replay_size_3_1024" &
wait
python experiment.py --env walker2d --dataset medium --model_type dt --seed 666 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_walker_medium_size_3_666" &
python experiment.py --env walker2d --dataset medium --model_type dt --seed 42 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_walker_medium_size_3_24" &
python experiment.py --env walker2d --dataset medium --model_type dt --seed 1024 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_walker_medium_size_3_1024" &
wait



python experiment.py --env hopper --dataset medium-expert --model_type dt --seed 42 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "checkpoints/dt_hopper_medium-expert_size_1_42" &
python experiment.py --env hopper --dataset medium-replay --model_type dt --seed 42 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "checkpoints/dt_hopper_medium-replay_size_1_42" &
python experiment.py --env hopper --dataset medium --model_type dt --seed 42 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "checkpoints/dt_hopper_medium_size_1_42" &
wait
python experiment.py --env walker2d --dataset medium-expert --model_type dt --seed 42 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "checkpoints/dt_walker2d_medium-expert_size_1_42" &
python experiment.py --env walker2d --dataset medium-replay --model_type dt --seed 42 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "checkpoints/dt_walker2d_medium-replay_size_1_42" &
python experiment.py --env walker2d --dataset medium --model_type dt --seed 42 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "checkpoints/dt_walker2d_medium_size_1_42" &
wait
python experiment.py --env halfcheetah --dataset medium-expert --model_type dt --seed 42 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "checkpoints/dt_halfcheetah_medium-expert_size_1_42" &
python experiment.py --env halfcheetah --dataset medium-replay --model_type dt --seed 42 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "checkpoints/dt_halfcheetah_medium-replay_size_1_42" &
python experiment.py --env halfcheetah --dataset medium --model_type dt --seed 42 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "checkpoints/dt_halfcheetah_medium_size_1_42" &
wait
python experiment.py --env hopper --dataset medium-expert --model_type dt --seed 42 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "checkpoints/dt_hopper_medium-expert_size_2_42" &
python experiment.py --env hopper --dataset medium-replay --model_type dt --seed 42 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "checkpoints/dt_hopper_medium-replay_size_2_42" &
python experiment.py --env hopper --dataset medium --model_type dt --seed 42 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "checkpoints/dt_hopper_medium_size_2_42" &
wait
python experiment.py --env walker2d --dataset medium-expert --model_type dt --seed 42 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "checkpoints/dt_walker2d_medium-expert_size_2_42" &
python experiment.py --env walker2d --dataset medium-replay --model_type dt --seed 42 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "checkpoints/dt_walker2d_medium-replay_size_2_42" &
python experiment.py --env walker2d --dataset medium --model_type dt --seed 42 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "checkpoints/dt_walker2d_medium_size_2_42" &
wait
python experiment.py --env halfcheetah --dataset medium-expert --model_type dt --seed 42 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "checkpoints/dt_halfcheetah_medium-expert_size_2_42" &
python experiment.py --env halfcheetah --dataset medium-replay --model_type dt --seed 42 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "checkpoints/dt_halfcheetah_medium-replay_size_2_42" &
python experiment.py --env halfcheetah --dataset medium --model_type dt --seed 42 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "checkpoints/dt_halfcheetah_medium_size_2_42" &
wait
python experiment_new.py --env helfcheetah --dataset medium-replay --model_type dt --seed 42 --data_size 0.75 --outdir "checkpoints/dt_helfcheetah_medium_replay_finetune_75_42" &
python experiment_new.py --env helfcheetah --dataset medium-expert --model_type dt --seed 42 --data_size 0.75 --outdir "checkpoints/dt_helfcheetah_medium_expert_finetune_75_42" &
python experiment_new.py --env helfcheetah --dataset medium --model_type dt --seed 42 --data_size 0.75 --outdir "checkpoints/dt_helfcheetah_medium_finetune_75_42" &
wait
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 42 --data_size 0.75 --outdir "checkpoints/dt_hopper_medium_replay_finetune_75_42" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 42 --data_size 0.75 --outdir "checkpoints/dt_hopper_medium_expert_finetune_75_42" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 42 --data_size 0.75 --outdir "checkpoints/dt_hopper_medium_finetune_75_42" &
wait
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 42 --data_size 0.75 --outdir "checkpoints/dt_walker_medium_replay_finetune_75_42" &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 42 --data_size 0.75 --outdir "checkpoints/dt_walker_medium_expert_finetune_75_42" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 42 --data_size 0.75 --outdir "checkpoints/dt_walker_medium_finetune_75_42" &
wait
python experiment_new.py --env helfcheetah --dataset medium-replay --model_type dt --seed 42 --data_size 0.5 --outdir "checkpoints/dt_helfcheetah_medium_replay_finetune_50_42" &
python experiment_new.py --env helfcheetah --dataset medium-expert --model_type dt --seed 42 --data_size 0.5 --outdir "checkpoints/dt_helfcheetah_medium_expert_finetune_50_42" &
python experiment_new.py --env helfcheetah --dataset medium --model_type dt --seed 42 --data_size 0.5 --outdir "checkpoints/dt_helfcheetah_medium_finetune_50_42" &
wait
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 42 --data_size 0.5 --outdir "checkpoints/dt_hopper_medium_replay_finetune_50_42" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 42 --data_size 0.5 --outdir "checkpoints/dt_hopper_medium_expert_finetune_50_42" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 42 --data_size 0.5 --outdir "checkpoints/dt_hopper_medium_finetune_50_42" &
wait
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 42 --data_size 0.5 --outdir "checkpoints/dt_walker_medium_replay_finetune_50_42" &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 42 --data_size 0.5 --outdir "checkpoints/dt_walker_medium_expert_finetune_50_42" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 42 --data_size 0.5 --outdir "checkpoints/dt_walker_medium_finetune_50_42" &
wait

python experiment_new.py --env helfcheetah --dataset medium-replay --model_type dt --seed 42 --data_size 0.25 --outdir "checkpoints/dt_helfcheetah_medium_replay_finetune_25_42" &
python experiment_new.py --env helfcheetah --dataset medium-expert --model_type dt --seed 42 --data_size 0.25 --outdir "checkpoints/dt_helfcheetah_medium_expert_finetune_25_42" &
python experiment_new.py --env helfcheetah --dataset medium --model_type dt --seed 42 --data_size 0.25 --outdir "checkpoints/dt_helfcheetah_medium_finetune_25_42" &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 42 --data_size 0.25 --outdir "checkpoints/dt_hopper_medium_replay_finetune_25_42" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 42 --data_size 0.25 --outdir "checkpoints/dt_hopper_medium_expert_finetune_25_42" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 42 --data_size 0.25 --outdir "checkpoints/dt_hopper_medium_finetune_25_42" &
wait
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 42 --data_size 0.25 --outdir "checkpoints/dt_walker_medium_replay_finetune_25_42" &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 42 --data_size 0.25 --outdir "checkpoints/dt_walker_medium_expert_finetune_25_42" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 42 --data_size 0.25 --outdir "checkpoints/dt_walker_medium_finetune_25_42" &
python experiment_new.py --env helfcheetah --dataset medium-replay --model_type dt --seed 42 --data_size 0.1 --outdir "checkpoints/dt_helfcheetah_medium_replay_finetune_10_42" &
python experiment_new.py --env helfcheetah --dataset medium-expert --model_type dt --seed 42 --data_size 0.1 --outdir "checkpoints/dt_helfcheetah_medium_expert_finetune_10_42" &
python experiment_new.py --env helfcheetah --dataset medium --model_type dt --seed 42 --data_size 0.1 --outdir "checkpoints/dt_helfcheetah_medium_finetune_10_42" &
wait
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 42 --data_size 0.1 --outdir "checkpoints/dt_hopper_medium_replay_finetune_10_42" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 42 --data_size 0.1 --outdir "checkpoints/dt_hopper_medium_expert_finetune_10_42" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 42 --data_size 0.1 --outdir "checkpoints/dt_hopper_medium_finetune_10_42" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 42 --data_size 0.1 --outdir "checkpoints/dt_walker_medium_replay_finetune_10_42" &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 42 --data_size 0.1 --outdir "checkpoints/dt_walker_medium_expert_finetune_10_42" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 42 --data_size 0.1 --outdir "checkpoints/dt_walker_medium_finetune_10_42" &
wait
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 666 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_halfcheetah_medium-replay_size_3_666" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 42 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_halfcheetah_medium-replay_size_3_42" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 1024 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_halfcheetah_medium-replay_size_3_1024" &
wait

python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 666 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_walker_medium-replay_size_3_666" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 42 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_walker_medium-replay_size_3_42" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 1024 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_walker_medium-replay_size_3_1024" &
wait
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 666 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_walker_medium_size_3_666" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 42 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_walker_medium_size_3_42" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 1024 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_walker_medium_size_3_1024" &
wait

python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 666 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_hopper_medium-replay_size_3_666" &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 42 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_hopper_medium-replay_size_3_42" &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 1024 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_hopper_medium-replay_size_3_1024" &
wait
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 666 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_hopper_medium_size_3_666" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 42 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_hopper_medium_size_3_42" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 1024 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "checkpoints/dt_hopper_medium_size_3_1024" &
wait

#test
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 1024 --data_size 0.1 --max_iters 2 --num_steps_per_iter 10 --outdir "dt_debug_finetune_10" --device cpu


##new setting reruns
python experiment_new.py --env helfcheetah --dataset medium-replay --model_type dt --seed 42 --data_size 0.25 --outdir "dt_data_size0.25" &
python experiment_new.py --env helfcheetah --dataset medium-expert --model_type dt --seed 42 --data_size 0.25 --outdir "dt_data_size0.25" &
python experiment_new.py --env helfcheetah --dataset medium --model_type dt --seed 42 --data_size 0.25 --outdir "dt_data_size0.25" &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 42 --data_size 0.25 --outdir "dt_data_size0.25" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 42 --data_size 0.25 --outdir "dt_data_size0.25" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 42 --data_size 0.25 --outdir "dt_data_size0.25" &
wait
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 42 --data_size 0.25 --outdir "dt_data_size0.25" &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 42 --data_size 0.25 --outdir "dt_data_size0.25" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 42 --data_size 0.25 --outdir "dt_data_size0.25" &
python experiment_new.py --env helfcheetah --dataset medium-replay --model_type dt --seed 42 --data_size 0.1 --outdir "dt_data_size0.1" &
python experiment_new.py --env helfcheetah --dataset medium-expert --model_type dt --seed 42 --data_size 0.1 --outdir "dt_data_size0.1" &
python experiment_new.py --env helfcheetah --dataset medium --model_type dt --seed 42 --data_size 0.1 --outdir "dt_data_size0.1" &
wait
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 42 --data_size 0.1 --outdir "dt_data_size0.1" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 42 --data_size 0.1 --outdir "dt_data_size0.1" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 42 --data_size 0.1 --outdir "dt_data_size0.1" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 42 --data_size 0.1 --outdir "dt_data_size0.1" &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 42 --data_size 0.1 --outdir "dt_data_size0.1" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 42 --data_size 0.1 --outdir "dt_data_size0.1" &
wait
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 666 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 42 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 1024 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
wait

python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 666 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "dt_embed_dim512_n_layer6_n_head8" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 42 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "dt_embed_dim512_n_layer6_n_head8" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 1024 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "dt_embed_dim512_n_layer6_n_head8" &
wait
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 666 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "dt_embed_dim512_n_layer6_n_head8" &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 42 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "dt_embed_dim512_n_layer6_n_head8" &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 1024 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "dt_embed_dim512_n_layer6_n_head8" &
wait
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 666 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "dt_embed_dim512_n_layer6_n_head8" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 42 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "dt_embed_dim512_n_layer6_n_head8" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 1024 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "dt_embed_dim512_n_layer6_n_head8" &
wait
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 666 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "dt_embed_dim512_n_layer6_n_head8" &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 42 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "dt_embed_dim512_n_layer6_n_head8" &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 1024 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "dt_embed_dim512_n_layer6_n_head8" &
wait
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 666 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "dt_embed_dim512_n_layer6_n_head8" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 42 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "dt_embed_dim512_n_layer6_n_head8" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 1024 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "dt_embed_dim512_n_layer6_n_head8" &
wait
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 666 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "dt_embed_dim512_n_layer6_n_head8" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 42 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "dt_embed_dim512_n_layer6_n_head8" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 1024 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "dt_embed_dim512_n_layer6_n_head8" &
wait

python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 666 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "dt_embed_dim512_n_layer6_n_head8" &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 42 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "dt_embed_dim512_n_layer6_n_head8" &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 1024 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "dt_embed_dim512_n_layer6_n_head8" &
wait
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 666 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "dt_embed_dim512_n_layer6_n_head8" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 42 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "dt_embed_dim512_n_layer6_n_head8" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 1024 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "dt_embed_dim512_n_layer6_n_head8" &
wait
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 666 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "dt_embed_dim512_n_layer6_n_head8" &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 42 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "dt_embed_dim512_n_layer6_n_head8" &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 1024 --embed_dim 512 --n_layer 6 --n_head 8 --outdir "dt_embed_dim512_n_layer6_n_head8" &
wait
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 666 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "dt_embed_dim256_n_layer4_n_head4" &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 42 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "dt_embed_dim256_n_layer4_n_head4" &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 1024 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "dt_embed_dim256_n_layer4_n_head4" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 666 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "dt_embed_dim256_n_layer4_n_head4" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 42 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "dt_embed_dim256_n_layer4_n_head4" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 1024 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "dt_embed_dim256_n_layer4_n_head4" &
wait
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 666 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "dt_embed_dim256_n_layer4_n_head4" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 42 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "dt_embed_dim256_n_layer4_n_head4" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 1024 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "dt_embed_dim256_n_layer4_n_head4" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 666 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "dt_embed_dim256_n_layer4_n_head4" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 42 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "dt_embed_dim256_n_layer4_n_head4" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 1024 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "dt_embed_dim256_n_layer4_n_head4" &
wait
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 666 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "dt_embed_dim256_n_layer4_n_head4" &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 42 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "dt_embed_dim256_n_layer4_n_head4" &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 1024 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "dt_embed_dim256_n_layer4_n_head4" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 666 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "dt_embed_dim256_n_layer4_n_head4" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 42 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "dt_embed_dim256_n_layer4_n_head4" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 1024 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "dt_embed_dim256_n_layer4_n_head4" &
wait
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 666 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "dt_embed_dim256_n_layer4_n_head4" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 42 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "dt_embed_dim256_n_layer4_n_head4" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 1024 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "dt_embed_dim256_n_layer4_n_head4" &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 666 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "dt_embed_dim256_n_layer4_n_head4" &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 42 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "dt_embed_dim256_n_layer4_n_head4" &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 1024 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "dt_embed_dim256_n_layer4_n_head4" &
wait
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 666 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "dt_embed_dim256_n_layer4_n_head4" &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 42 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "dt_embed_dim256_n_layer4_n_head4" &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 1024 --embed_dim 256 --n_layer 4 --n_head 4 --outdir "dt_embed_dim256_n_layer4_n_head4" &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 666 --outdir "dt" &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 42 --outdir "dt" &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 1024 --outdir "dt" &
wait
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 666 --outdir "dt" &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 42 --outdir "dt" &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 1024 --outdir "dt" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 666 --outdir "dt" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 42 --outdir "dt" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 1024 --outdir "dt" &
wait
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 666 --outdir "dt" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 42 --outdir "dt" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 1024 --outdir "dt" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 666 --outdir "dt" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 42 --outdir "dt" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 1024 --outdir "dt" &
wait
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 666 --outdir "dt" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 42 --outdir "dt" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 1024 --outdir "dt" &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 666 --outdir "dt" &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 42 --outdir "dt" &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 1024 --outdir "dt" &
wait
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 666 --outdir "dt" &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 42 --outdir "dt" &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 1024 --outdir "dt" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 666 --outdir "dt" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 42 --outdir "dt" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 1024 --outdir "dt" &
wait

python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 666 --data_size 0.1 --outdir "dt_data_size0.1" &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 42 --data_size 0.1 --outdir "dt_data_size0.1" &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 1024 --data_size 0.1 --outdir "dt_data_size0.1" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 666 --data_size 0.1 --outdir "dt_data_size0.1" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 42 --data_size 0.1 --outdir "dt_data_size0.1" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 1024 --data_size 0.1 --outdir "dt_data_size0.1" &
wait
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 666 --data_size 0.1 --outdir "dt_data_size0.1" &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 42 --data_size 0.1 --outdir "dt_data_size0.1" &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 1024 --data_size 0.1 --outdir "dt_data_size0.1" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 666 --data_size 0.1 --outdir "dt_data_size0.1" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 42 --data_size 0.1 --outdir "dt_data_size0.1" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 1024 --data_size 0.1 --outdir "dt_data_size0.1" &
wait
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 666 --data_size 0.1 --outdir "dt_data_size0.1" &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 42 --data_size 0.1 --outdir "dt_data_size0.1" &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 1024 --data_size 0.1 --outdir "dt_data_size0.1" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 666 --data_size 0.1 --outdir "dt_data_size0.1" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 42 --data_size 0.1 --outdir "dt_data_size0.1" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 1024 --data_size 0.1 --outdir "dt_data_size0.1" &
wait
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 666 --data_size 0.1 --outdir "dt_data_size0.1" &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 42 --data_size 0.1 --outdir "dt_data_size0.1" &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 1024 --data_size 0.1 --outdir "dt_data_size0.1" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 666 --data_size 0.1 --outdir "dt_data_size0.1" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 42 --data_size 0.1 --outdir "dt_data_size0.1" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 1024 --data_size 0.1 --outdir "dt_data_size0.1" &
wait
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 666 --data_size 0.1 --outdir "dt_data_size0.1" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 42 --data_size 0.1 --outdir "dt_data_size0.1" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 1024 --data_size 0.1 --outdir "dt_data_size0.1" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 666 --data_size 0.25 --outdir "dt_data_size0.25" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 42 --data_size 0.25 --outdir "dt_data_size0.25" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 1024 --data_size 0.25 --outdir "dt_data_size0.25" &
wait
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 666 --data_size 0.25 --outdir "dt_data_size0.25" &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 42 --data_size 0.25 --outdir "dt_data_size0.25" &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 1024 --data_size 0.25 --outdir "dt_data_size0.25" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 666 --data_size 0.25 --outdir "dt_data_size0.25" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 42 --data_size 0.25 --outdir "dt_data_size0.25" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 1024 --data_size 0.25 --outdir "dt_data_size0.25" &
wait
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 666 --data_size 0.25 --outdir "dt_data_size0.25" &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 42 --data_size 0.25 --outdir "dt_data_size0.25" &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 1024 --data_size 0.25 --outdir "dt_data_size0.25" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 666 --data_size 0.25 --outdir "dt_data_size0.25" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 42 --data_size 0.25 --outdir "dt_data_size0.25" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 1024 --data_size 0.25 --outdir "dt_data_size0.25" &
wait
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 666 --data_size 0.25 --outdir "dt_data_size0.25" &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 42 --data_size 0.25 --outdir "dt_data_size0.25" &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 1024 --data_size 0.25 --outdir "dt_data_size0.25" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 666 --data_size 0.25 --outdir "dt_data_size0.25" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 42 --data_size 0.25 --outdir "dt_data_size0.25" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 1024 --data_size 0.25 --outdir "dt_data_size0.25" &
wait
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 666 --data_size 0.25 --outdir "dt_data_size0.25" &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 42 --data_size 0.25 --outdir "dt_data_size0.25" &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 1024 --data_size 0.25 --outdir "dt_data_size0.25" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 666 --data_size 0.25 --outdir "dt_data_size0.25" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 42 --data_size 0.25 --outdir "dt_data_size0.25" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 1024 --data_size 0.25 --outdir "dt_data_size0.25" &
wait
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 666 --data_size 0.5 --outdir "dt_data_size0.5" &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 42 --data_size 0.5 --outdir "dt_data_size0.5" &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 1024 --data_size 0.5 --outdir "dt_data_size0.5" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 666 --data_size 0.5 --outdir "dt_data_size0.5" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 42 --data_size 0.5 --outdir "dt_data_size0.5" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 1024 --data_size 0.5 --outdir "dt_data_size0.5" &
wait
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 666 --data_size 0.5 --outdir "dt_data_size0.5" &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 42 --data_size 0.5 --outdir "dt_data_size0.5" &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 1024 --data_size 0.5 --outdir "dt_data_size0.5" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 666 --data_size 0.5 --outdir "dt_data_size0.5" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 42 --data_size 0.5 --outdir "dt_data_size0.5" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 1024 --data_size 0.5 --outdir "dt_data_size0.5" &
wait
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 666 --data_size 0.5 --outdir "dt_data_size0.5" &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 42 --data_size 0.5 --outdir "dt_data_size0.5" &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 1024 --data_size 0.5 --outdir "dt_data_size0.5" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 666 --data_size 0.5 --outdir "dt_data_size0.5" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 42 --data_size 0.5 --outdir "dt_data_size0.5" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 1024 --data_size 0.5 --outdir "dt_data_size0.5" &
wait
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 666 --data_size 0.5 --outdir "dt_data_size0.5" &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 42 --data_size 0.5 --outdir "dt_data_size0.5" &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 1024 --data_size 0.5 --outdir "dt_data_size0.5" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 666 --data_size 0.5 --outdir "dt_data_size0.5" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 42 --data_size 0.5 --outdir "dt_data_size0.5" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 1024 --data_size 0.5 --outdir "dt_data_size0.5" &
wait
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 666 --data_size 0.5 --outdir "dt_data_size0.5" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 42 --data_size 0.5 --outdir "dt_data_size0.5" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 1024 --data_size 0.5 --outdir "dt_data_size0.5" &
wait

python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 666 --data_size 0.75 --outdir "dt_data_size0.75" &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 42 --data_size 0.75 --outdir "dt_data_size0.75" &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 1024 --data_size 0.75 --outdir "dt_data_size0.75" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 666 --data_size 0.75 --outdir "dt_data_size0.75" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 42 --data_size 0.75 --outdir "dt_data_size0.75" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 1024 --data_size 0.75 --outdir "dt_data_size0.75" &
wait
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 666 --data_size 0.75 --outdir "dt_data_size0.75" &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 42 --data_size 0.75 --outdir "dt_data_size0.75" &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 1024 --data_size 0.75 --outdir "dt_data_size0.75" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 666 --data_size 0.75 --outdir "dt_data_size0.75" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 42 --data_size 0.75 --outdir "dt_data_size0.75" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 1024 --data_size 0.75 --outdir "dt_data_size0.75" &
wait
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 666 --data_size 0.75 --outdir "dt_data_size0.75" &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 42 --data_size 0.75 --outdir "dt_data_size0.75" &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 1024 --data_size 0.75 --outdir "dt_data_size0.75" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 666 --data_size 0.75 --outdir "dt_data_size0.75" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 42 --data_size 0.75 --outdir "dt_data_size0.75" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 1024 --data_size 0.75 --outdir "dt_data_size0.75" &
wait
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 666 --data_size 0.75 --outdir "dt_data_size0.75" &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 42 --data_size 0.75 --outdir "dt_data_size0.75" &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 1024 --data_size 0.75 --outdir "dt_data_size0.75" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 666 --data_size 0.75 --outdir "dt_data_size0.75" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 42 --data_size 0.75 --outdir "dt_data_size0.75" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 1024 --data_size 0.75 --outdir "dt_data_size0.75" &
wait
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 666 --data_size 0.75 --outdir "dt_data_size0.75" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 42 --data_size 0.75 --outdir "dt_data_size0.75" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 1024 --data_size 0.75 --outdir "dt_data_size0.75" &
wait
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 666 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 42 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 1024 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
wait


python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 666 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 42 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 1024 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
wait
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 666 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 42 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 1024 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
wait

python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 666 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 42 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 1024 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
wait
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 666 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 42 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 1024 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
wait

python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 666 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 42 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 1024 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
wait
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 666 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 42 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 1024 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
wait

python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 666 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 42 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 1024 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
wait
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 666 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 42 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 1024 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
wait

#chibiT

python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1  --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait

python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25  --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5  --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait

python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer4e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 4e0 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer4e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 4e0 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer4e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 4e0 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer4e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 4e0 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer4e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 4e0 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer4e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 4e0 --perturb_transformer_only &
wait
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer4e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 4e0 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer4e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 4e0 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer4e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 4e0 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer4e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 4e0 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer4e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 4e0 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer4e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 4e0 --perturb_transformer_only &
wait
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer4e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 4e0 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer4e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 4e0 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer4e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 4e0 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer4e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 4e0 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer4e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 4e0 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer4e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 4e0 --perturb_transformer_only &
wait
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer4e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 4e0 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer4e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 4e0 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer4e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 4e0 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer4e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 4e0 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer4e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 4e0 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer4e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 4e0 --perturb_transformer_only &
wait
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer4e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 4e0 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer4e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 4e0 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer4e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 4e0 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
wait
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
wait
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
wait
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
wait
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
wait

##rerun
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1  --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait

python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.1" --data_size 0.1 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25  --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait

python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.25" --data_size 0.25 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5  --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait

python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.5" --data_size 0.5 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_data_size0.75" --data_size 0.75 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_data_size0.75" --data_size 0.75 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_data_size0.75" --data_size 0.75 --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj &
wait
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer1e-1" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 1e-1 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer1e-1" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 1e-1 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer1e-1" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 1e-1 --perturb_transformer_only &
wait
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer1e-1" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 1e-1 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer1e-1" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 1e-1 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer1e-1" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 1e-1 --perturb_transformer_only &
wait
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer1e-1" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 1e-1 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer1e-1" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 1e-1 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer1e-1" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 1e-1 --perturb_transformer_only &
wait
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer1e-1" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 1e-1 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer1e-1" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 1e-1 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer1e-1" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 1e-1 --perturb_transformer_only &
wait
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer1e-1" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 1e-1 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer1e-1" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 1e-1 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer1e-1" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 1e-1 --perturb_transformer_only &
wait

python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer1e-1" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 1e-1 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer1e-1" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 1e-1 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer1e-1" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 1e-1 --perturb_transformer_only &
wait
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer1e-1" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 1e-1 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer1e-1" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 1e-1 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer1e-1" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 1e-1 --perturb_transformer_only &
wait
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer1e-1" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 1e-1 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer1e-1" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 1e-1 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer1e-1" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 1e-1 --perturb_transformer_only &
wait
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer1e-1" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 1e-1 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer1e-1" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 1e-1 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer1e-1" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 1e-1 --perturb_transformer_only &
wait
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer1e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 1e0 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer1e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 1e0 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer1e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 1e0 --perturb_transformer_only &
wait
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer2e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 2e0 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer2e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 2e0 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer2e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 2e0 --perturb_transformer_only &
wait
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer2e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 2e0 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer2e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 2e0 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer2e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 2e0 --perturb_transformer_only &
wait
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer2e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 2e0 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer2e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 2e0 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer2e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 2e0 --perturb_transformer_only &
wait

python experiment_new.py --env hopper --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer2e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 2e0 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer2e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 2e0 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer2e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 2e0 --perturb_transformer_only &
wait
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer2e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 2e0 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer2e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 2e0 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer2e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 2e0 --perturb_transformer_only &
wait
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer2e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 2e0 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer2e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 2e0 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer2e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 2e0 --perturb_transformer_only &
wait
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer2e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 2e0 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer2e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 2e0 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer2e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 2e0 --perturb_transformer_only &
wait
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer2e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 2e0 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer2e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 2e0 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer2e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 2e0 --perturb_transformer_only &
wait
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer2e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 2e0 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer2e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 2e0 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer2e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 2e0 --perturb_transformer_only &
wait
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer4e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 4e0 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer4e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 4e0 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer4e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 4e0 --perturb_transformer_only &
wait
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
wait

python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
wait
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
wait
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
wait
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
wait
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
wait
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env hopper --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
wait
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium-expert --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
wait
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 666 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 42 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
python experiment_new.py --env walker2d --dataset medium-replay --model_type dt --seed 1024 --pretrained_lm chibiT --outdir "chibiT_perturb_per_layer8e0" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0 --perturb_transformer_only &
wait

python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 666 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 42 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
python experiment_new.py --env halfcheetah --dataset medium-replay --model_type dt --seed 1024 --embed_dim 768 --n_layer 12 --n_head 12 --outdir "dt_embed_dim768_n_layer12_n_head12" &
wait
