## Example commands
#  run DT on walker2d
python experiment.py --env walker2d --dataset medium --model_type dt --seed 666    --outdir "checkpoints/dt_kmeans_medium_positions_walker2d_paper_666" 
python experiment.py --env walker2d --dataset medium --model_type dt --seed 42  --outdir "checkpoints/dt_kmeans_medium_positions_walker2d_paper_42"  
python experiment.py --env walker2d --dataset medium --model_type dt --seed 1024  --outdir "checkpoints/dt_kmeans_medium_positions_walker2d_paper_1024"  

# run ChibiT on hopper medium-expert
python experiment.py --env hopper --dataset medium-expert --model_type dt --seed 666  --pretrained_lm chibiT  --outdir "checkpoints/cibiT_kmeans_medium_expert_positions_hopper_paper_666" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj
python experiment.py --env hopper --dataset medium-expert --model_type dt --seed 42  --pretrained_lm chibiT  --outdir "checkpoints/cibiT_kmeans_medium_expert_positions_hopper_paper_42" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj
python experiment.py --env hopper --dataset medium-expert --model_type dt --seed 1024  --pretrained_lm chibiT  --outdir "checkpoints/cibiT_kmeans_medium_expert_positions_hopper_paper_1024" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj

# run GPT2 on reacher2d medium
python experiment.py --env reacher2d --dataset medium --model_type dt --seed 1024 --pretrained_lm gpt2  --outdir "checkpoints/gpt2_kmeans_medium_positions_reacher2d_1024" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/gpt2_lm_1000.pt" --gpt_kmeans_const 0.1 --dropout 0.2 --share_input_output_proj
python experiment.py --env reacher2d --dataset medium --model_type dt --seed 666  --pretrained_lm gpt2  --outdir "checkpoints/gpt2_kmeans_medium_positions_reacher2d_666" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/gpt2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj
python experiment.py --env reacher2d --dataset medium --model_type dt --seed 42  --pretrained_lm gpt2  --outdir "checkpoints/gpt2_kmeans_medium_positions_reacher2d_42" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/gpt2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj

# run perturbation per layer on hopper medium
python experiment.py --env hopper --dataset medium --model_type dt --seed 666  --pretrained_lm chibiT  --outdir "checkpoints/cibiT_kmeans_medium_positions_hopper_perturb_8e0_666" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0
python experiment.py --env hopper --dataset medium --model_type dt --seed 42  --pretrained_lm chibiT  --outdir "checkpoints/cibiT_kmeans_medium_positions_hopper_perturb_8e0_42" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0
python experiment.py --env hopper --dataset medium --model_type dt --seed 1024  --pretrained_lm chibiT  --outdir "checkpoints/cibiT_kmeans_medium_positions_hopper_perturb_8e0_1024" --extend_positions --gpt_kmeans 1000 --kmeans_cache "kmeans_cache/chibiv2_lm_1000.pt" --gpt_kmeans_const 0.1  --dropout 0.2 --share_input_output_proj --perturb --perturb_per_layer 8e0

