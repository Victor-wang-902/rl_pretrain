import torch
from decision_transformer.models.decision_transformer import DecisionTransformer
import argparse
import gym


def main(variant):
    K = variant["K"]
    env_name = variant["env"]
    if env_name == "hopper":
        env = gym.make("Hopper-v3")
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.0  # normalization for rewards/returns
    elif env_name == "halfcheetah":
        env = gym.make("HalfCheetah-v3")
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.0
    elif env_name == "walker2d":
        env = gym.make("Walker2d-v3")
        max_ep_len = 1000
        env_targets = [5000, 2500]
        scale = 1000.0
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]


    model = DecisionTransformer(
                args=variant,
                state_dim=state_dim,
                act_dim=act_dim,
                max_length=K,
                max_ep_len=max_ep_len,
                hidden_size=variant["embed_dim"],
                n_layer=variant["n_layer"],
                n_head=variant["n_head"],
                n_inner=4 * variant["embed_dim"],
                activation_function=variant["activation_function"],
                n_positions=1024,
                resid_pdrop=variant["dropout"],
                attn_pdrop=0.1,
            )
    
    param = sum(p.numel() for p in model.parameters())
    print(param)
    
'''
    if variant["load_checkpoint"]:
        state_dict = torch.load(variant["load_checkpoint"])
        model.load_state_dict(state_dict)
        print(f"Loaded from {variant['load_checkpoint']}")
'''

    #check_std(model)

def check_std(model):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="hopper")
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=1)

    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--perturb", action="store_true", default=False)
    parser.add_argument("--perturb_per_layer", type=float, default=1.)
    parser.add_argument("--perturb_absolute", type=float, default=1.)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--pretrained_lm", type=str, default=None)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--frozen", action="store_true", default=False)
    parser.add_argument("--extend_positions", action="store_true", default=False)
    parser.add_argument("--share_input_output_proj", action="store_true", default=False)


    args = parser.parse_args()
    main(vars(args))

#python calculate_model_std.py --env hopper --embed_dim 128 --n_layer 3 --n_head 1

