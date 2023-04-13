import torch
from decision_transformer.models.decision_transformer import DecisionTransformer
import argparse


def main(args):
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

    if variant["load_checkpoint"]:
        state_dict = torch.load(variant["load_checkpoint"])
        model.load_state_dict(state_dict)
        print(f"Loaded from {variant['load_checkpoint']}")

    check_std(model)

def check_std(model):
    for p in model.parameters():
        torch.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--perturb", action="store_true", default=False)
    parser.add_argument("--perturb_per_layer", type=float, default=1.)
    parser.add_argument("--perturb_absolute", type=float, default=1.)
    args = parser.parse_args()


