from transformers import GPT2Config
from tqdm import tqdm
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from data import get_dataloader, GPT2Dataset, GPT2Collator, worker_init_fn
import math
import csv
from datasets import load_dataset
import time
import torch
import argparse
import os

##########simple one GPU pretraining#########

@torch.no_grad()
def eval(args, dataloader, model):
    model.eval()
    cumulative_loss = 0.
    for step, batch in tqdm(enumerate(dataloader)):
        inputs = batch['input_ids']
        labels = batch["input_ids"].detach().clone().long()
        attn_msk = batch["attention_mask"]
        outputs = model(inputs, attention_mask=attn_msk, labels=labels)  # fix here
        loss = outputs.loss
        cumulative_loss += loss.detach().cpu().item()
    return cumulative_loss
    

def main(args):  # add argparser
    device = torch.device(args.device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_embd=args.embed_dim,
        n_layer=args.n_layer,
        n_head=args.n_head,
        activation_function=args.activation_function,
        n_positions=1024,
        resid_pdrop=args.dropout,
    )

    model = GPT2LMHeadModel(config)
    model.to(device)
    if args.load_checkpoint:
        state_dict = torch.load(args.load_checkpoint)
        model.load_state_dict(state_dict)
        print(f"Loaded from {variant['load_checkpoint']}")

    data = load_dataset(args.dataset, args.subsplit)
    
    test_data = data["test"]
    valid_data = data["validation"]

    valid_dataset = GPT2Dataset(valid_data, tokenizer, split="valid")
    test_dataset = GPT2Dataset(test_data, tokenizer, split="test")

    collator = GPT2Collator(device)

    batch_size = args.batch_size // 1024
    num_steps = args.num_steps
    num_warmup_steps = args.warmup_steps
    num_tokens = args.batch_size - args.batch_size // 1024

    test_data_loader = get_dataloader(
        test_dataset,
        collator,
        batch_size=batch_size,  #fix here
        worker_init_fn=worker_init_fn,
        num_workers=args.num_workers,
        drop_last=False
    )

    valid_data_loader = get_dataloader(
        valid_dataset,
        collator,
        batch_size=batch_size,  #fix here
        worker_init_fn=worker_init_fn,
        num_workers=args.num_workers,
        drop_last=False
    )

    with open(os.path.join(args.outdir, "progress.csv"), "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["current_train_time", "current_eval_time", "total_time", "steps", "train_loss", "train_ppl", "eval_ppl"])

    if not args.eval_only:
        train_data = data["train"]
        train_dataset = GPT2Dataset(train_data, tokenizer, seed=args.seed, data_size=args.data_size)

        train_data_loader = get_dataloader(
            train_dataset,
            collator,
            batch_size=batch_size,  #fix here
            worker_init_fn=worker_init_fn,
            num_workers=args.num_workers,
            drop_last = True
        )
        
        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps,
            num_steps
        )
        total_start_time = time.time()
        cur_start_time = time.time()
        cumulative_loss = 0.
        for step, batch in tqdm(enumerate(train_data_loader), total=num_steps):
            inputs = batch['input_ids']
            labels = batch["input_ids"].detach().clone().long()
            attn_msk = batch["attention_mask"]
            model.train()
            outputs = model(inputs, attention_mask=attn_msk, labels=labels)  # fix here
            optimizer.zero_grad()  
            loss = outputs.loss
            loss.backward()
            cumulative_loss += loss.detach().cpu().item()
            optimizer.step()
            scheduler.step()
            if (step + 1) % args.num_steps_per_save == 0:
                train_ppl = math.exp(cumulative_loss)
                cur_train_end_time = time.time()
                valid_loss = eval(args, valid_data_loader, model)
                valid_ppl = math.exp(valid_loss)
                cur_eval_end_time = time.time()
                print(f'Step {step + 1}/{num_steps}, train loss = {cumulative_loss}, train ppl = {train_ppl}, valid ppl = {valid_ppl}')
                if args.outdir is not None:
                    torch.save(
                        model.state_dict(),
                        f"{args.outdir}/model_{step + 1}.pt",
                    )
                cur_total_time = time.time()
                with open(os.path.join(args.outdir, "progress.csv"), "a") as f:
                    writer = csv.writer(f, delimiter="\t")
                    writer.writerow([cur_train_end_time - cur_start_time, cur_eval_end_time - cur_train_end_time, cur_total_time - total_start_time, step + 1, cumulative_loss, train_ppl, valid_ppl])
                cur_start_time = time.time()
                cumulative_loss = 0.
    test_start_time = time.time()
    test_loss = eval(args, test_data_loader, model)
    test_ppl = math.exp(test_loss)
    test_end_time = time.time()
    print(f'Training complete, final test ppl = {test_ppl}')
    with open(os.path.join(args.outdir, "progress.csv"), "a") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([0., test_end_time - test_start_time, test_end_time - total_start_time, "test", 0., 0., test_ppl])






def set_dt_args(args_to_parse=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="wikitext"
    )  # medium, medium-replay, medium-expert, expert
    parser.add_argument(
        "--subsplit", type=str, default="wikitext-103-raw-v1"
    )  # normal for standard setting, delayed for sparse
    parser.add_argument("--batch_size", type=int, default=65536)
    parser.add_argument(
        "--model_type", type=str, default="dt"
    )  # dt for decision transformer, bc for behavior cloning
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--load_checkpoint", type=str, default=None)

    parser.add_argument("--learning_rate", "-lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-2)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--num_steps", type=int, default=80000)
    parser.add_argument("--num_steps_per_save", type=int, default=10000)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_only", action="store_true", default=False)

    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--output_fname", type=str, default="progress.csv")

    parser.add_argument("--data_size", type=float, default=1.0)

    if args_to_parse is not None:
        args = parser.parse_args(args_to_parse)
    else:
        args = parser.parse_args()

    return args

if __name__ == "__main__":
    start_time = time.time()
    args = set_dt_args()
    data_dir = 'pretrain/checkpoints'
    exp_name_full = args.outdir
    args.outdir = os.path.join(data_dir, exp_name_full)
    os.makedirs(args.outdir, exist_ok=True)
    if args.load_checkpoint is not None:
        args.load_checkpoint = os.path.join(data_dir, args.load_checkpoint)
    main(args)
    print("Total time used: %.3f hours." % ((time.time() - start_time)/3600))

