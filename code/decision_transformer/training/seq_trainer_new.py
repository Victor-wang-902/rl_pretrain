import numpy as np
import torch
import torch.nn.functional as F

from decision_transformer.training.trainer_new import Trainer
import torch_semiring_einsum
import time
import tqdm
EQUATION = torch_semiring_einsum.compile_equation("iaj,bj->iabj")


def kmeans_cosine_max_loss(centers, seq, mean=False):
    assert centers.device == seq.device
    # loss = -(torch.einsum("iaj,bj->iabj", [seq, centers]).max(2).values.mean())
    if mean:
        loss = -(
            torch_semiring_einsum.einsum(EQUATION, seq, centers, block_size=5).mean()
        )
    else:
        loss = -(
            torch_semiring_einsum.einsum(EQUATION, seq, centers, block_size=5)
            .max(2)
            .values.mean()
        )


    return loss


kmeans_anneal = lambda x: 1 / (1 + np.exp(-(((5000 - x) / (5000 / 10)) - 5)))


class SequenceTrainer(Trainer):
    def train_step(self):
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            timesteps,
            attention_mask,
        ) = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds, all_embs = self.model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            attention_mask=attention_mask,
        )

        self.step += 1
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[
            attention_mask.reshape(-1) > 0
        ]

        loss = self.loss_fn(
            None,
            action_preds,
            None,
            None,
            action_target,
            None,
        )
        if self.args["gpt_kmeans"]:
            loss += (
                self.args["gpt_kmeans_const"]
                * kmeans_anneal(self.step)
                * kmeans_cosine_max_loss(
                    F.normalize(self.model.cluster_centers, dim=-1),
                    F.normalize(all_embs, dim=-1),
                    mean=self.args["kmeans_mean"],
                )
            )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics["training/action_error"] = (
                torch.mean((action_preds - action_target) ** 2).detach().cpu().item()
            )

        return loss.detach().cpu().item()


class StateTrainer(Trainer):
    def __init__(
        self,
        args,
        model,
        optimizer,
        batch_size,
        get_batch,
        loss_fn,
        scheduler=None,
        eval_fns=None,
        eval_only=False,
        device=None
    ):
        super().__init__(args, model, optimizer, batch_size, get_batch, loss_fn, scheduler, None, False)
        #self.return_placeholder = return_placeholder
        self.device = device

    def train_step(self):
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            timesteps,
            attention_mask,
        ) = self.get_batch(self.batch_size)
        #print(states.shape)
        rewards = torch.zeros_like(rewards).to(self.device)
        rtg = torch.zeros_like(rtg).to(self.device)
        state_target = torch.clone(states)
        state_target = state_target[:, 1:, :]
        target_msk = attention_mask[:, 1:]
        state_preds, action_preds, reward_preds, all_embs = self.model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            attention_mask=attention_mask,
        )
        state_preds = state_preds[:, :-1, :]
        pred_msk = attention_mask[:, :-1]

        new_msk = target_msk.int() & pred_msk.int()
        self.step += 1
        '''
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[
            attention_mask.reshape(-1) > 0
        ]
        '''
        state_dim = state_preds.shape[2]
        '''
        if not torch.all(target_msk[0]):
            print(target_msk[0])
            print(pred_msk[0])
            print(new_msk[0])
            print(state_preds[0, :, 0])
            print(state_target[0, :, 0])
            #raise Exception
        '''
        state_preds = state_preds.reshape(-1, state_dim)[new_msk.reshape(-1) > 0]
        #print(state_target.shape)
        #print(state_preds.shape)

        state_target = state_target.reshape(-1, state_dim)[
            new_msk.reshape(-1) > 0
        ]
        #if not torch.all(target_msk[0]):
        #    print(state_preds[:20, 0])
        #    print(state_target[:20, 0])
        #    raise Exception
        loss = self.loss_fn(
            state_preds,
            None,
            None,
            state_target,
            None,
            None,
        )
        if self.args["gpt_kmeans"]:
            loss += (
                self.args["gpt_kmeans_const"]
                * kmeans_anneal(self.step)
                * kmeans_cosine_max_loss(
                    F.normalize(self.model.cluster_centers, dim=-1),
                    F.normalize(all_embs, dim=-1),
                    mean=self.args["kmeans_mean"],
                )
            )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics["training/action_error"] = (
                torch.mean((state_preds - state_target) ** 2).detach().cpu().item()
            )

        return loss.detach().cpu().item()


    def train_iteration(self, num_steps, iter_num=0, seeder=None, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        if not self.eval_only:
            self.model.train()
            for _ in tqdm.tqdm(range(num_steps), desc="Training"):
                seeder(epoch=(iter_num - 1) * num_steps + _)
                train_loss = self.train_step()
                train_losses.append(train_loss)
                if self.scheduler is not None:
                    self.scheduler.step()

                logs["time/training"] = time.time() - train_start
                logs["training/train_loss_mean"] = np.mean(train_losses)
                logs["training/train_loss_std"] = np.std(train_losses)
        '''
        eval_start = time.time()

        self.model.eval()
        for eval_fn in tqdm.tqdm(self.eval_fns, desc="Evaluating"):

            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f"evaluation/{k}"] = v

        if not self.eval_only:
            logs["time/total"] = time.time() - self.start_time
        logs["time/evaluation"] = time.time() - eval_start
        '''
        logs["time/total"] = time.time() - self.start_time

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print("=" * 80)
            print(f"Iteration {iter_num}")
            for k, v in logs.items():
                print(f"{k}: {v}")

        if not self.eval_only:
            if self.args.get("outdir"):
                torch.save(
                    self.model.state_dict(),
                    f"{self.args['outdir']}/model_{iter_num}.pt",
                )

        return logs