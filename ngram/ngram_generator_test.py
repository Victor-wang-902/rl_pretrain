import torch
import random
import torch.multiprocessing as mp
import os
import time
import traceback
import numpy as np
from numpy.random import Generator, SeedSequence, default_rng
from sklearn.utils.extmath import softmax


class NGramGenerator:
    def __init__(
        self,
        ngram,
        nvocab,
        temperature,
        seed,
    ):
        torch.manual_seed(seed)
        self.vocab = nvocab
        self.parameters = [torch.rand(size=[nvocab for d in range(n+1)]) for n in range(ngram + 1)]
        self.seed = seed
        for i in range(len(self.parameters)):
            self.parameters[i] = softmax(self.parameters[i] / temperature, dim=-1)
        self.ngram = ngram
    def set_seed(self, iter_o, itr_i, max_len):
        itr = iter_o * max_len + itr_i
        seed_shift = itr * 9999
        mod_value = 9999999
        env_seed = (self.seed + seed_shift) % mod_value

        torch.manual_seed(env_seed)

    def generate(self, total_itr, max_length=1024, batch_size=64):
        generated = None
        gram = 0
        for itr in range(max_length):
            self.set_seed(total_itr, itr, max_length)
            if generated is not None:
                if len(generated) >= self.ngram:
                    gram = self.ngram
                else:
                    gram = len(generated)
                indices = generated[-gram:,:]
                probs = self.parameters[gram][indices.chunk(chunks=gram, dim=0)].squeeze()
                tok = torch.multinomial(probs, 1).squeeze().unsqueeze(0)
                generated = torch.cat([generated, tok], dim=0)
            else:
                tok = torch.multinomial(self.parameters[0],batch_size, replacement=True)
                generated = tok.unsqueeze(0)
        return generated.T

class NGramGeneratorOnline:
    def __init__(
        self,
        ngram,
        nvocab,
        temperature,
        seed,
        num_workers=4
    ):
        torch.manual_seed(seed)
        self.nvocab = nvocab
        self.seed = seed
        self.ngram = ngram
        self.sample_seeder = [torch.Generator() for _ in range(num_workers)]
        self.param_seeder = [[torch.Generator() for _ in range(self.ngram)] for __ in range(num_workers)]
        self.temperature = temperature
        self.pos_embed = [torch.rand(nvocab) * 2 for _ in range(self.ngram)]
        self.num_workers = num_workers

    def set_sample_seed(self, worker_id, iter_o, itr_i, max_len):
        itr = iter_o * max_len * self.num_workers + self.num_workers * itr_i + worker_id
        env_seed = self.seed + itr
        self.sample_seeder[worker_id].manual_seed(env_seed)

    def set_param_seed(self, worker_id, item=None):
        if item is not None:
            for i, n in enumerate(item):
                self.param_seeder[worker_id][i].manual_seed(self.seed + n + 1)
        else:
            self.param_seeder[worker_id][0].manual_seed(self.seed)

    def get_param(self, worker_id, indices=None):
        if indices is not None:
            transposed_inds = indices.T
            cur_ngram = transposed_inds.shape[1]
            params = []
            for item in transposed_inds:
                self.set_param_seed(worker_id, item.tolist())
                params.append(torch.sum(torch.stack([torch.rand(self.nvocab, generator=self.param_seeder[worker_id][i]) / self.ngram * self.pos_embed[i] for i in range(cur_ngram)]), dim=0))
            return torch.stack(params, dim=0)
        else:
            self.set_param_seed(worker_id)
            return torch.rand(self.nvocab, generator=self.param_seeder[worker_id][0])

    def generate_worker(self, worker_id, total_itr, max_length, batch_size, results_queue, event):
        generated = None
        for itr in range(max_length):
            self.set_sample_seed(worker_id, total_itr, itr, max_length)
            if generated is not None:
                indices = generated[-self.ngram:,:]
                probs = torch.softmax(torch.tensor(self.get_param(worker_id, indices).numpy() / self.temperature), dim=-1)
                tok = torch.multinomial(probs, 1, generator=self.sample_seeder[worker_id]).squeeze().reshape(1, -1) 
                generated = torch.cat([generated, tok], dim=0)
            else:
                probs = torch.softmax(torch.tensor(self.get_param(worker_id).numpy() / self.temperature), dim=-1)
                tok = torch.multinomial(probs, batch_size, replacement=True).squeeze()
                generated = tok.reshape(1, -1)
        results_queue.put(generated.T)
        event.set()
        return

    def generate(self, total_itr, max_length=1024, batch_size=32):
        results_queue = mp.Manager().Queue()
        processes = []
        event = mp.Event()

        for worker_id in range(self.num_workers):
            p = mp.Process(target=self.generate_worker, args=(worker_id, total_itr, max_length, batch_size // self.num_workers, results_queue, event))
            processes.append(p)
            p.start()
        event.wait()
        generated_batches = []
        while not results_queue.empty():
            generated_batches.append(results_queue.get())
        for p in processes:
            p.join()
        generated = torch.cat(generated_batches, dim=0)
        return generated


class NGramGeneratorOnlineNumpy:
    def __init__(
        self,
        ngram,
        nvocab,
        temperature,
        seed,
        num_workers=4
    ):
        self.seed = seed
        self.nvocab = nvocab
        self.ngram = ngram
        self.sample_seeder = [default_rng(self.seed) for _ in range(num_workers)]
        self.param_seeder = [default_rng(self.seed) for i in range(num_workers)]
        self.temperature = temperature
        #self.pos_embed = [np.random.rand(nvocab) * 2 for _ in range(self.ngram)]
        self.num_workers = num_workers

    def set_sample_seed(self, worker_id, iter_o, itr_i, max_len):
        itr = iter_o * max_len * self.num_workers + self.num_workers * itr_i + worker_id
        env_seed = self.seed + itr
        self.sample_seeder[worker_id] = default_rng(env_seed)

    def set_param_seed(self, worker_id, item=None):

        if item is not None:
            cur_seed = item + 1 + self.seed
            self.param_seeder[worker_id] = default_rng(cur_seed)


            #for i, n in enumerate(item):
            #    self.param_seeder[worker_id][i].bit_generator = SeedSequence(self.seed + n + 1)
        else:
            self.param_seeder[worker_id] = default_rng(self.seed)

    def get_param(self, worker_id, indices=None):
        if indices is not None:
            transposed_inds = indices.T
            cur_ngram = transposed_inds.shape[1]
            params = []
            for item in transposed_inds:
                self.set_param_seed(worker_id, item)
                param = self.param_seeder[worker_id].random(self.nvocab) 
                params.append(param)
            return np.stack(params, axis=0)
        else:
            self.set_param_seed(worker_id)
            return self.param_seeder[worker_id].random(self.nvocab).reshape(1,-1)

    def generate_worker(self, worker_id, total_itr, max_length, batch_size, results_queue, event):
        print("workerid", worker_id)
        generated = None
        for itr in range(max_length):
            self.set_sample_seed(worker_id, total_itr, itr, max_length)
            if generated is not None:
                indices = generated[-self.ngram:, :]
                #print(indices.shape)
                param_np = self.get_param(worker_id, indices)
                probs = softmax(param_np / self.temperature)
                #toks = []
                #for prob in probs:
                #    tok = self.sample_seeder[worker_id].multinomial(self.nvocab, size=(1,), p=prob.squeeze())
                #    toks.append(tok)
                #toks = np.stack(toks).reshape(1,-1)
                #print(probs)
                #print(self.sample_seeder[worker_id].multinomial(1, probs).argmax(1))
                #raise Exception
                toks = self.sample_seeder[worker_id].multinomial(1, probs).argmax(1).reshape(1,-1)
                #print(toks)
                generated = np.concatenate([generated, toks], axis=0)
            else:
                param_np = self.get_param(worker_id)
                probs = softmax(param_np / self.temperature)
                tok = self.sample_seeder[worker_id].choice(self.nvocab, size=(batch_size,), p=probs.squeeze())
                #print(self.sample_seeder[worker_id].multinomial(batch_size, probs).shape)

                #tok = self.sample_seeder[worker_id].multinomial(batch_size, probs).argmax(1).reshape(1,-1)
                #print(tok)
                #raise Exception
                tok = tok.reshape(1, -1)
                generated = tok
        results_queue.put(generated.T)
        event.set()
        return

    def generate(self, total_itr, max_length=1024, batch_size=32):
        results_queue = mp.Manager().Queue()
        processes = []
        event = mp.Event()

        for worker_id in range(self.num_workers):
            p = mp.Process(target=self.generate_worker, args=(worker_id, total_itr, max_length, batch_size // self.num_workers, results_queue, event))
            processes.append(p)
            p.start()
        event.wait()
        generated_batches = []
        for i in range(self.num_workers):
        #while not results_queue.empty():
            #print("check")
            generated_batches.append(results_queue.get())
        for p in processes:
            p.join()
        #print(len(generated_batches))
        #raise Exception
        generated = np.concatenate(generated_batches, axis=0)
        return generated


class NGramGeneratorOnlineNumpyFast:
    def __init__(
        self,
        ngram,
        nvocab,
        temperature,
        seed,
        num_workers=4
    ):
        self.seed = seed
        self.nvocab = nvocab
        self.ngram = ngram
        self.sample_seeder = [Generator(SeedSequence(seed)) for _ in range(num_workers)]
        self.param_seeder = [Generator(SeedSequence(seed)) for i in range(num_workers)]
        self.temperature = temperature
        #self.pos_embed = [np.random.rand(nvocab) * 2 for _ in range(self.ngram)]
        self.num_workers = num_workers

    def set_sample_seed(self, worker_id, iter_o, itr_i, max_len):
        itr = iter_o * max_len * self.num_workers + self.num_workers * itr_i + worker_id
        env_seed = self.seed + itr
        self.sample_seeder[worker_id].bit_generator = SeedSequence(env_seed)

    def set_param_seed(self, worker_id, item=None):
        if item is not None:
            for i, n in enumerate(item):
                self.param_seeder[worker_id][i].bit_generator = SeedSequence(self.seed + n + 1)
        else:
            self.param_seeder[worker_id][0].bit_generator = SeedSequence(self.seed)

    def get_param(self, worker_id, indices=None):
        if indices is not None:
            transposed_inds = indices.T
            cur_ngram = transposed_inds.shape[1]
            params = []
            for item in transposed_inds:
                self.set_param_seed(worker_id, item.tolist())
                param_sum = np.sum(
                    [
                        self.param_seeder[worker_id][i].random(self.nvocab) / self.ngram * self.pos_embed[i]
                        for i in range(cur_ngram)
                    ],
                    axis=0
                )
                params.append(param_sum)
            return np.stack(params, axis=0)
        else:
            self.set_param_seed(worker_id)
            return self.param_seeder[worker_id][0].random(self.nvocab)

    def generate_worker(self, worker_id, total_itr, max_length, batch_size, results_queue, event):
        generated = None
        for itr in range(max_length):
            self.set_sample_seed(worker_id, total_itr, itr, max_length)
            if generated is not None:
                indices = generated[-self.ngram:, :]
                param_np = self.get_param(worker_id, indices)
                probs = np.exp(param_np / self.temperature) / np.sum(np.exp(param_np / self.temperature), axis=-1, keepdims=True)
                tok = self.sample_seeder[worker_id].choice(self.nvocab, size=(1,), p=probs.squeeze())
                tok = tok.reshape(1, -1)
                generated = np.concatenate([generated, tok], axis=0)
            else:
                param_np = self.get_param(worker_id)
                probs = np.exp(param_np / self.temperature) / np.sum(np.exp(param_np / self.temperature), axis=-1, keepdims=True)
                tok = self.sample_seeder[worker_id].choice(self.nvocab, size=(batch_size,), p=probs.squeeze())
                tok = tok.reshape(1, -1)
                generated = tok
        results_queue.put(generated.T)
        event.set()
        return

    def generate(self, total_itr, max_length=1024, batch_size=32):
        results_queue = mp.Manager().Queue()
        processes = []
        event = mp.Event()

        for worker_id in range(self.num_workers):
            p = mp.Process(target=self.generate_worker, args=(worker_id, total_itr, max_length, batch_size // self.num_workers, results_queue, event))
            processes.append(p)
            p.start()
        event.wait()
        generated_batches = []
        while not results_queue.empty():
            generated_batches.append(results_queue.get())
        for p in processes:
            p.join()
        generated = np.concatenate(generated_batches, axis=0)
        return generated
