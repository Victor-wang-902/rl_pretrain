import torch
from torch.nn.functional import softmax
import random
import torch.multiprocessing as mp
import os
import time
import traceback

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
        #print(itr)
        env_seed = self.seed + itr

        self.sample_seeder[worker_id].manual_seed(env_seed)

    def set_param_seed(self, worker_id, item=None):
        if item is not None:
            for i, n in enumerate(item):
                self.param_seeder[worker_id][i].manual_seed(self.seed + n + 1)
        else:
            self.param_seeder[worker_id][0].manual_seed(self.seed)

    def get_param(self, worker_id, indices=None):
        #print(worker_id, "check 15")

        if indices is not None:
            transposed_inds = indices.T
            #print(worker_id, "check 16")

            cur_ngram = transposed_inds.shape[1]
            #print(worker_id, "check 17")

            params = []
            for item in transposed_inds:
                #print(worker_id, "check 18")

                self.set_param_seed(worker_id, item.tolist())
                #print(worker_id, "check 19")

                params.append(torch.sum(torch.stack([torch.rand(self.nvocab, generator=self.param_seeder[worker_id][i]) / self.ngram * self.pos_embed[i] for i in range(cur_ngram)]), dim=0))
                #print(worker_id, "check 20")

            #print(params)
            return torch.stack(params, dim=0)
        else:
            #print(worker_id, "check 21")

            self.set_param_seed(worker_id)
            #print(worker_id, "check 22")
            #params = torch.rand(self.nvocab, generator=self.param_seeder[worker_id][0]).numpy() / self.temperature
            return torch.rand(self.nvocab, generator=self.param_seeder[worker_id][0])
            #return params

    #def generate_worker(self, worker_id, total_itr, max_length, batch_size, results_queue):#, event):
    def generate_worker(self, worker_id, total_itr, max_length, batch_size, results_queue, event):
        #start_seq = worker_id * batch_size
        #end_seq = (worker_id + 1) * batch_size
        #print(worker_id, "check 1")
        generated = None
        for itr in range(max_length):
            #print(worker_id, "check 2, itr", itr)

            self.set_sample_seed(worker_id, total_itr, itr, max_length)
            #print(worker_id, "check 3, itr", itr)

            if generated is not None:
                #print(os.getpid(), generated)
                indices = generated[-self.ngram:,:]
                #print(os.getpid(), indices)
                #print(worker_id, "check 4, itr", itr)
                probs = torch.softmax(self.get_param(worker_id, indices) / self.temperature, dim=-1)
                #probs = torch.softmax(self.get_param(worker_id, indices), dim=-1)

                #probs = torch.softmax(torch.tensor(self.get_param(worker_id, indices).numpy() / self.temperature), dim=-1)
                #print(worker_id, "check 5, itr", itr)

                #print(worker_id, self.sample_seeder[worker_id].initial_seed(), probs)
                tok = torch.multinomial(probs, 1, generator=self.sample_seeder[worker_id]).squeeze().reshape(1, -1)
                #print(worker_id, "check 6, itr", itr)
 
                generated = torch.cat([generated, tok], dim=0)
                #print(worker_id, "check 7, itr", itr)

            else:
                #print(worker_id, "check 8, itr", itr)
                #try:
                #    print(self.get_param(worker_id) / self.temperature)
                #except Exception as err:
                #    traceback.print_exc()
                #print(torch.tensor(self.get_param(worker_id).numpy() / self.temperature))
                #print("worked")
                #probs = torch.softmax(torch.tensor(self.get_param(worker_id).numpy() / self.temperature), dim=-1)
                probs = torch.softmax(self.get_param(worker_id) / self.temperature, dim=-1)
                #probs = torch.softmax(self.get_param(worker_id), dim=-1)

                #transformed_probs = self.get_param(worker_id)
                #print(transformed_probs.dtype)
                #transformed_probs = transformed_probs / torch.full((self.nvocab,),self.temperature,dtype=torch.float32)
                #probs = softmax(transformed_probs, dim=-1)
                #print(worker_id, "check 9, itr", itr)

                tok = torch.multinomial(probs, batch_size, replacement=True).squeeze()
                #print(worker_id, "check 10, itr", itr)

                generated = tok.reshape(1, -1)
                #print(worker_id, "check 11, itr", itr)

        #results_queue.put(generated.T.numpy())
        #print(worker_id, "check 12")

        results_queue.put(generated.T)
        #print(worker_id, "check 13")

        #print(os.getpid(), "check")
        event.set()
        #print(worker_id, "check 14")

        return
        #if generated.numel() > 0:
        #    results_queue.put(generated.T.numpy())
        #else:
        #    results_queue.put(torch.zeros((max_length, batch_size)).long().numpy())


    def generate(self, total_itr, max_length=1024, batch_size=32):
        results_queue = mp.Manager().Queue()
        processes = []
        event = mp.Event()

        for worker_id in range(self.num_workers):
            p = mp.Process(target=self.generate_worker, args=(worker_id, total_itr, max_length, batch_size // self.num_workers, results_queue, event))
            #p = mp.Process(target=self.generate_worker, args=(worker_id, total_itr, max_length, batch_size // self.num_workers, results_queue))
            processes.append(p)
            p.start()
            #time.sleep(1)
        #time.sleep(10)
        #print("main starts to wait")
        event.wait()
        generated_batches = []

        #print(results_queue.empty())
        for _ in range(self.num_workers):
            #while not results_queue.empty():
            #generated_batches.append(torch.tensor(results_queue.get()))
            generated_batches.append(results_queue.get())
        #print(generated_batches)
        for p in processes:
            p.join()
            #p.terminate()

        generated = torch.cat(generated_batches, dim=0)
        return generated