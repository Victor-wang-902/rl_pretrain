import time
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from datasets import load_dataset
import tqdm
import random
import math
from transformers import AutoTokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class GPT2Dataset(IterableDataset):
    def __init__(self, data, tokenizer, split="train", n_tokens=1024, seed=0, data_size=1.0):
        super().__init__()
        self.tokenizer = tokenizer
        self.n_tokens = n_tokens
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.tokenized = []
        ind = list(range(len(data)))
        if split == "train":
            random.seed(seed)
            random.shuffle(ind)
            new_len = int(len(ind) * data_size)
            ind = ind[:new_len]
        print("read %s entries from the raw dataset" % str(len(ind)))
        print("tokenizing...")
        for i in tqdm.tqdm(ind):
            line = data[i]
            if line["text"] == "":
                continue
            temp = self.tokenizer(line["text"], padding=False, truncation=False)
            input_ids = [self.bos_token_id] + temp["input_ids"] + [self.eos_token_id]
            attention_mask = [1] + temp["attention_mask"] + [1]
            self.tokenized.append({"input_ids": input_ids, "attention_mask": attention_mask})
        print("processed %s entries in the tokenized dataset" % str(len(self.tokenized)))
        self.start = 0
        self.end = len(self.tokenized)
    def __iter__(self):
        input_ids_list = []
        attention_mask_list = []
        for i, sent in enumerate(self.tokenized[self.start:self.end]):
            #print(self.start + i, "\n")
            input_ids = sent["input_ids"]
            attention_mask = sent["attention_mask"]
            while len(input_ids) > 0:
                input_ids_list += input_ids[:self.n_tokens - len(input_ids_list)]
                attention_mask_list += attention_mask[:self.n_tokens - len(attention_mask_list)]
                input_ids = input_ids[self.n_tokens - len(input_ids_list):]
                attention_mask = attention_mask[self.n_tokens - len(attention_mask_list):]
                if len(input_ids_list) >= self.n_tokens:
                    yield {"input_ids": input_ids_list, "attention_mask": attention_mask_list}
                    input_ids_list = []
                    attention_mask_list = []

class GPT2Collator:
    def __init__(self, device):
        self.device = device

    def __call__(self, batch):
        input_ids_list = []
        attention_mask_list = []
        for item in batch:
            input_ids = item["input_ids"] 
            attention_mask =item["attention_mask"]
            input_ids_list.append(torch.tensor(input_ids, dtype=torch.int))
            attention_mask_list.append(torch.tensor(attention_mask, dtype=torch.int))
        return torch.stack(input_ids_list).to(self.device), torch.stack(attention_mask_list).to(self.device)

def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    overall_start = dataset.start
    overall_end = dataset.end
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)

def get_dataloader(dataset, collator, batch_size, worker_init_fn=None, num_workers=0, drop_last=True):
    return DataLoader(dataset, collate_fn=collator, batch_size=batch_size, num_workers=num_workers, worker_init_fn=worker_init_fn, drop_last=drop_last)

#9:14W

tokenizer = AutoTokenizer.from_pretrained("gpt2")
device = torch.device("cpu")
test = load_dataset("wikitext", "wikitext-103-raw-v1")["train"]
dataset = GPT2Dataset(test, tokenizer, split="train", data_size=0.1)
collator = GPT2Collator(device)
for i in [2,4,8,16]:
  dataloader = get_dataloader(dataset, collator, batch_size=64, worker_init_fn=worker_init_fn, num_workers=i)
  start_time = time.time()
  for batch in dataloader:
    continue
  end_time = time.time()
  print("num_workers: %s, time: %s" % (i, (end_time - start_time)))