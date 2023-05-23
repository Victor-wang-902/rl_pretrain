import torch
from torch.nn.functional import softmax
import random

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
                tok = torch.multinomial(self.parameters[0],4, replacement=True)
                generated = tok.unsqueeze(0)
        return generated.T
