from ngram_generator import NGramGenerator, NGramGeneratorOnline
import argparse
import os
import csv
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvocab", type=int)
    parser.add_argument("--ngram", type=int)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--raw", action="store_true", default=False)
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--online", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    args.batch_size = args.batch_size * args.num_workers
    if args.online:
        gen = NGramGeneratorOnline(ngram=args.ngram, nvocab=args.nvocab, seed=args.seed, temperature=args.temperature, num_workers=args.num_workers)
    else:
        gen = NGramGenerator(ngram=args.ngram, nvocab=args.nvocab, seed=args.seed, temperature=args.temperature)
    filepath = os.path.join(args.outdir, "data_ngram_" + str(args.ngram) + "_nvocab_" + str(args.nvocab) + "_temperature_" + str(args.temperature) + ".csv")
    print("saving to", filepath)
    for itr in tqdm(range(args.iterations)):
        batch = gen.generate(itr, max_length=args.length, batch_size=args.batch_size)
        if args.raw:
            with open(filepath, "a") as f:
                writer = csv.writer(f)
                #print(batch.shape)
                #print(batch.tolist())
                writer.writerows(batch.tolist())
                #raise Exception
        else:
            raise NotImplementedError
