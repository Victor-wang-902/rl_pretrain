import os

path = "checkpoints"
exps = []
for exp in os.listdir(path):
    finished = True
    for seed in os.listdir(os.path.join(path,exp)):
        seed_finished = False
        for file in os.listdir(os.path.join(path,exp,seed)):
            if "extra.json" in file:
                seed_finished = True
                break
        if not seed_finished:
            finished = False
            break
    if not finished:
        exps.append(exp)
for x in sorted(exps):
    print(x)
        