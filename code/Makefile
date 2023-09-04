check:
	squeue -u zw2374

attach:
	srun --jobid $(id) --pty /bin/bash

clean:
	rm -f *.err *.out

index:
	sbatch encode.slurm

train:
	sbatch train_rl.slurm

pre:
	sbatch pretrain/pretrain.slurm

dist:
	sbatch pretrain/pretrain_dist.slurm

cancel:
	scancel  $(id)
