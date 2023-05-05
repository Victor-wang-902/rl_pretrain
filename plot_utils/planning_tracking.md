In this file we will keep track of the experiments we have run, and also plan for future experiments.

## Plot planning

Tables:
- main table: comparing a number of main variants: DT, DT with wiki pretrain, DT with RL pretrain, IL, IL with wiki pretrain, IL with RL pretrain, cql, cql with RL pretrain, maybe IQL, IQL with RL pretrain (currently we still need ILx3, IQLx2 and DT with RL pretrain)

With different y-axis values: 
- x-axis: CQL pretrain epoch
- x-axis: CQL layers (2 curves, pretrain/no pretrain)
- x-axis: DT different RL dataset size (2 curves, pretrain/no pretrain)
- x-axis: DT model size (2 curves, pretrain one still running)
- x-axis: DT perturb noise (later might want to add CQL perturb here?)

Things to run: 
- CQL, pretrain with forward dynamics, perturb noise
- CQL, different RL data sizes (0.1, 0.25, 0.5, 0.75)
- IL with DT codebase, a full set 

Why don't we get the figures out first, and then work on the new experiments? 

## Experiment tracking

### baselines
```
cql_prenone # default 2 layers
cql_preq_sprime

```
| Exp folder name | Note                                | 
|-----------------|-------------------------------------| 
| cql_prenone | cql baseline, no pretrain           | 
| cql_preq_sprime | cql1 with forward dynamics pretrain | 

Pretraining with different epoches: 'cql_preq_sprime_pe2', 'cql_preq_sprime_pe20', 'cql_preq_sprime' (default 200)

No pretrain with different layers: 'cql_prenone_pe200_layer4', 'cql_prenone_pe200_layer6', 'cql_prenone_pe200_layer8', 'cql_prenone' (default 2 layers)

Pretrain with different layers: 'cql_preq_sprime_pe200_layer4', 'cql_preq_sprime_pe200_layer6', 'cql_preq_sprime_pe200_layer8', 'cql_preq_sprime' (default 2 layers)


# dt related
dt baseline: 'dt'

dt wiki pretrain: 'chibiT'

dt no pretrain different layers: 
'dt_embed_dim256_n_layer4_n_head4', 'dt_embed_dim512_n_layer6_n_head8', 'dt_embed_dim768_n_layer12_n_head12', 'dt' (default 3 layers, 128 dim)

dt pretrain different layers: still working on it

dt with and w/o pretrain, different dataset size: 'dt_data_size0.1', 'dt_data_size0.25', 'dt_data_size0.5', 'dt_data_size0.75', 'dt' (default 100%)

'chibiT_data_size0.1', 'chibiT_data_size0.25', 'chibiT_data_size0.5', 'chibiT_data_size0.75', 'chibiT'

DT wikipedia pretrain, perturbation: 
'chibiT_perturb_per_layer1e0','chibiT_perturb_per_layer1e-1',
 'chibiT_perturb_per_layer2e0','chibiT_perturb_per_layer4e0', 'chibiT_perturb_per_layer8e0', 'chibiT' (default no perturb)







