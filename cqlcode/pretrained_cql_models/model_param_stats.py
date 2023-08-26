import os

import torch
import torch.nn as nn
import sys
sys.path.append('../')
import SimpleSAC


pretrained_model_folder = './sendbackmodel'
pretrained_models_names = [
    'cql_ant_1000_100_100_1_1_111_8_mdp_same_noproj_2_256_200.pth',
    'cql_halfcheetah_1000_100_100_1_1_17_6_mdp_same_noproj_2_256_200.pth',
    'cql_hopper_1000_100_100_1_1_11_3_mdp_same_noproj_2_256_200.pth',
    'cql_walker2d_1000_100_100_1_1_17_6_mdp_same_noproj_2_256_200.pth'
]

for name in pretrained_models_names:
    path = os.path.join(pretrained_model_folder, name)

    model = torch.load(path)
    print(model)
    break