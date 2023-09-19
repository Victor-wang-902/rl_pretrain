import os
import pandas as pd
import json
import numpy as np
from infos import REF_MAX_SCORE, REF_MIN_SCORE
import shutil


def correct_TestEpNormRet(base_path, exp_prefix):
    for root, dirs, files in os.walk(base_path):
        if exp_prefix in root:
            for dir in dirs:
                subfolder = os.path.join(root, dir)
                for file in os.listdir(subfolder):
                    if file == 'progress.csv':
                        try:
                            progress_file = os.path.join(subfolder, file)
                            config_file = os.path.join(subfolder, 'config.json')
                            with open(progress_file, 'r') as f1, open(config_file, 'r') as f2:
                                df = pd.read_csv(f1, delimiter='\t', header=0)
                                config = json.load(f2)
                            setting = config['env'] + '-random-v0'
                            ref_min = REF_MIN_SCORE[setting]
                            ref_max = REF_MAX_SCORE[setting]
                            df['TestEpNormRet'] = 100 * (df['TestEpRet'] - ref_min) / (ref_max - ref_min)
                            df.to_csv(str(progress_file), sep='\t', index=False)
                            print(subfolder)
                        except Exception as e:
                            print(e)


def delete_seeds(base_path):
    for e in ['hopper', 'walker2d', 'halfcheetah', 'ant']:
        for d in ['medium', 'medium-replay', 'medium-expert']:
            folder = base_path + f'/iclr_cqlr3n_prenone_l2_{e}_{d}'
            for s in [4096]:
                path = folder + f'/iclr_cqlr3n_prenone_l2_{e}_{d}_s{s}'
                try:
                    shutil.rmtree(path)
                except:
                    print(path)


def check_pretrain_model(base_path, exp_prefix):
    for root, dirs, files in os.walk(base_path):
        if exp_prefix in root and 'iclr_cqlr3n_True' not in root:
            for dir in dirs:
                subfolder = os.path.join(root, dir)
                for file in os.listdir(subfolder):
                    if file == 'pretrain_progress.csv':
                        try:
                            progress_file = os.path.join(subfolder, file)
                            with open(progress_file, 'r') as f1:
                                df = pd.read_csv(f1, delimiter='\t', header=0)
                            if not df.empty:
                                print(subfolder)
                        except Exception as e:
                            print(e)


base_path = '../code/checkpoints/new_20seeds'
exp_prefix = 'new_iclr_cqlr3n'
correct_TestEpNormRet(base_path, exp_prefix)
# delete_seeds(base_path)
# check_pretrain_model(base_path, exp_prefix)
