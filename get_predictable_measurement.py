import os
import os.path as osp
import numpy as np
import yaml
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from glob import glob
from dataset import PenstateDataset
from builder import build_dataloader
from importlib import import_module

def get_err(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()
    baselines = []
    fuse_dist = []
    for line in lines:
        terms = line.rstrip().split(',')
        baselines.append(float(terms[-2].split(': ')[-1]))
        fuse_dist.append(float(terms[-1].split(': ')[-1]))

    #for b, f in zip(baselines, fuse_dist):
    #    print(b, f)
    baselines = np.array(baselines)
    fuse_dist = np.array(fuse_dist)

    return baselines, fuse_dist

# hfn v0 v1 v2
proj_dirs = glob("project/sgd_l2_721_vfn_12/2021*")
for proj_dir in proj_dirs:
    config_file = osp.join(proj_dir, 'configs.yml')
    with open(config_file, 'r') as f:
        configs = yaml.load(f, yaml.SafeLoader)

    val_path = proj_dir + '/val.log'
    val_baseline, val_err = get_err(val_path)
    eval_path = proj_dir + '/eval.log'
    eval_baseline, eval_err = get_err(eval_path)

    #print(val_baseline.shape, val_err.shape,
    #        eval_baseline.shape, eval_err.shape)

    index = np.argmin(val_err)
    print('{:5d}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}'.format(
        configs['data']['train']['dataset']['seed'],
        val_baseline[index].item(),
        eval_err[index].item(),
        eval_baseline[index].item(),
        eval_err[index].item(),
        (eval_baseline[index] - eval_err[index]).item(),
        (eval_err[index] / eval_baseline[index]).item()))


