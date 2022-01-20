import os
import os.path as osp
import numpy as np
import yaml
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from glob import glob
from dataset import PenstateDataset
from builder import build_dataloader
from importlib import import_module
from scipy import stats

def parse_log(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()
    iters = []
    mean_dists = []
    fuse_dists = []
    selected_dists = []
    baselines = []
    for line in lines:
        terms = line.rstrip().split(',')
        iters.append(float(terms[1].split(': ')[-1]))
        mean_dists.append(float(terms[3].split(': ')[-1]))
        fuse_dists.append(float(terms[4].split(': ')[-1]))
        selected_dists.append(float(terms[5].split(': ')[-1]))
        baselines.append(float(terms[6].split(': ')[-1]))

    #for b, f in zip(baselines, fuse_dist):
    #    print(b, f)
    iters = np.array(iters)
    mean_dists = np.array(mean_dists)
    fuse_dists = np.array(fuse_dists)
    selected_dists = np.array(selected_dists)
    baselines = np.array(baselines)

    return iters, mean_dists, fuse_dists, selected_dists, baselines

def get_mvavg(data, wsize):
    avg_data = []
    for i in range(len(data)):
        st = max(0, i-wsize//2)
        ed = min(len(data), i+wsize//2+1)
        avg_data.append(sum(data[st:ed]) / len(data[st:ed]))

    return avg_data

def get_statistics(data):
    n = len(data)
    mu = data.mean()
    std = data.std()
    t_val = mu / std * np.sqrt(n)
    p_val = stats.t.sf(np.abs(t_val), n-1)
    CI = [mu - 1.676 * std / np.sqrt(n), mu + 1.676 * std / np.sqrt(n)]

    return t_val, p_val, CI, mu, std


# hfn v0 v1 v2
proj_dirs = glob("project/sgd_l2_vfn_22/2022*")
proj_dirs.sort()
wsize = 3
x1 = []
x2 = []
x3 = []
for idx, proj_dir in enumerate(proj_dirs):
    config_file = osp.join(proj_dir, 'configs.yml')
    with open(config_file, 'r') as f:
        configs = yaml.load(f, yaml.SafeLoader)

    val_path = proj_dir + '/val.log'
    val_iters, val_mean_dists, val_fuse_dists, val_selected_dists, val_baseline = parse_log(val_path)
    val_fuse_dists_ma = get_mvavg(val_fuse_dists, wsize)
    val_selected_dists_ma = get_mvavg(val_selected_dists, wsize)

    eval_path = proj_dir + '/eval.log'
    eval_iters, eval_mean_dists, eval_fuse_dists, eval_selected_dists, eval_baseline = parse_log(eval_path)

    # print(val_iters.shape, val_fuse_dists.shape, val_selected_dists.shape, val_baseline.shape,
    #         eval_iters.shape, eval_fuse_dists.shape, eval_selected_dists.shape, eval_baseline.shape)
    # xxxx

    index = np.argmin(val_fuse_dists)
    model_iter = val_iters[index].item()
    v_baseline = val_baseline[index].item()
    val_fuse_dist = val_fuse_dists[index].item()
    e_baseline = eval_baseline[index].item()
    eval_mean_dist = eval_mean_dists[index].item()
    eval_fuse_dist = eval_fuse_dists[index].item()
    eval_selected_dist = eval_selected_dists[index].item()

    print('{:2d}, {:5d}, {:6.3f}, {:6.3f}, {:6.3f}, {:6.3f}, {:6.3f}'.format(
        idx, configs['data']['train']['dataset']['seed'],
        eval_fuse_dist, eval_selected_dist, e_baseline,
        e_baseline - eval_fuse_dist,
        e_baseline - eval_selected_dist))
    
    x1.append(e_baseline - eval_fuse_dist)
    x2.append(e_baseline - eval_selected_dist)
    x3.append(eval_mean_dist - eval_fuse_dist)
    
# critical value
x1 = np.array(x1)
t1, p1, CI1, mu1, std1 = get_statistics(x1)

x2 = np.array(x2)
t2, p2, CI2, mu2, std2 = get_statistics(x2)

x3 = np.array(x3)
t3, p3, CI3, mu3, std3 = get_statistics(x3)


print('t-score: {:8.5f}, p-value: {:8.5e}, CI-0.95: [{:8.5f}, {:8.5f}], mean: {:8.5f}, std: {:8.5f}'.format(t1, p1, CI1[0], CI1[1], mu1, std1))
print('t-score: {:8.5f}, p-value: {:8.5e}, CI-0.95: [{:8.5f}, {:8.5f}], mean: {:8.5f}, std: {:8.5f}'.format(t2, p2, CI2[0], CI2[1], mu2, std2))
print('t-score: {:8.5f}, p-value: {:8.5e}, CI-0.95: [{:8.5f}, {:8.5f}], mean: {:8.5f}, std: {:8.5f}'.format(t3, p3, CI3[0], CI3[1], mu3, std3))

