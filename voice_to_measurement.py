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
from builder import build_dataloader


def load_net(net_config, module):
    obj_type = net_config.pop('type')
    obj_cls = getattr(import_module('model.{}'.format(module)), obj_type)
    net = obj_cls(**net_config)

    model_path = osp.join(proj_dir, 'models/{}.pth'.format(module))
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()

    return net

def get_statistics(data):
    n = len(data)
    mu = data.mean()
    std = data.std()

    t_val = mu / std * np.sqrt(n)
    p_val = stats.t.sf(np.abs(t_val), n-1)
    CI = [mu - 1.676 * std / np.sqrt(n), mu + 1.676 * std / np.sqrt(n)]
    
    return t_val, p_val, CI, mu, std

# hfn v0 v1 v2
proj_dirs = glob("project/sgd_l2_vfn_12/2022*")
proj_dirs.sort()
all_mean_mse = []
all_fuse_mse = []
all_selected_mse_25 = []
all_selected_mse_50 = []
all_selected_mse_75 = []
all_base_mse = []
for idx, proj_dir in enumerate(proj_dirs):
    config_file = osp.join(proj_dir, 'configs.yml')
    with open(config_file, 'r') as f:
        configs = yaml.load(f, yaml.SafeLoader)

    # data 
    eval_config = copy.deepcopy(configs['data']['eval'])
    eval_loader = build_dataloader(eval_config)
    # print(eval_loader)

    # network
    bkb_config = copy.deepcopy(configs['model']['backbone']['net'])
    # print(bkb_config)
    bkb = load_net(bkb_config, 'backbone')
    head_config = copy.deepcopy(configs['model']['head']['net'])
    # print(head_config)
    head = load_net(head_config, 'head')

    # eval
    sample_mean_preds = []
    sample_fuse_preds = []
    sample_fuse_confs = []
    sample_targets = []
    for idx, voices, targets, _, _ in eval_loader:
        voices, targets = voices.cuda(), targets.cuda()
        targets = torch.unsqueeze(targets, -1)

        preds, confs = head(bkb(voices))

        mean_preds = torch.mean(preds, dim=2, keepdim=True)
        fuse_preds = torch.sum(preds * confs, dim=2, keepdim=True)
        fuse_confs = torch.sum(confs, dim=2, keepdim=True)
        fuse_preds = fuse_preds / fuse_confs

        sample_mean_preds.append(mean_preds)
        sample_fuse_preds.append(fuse_preds)
        #fuse_confs = torch.mean(confs, dim=2, keepdim=True)
        sample_fuse_confs.append(fuse_confs)
        sample_targets.append(targets)
    

    sample_mean_preds = torch.cat(sample_mean_preds, dim=0)
    sample_fuse_preds = torch.cat(sample_fuse_preds, dim=0)
    sample_fuse_confs = torch.cat(sample_fuse_confs, dim=0)
    sample_targets = torch.cat(sample_targets, dim=0)
    
    sample_mean_dists = torch.square(sample_mean_preds - sample_targets)
    sample_fuse_dists = torch.square(sample_fuse_preds - sample_targets)
    sample_baseline = torch.square(sample_targets)

    count = torch.numel(sample_fuse_confs)
    [_, conf_indices] = torch.sort(sample_fuse_confs.flatten())
    conf_indices_25 = conf_indices[-int(0.25 * count):]
    conf_indices_50 = conf_indices[-int(0.50 * count):]
    conf_indices_75 = conf_indices[-int(0.75 * count):]

 
    mean_mse = sample_mean_dists.mean().item()
    fuse_mse = sample_fuse_dists.mean().item()
    selected_mse_25 = sample_fuse_dists.flatten()[conf_indices_25].mean().item()
    selected_mse_50 = sample_fuse_dists.flatten()[conf_indices_50].mean().item()
    selected_mse_75 = sample_fuse_dists.flatten()[conf_indices_75].mean().item()
    base_mse = sample_baseline.mean().item()
    

    print(', '.join([
        'mean_dist: {:8.5f}'.format(mean_mse),
        'fuse_dist: {:8.5f}'.format(fuse_mse),
        'selected_dist_25: {:8.5f}'.format(selected_mse_25),
        'selected_dist_50: {:8.5f}'.format(selected_mse_50),
        'selected_dist_75: {:8.5f}'.format(selected_mse_75),
        'baseline: {:8.5f}'.format(base_mse),
        ]))

    all_mean_mse.append(mean_mse)
    all_fuse_mse.append(fuse_mse)
    all_selected_mse_25.append(selected_mse_25)
    all_selected_mse_50.append(selected_mse_50)
    all_selected_mse_75.append(selected_mse_75)
    all_base_mse.append(base_mse)


all_mean_mse = np.array(all_mean_mse)
all_fuse_mse = np.array(all_fuse_mse)
all_selected_mse_25 = np.array(all_selected_mse_25)
all_selected_mse_50 = np.array(all_selected_mse_50)
all_selected_mse_75 = np.array(all_selected_mse_75)
all_base_mse = np.array(all_base_mse)

t1, p1, CI1, mu1, std1 = get_statistics(all_base_mse - all_mean_mse)
t2, p2, CI2, mu2, std2 = get_statistics(all_base_mse - all_fuse_mse)
t3, p3, CI3, mu3, std3 = get_statistics(all_base_mse - all_selected_mse_25)
t4, p4, CI4, mu4, std4 = get_statistics(all_base_mse - all_selected_mse_50)
t5, p5, CI5, mu5, std5 = get_statistics(all_base_mse - all_selected_mse_75)
t6, p6, CI6, mu6, std6 = get_statistics(all_mean_mse - all_fuse_mse)


print('t-score: {:8.5f}, p-value: {:8.5e}, CI-0.95: [{:8.5f}, {:8.5f}], mean: {:8.5f}, std: {:8.5f}'.format(t1, p1, CI1[0], CI1[1], mu1, std1))
print('t-score: {:8.5f}, p-value: {:8.5e}, CI-0.95: [{:8.5f}, {:8.5f}], mean: {:8.5f}, std: {:8.5f}'.format(t2, p2, CI2[0], CI2[1], mu2, std2))
print('t-score: {:8.5f}, p-value: {:8.5e}, CI-0.95: [{:8.5f}, {:8.5f}], mean: {:8.5f}, std: {:8.5f}'.format(t3, p3, CI3[0], CI3[1], mu3, std3))
print('t-score: {:8.5f}, p-value: {:8.5e}, CI-0.95: [{:8.5f}, {:8.5f}], mean: {:8.5f}, std: {:8.5f}'.format(t4, p4, CI4[0], CI4[1], mu4, std4))
print('t-score: {:8.5f}, p-value: {:8.5e}, CI-0.95: [{:8.5f}, {:8.5f}], mean: {:8.5f}, std: {:8.5f}'.format(t5, p5, CI5[0], CI5[1], mu5, std5))
print('t-score: {:8.5f}, p-value: {:8.5e}, CI-0.95: [{:8.5f}, {:8.5f}], mean: {:8.5f}, std: {:8.5f}'.format(t6, p6, CI6[0], CI6[1], mu6, std6))



