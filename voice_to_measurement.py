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

def paired_two_sample_t_test(sample1, sample2):
    sample1 = np.array(sample1)
    sample2 = np.array(sample2)

    diff = sample1 - sample2
    n = len(diff)
    mu = diff.mean()
    std = diff.std()

    t_val = mu / std * np.sqrt(n)
    p_val = stats.t.sf(np.abs(t_val), n-1)
    CI = [mu - 1.676 * std / np.sqrt(n), mu + 1.676 * std / np.sqrt(n)]
    
    return t_val, p_val, CI, mu, std

# hfn v0 v1 v2
meas_id = '51'
proj_dirs = glob("project/seed_666_adam/adam_l2_vfn_{}/2022*".format(meas_id))
proj_dirs.sort()
print(proj_dirs[0])
mean_mse = []
fuse_mse = []
selected_mse_25 = []
selected_mse_50 = []
selected_mse_75 = []
base_mse = []
for idx, proj_dir in enumerate(proj_dirs):
    config_file = osp.join(proj_dir, 'configs.yml')
    with open(config_file, 'r') as f:
        configs = yaml.load(f, yaml.SafeLoader)

    # data 
    eval_config = copy.deepcopy(configs['data']['eval'])
    eval_config['dataset']['norm_mu_path'] = osp.join(proj_dir, 'norm_mu.txt')
    eval_config['dataset']['norm_std_path'] = osp.join(proj_dir, 'norm_std.txt')
    eval_loader = build_dataloader(eval_config)

    # network
    bkb_config = copy.deepcopy(configs['model']['backbone']['net'])
    bkb = load_net(bkb_config, 'backbone')
    head_config = copy.deepcopy(configs['model']['head']['net'])
    head = load_net(head_config, 'head')

    # eval
    mean_measurements = []
    fuse_measurements = []
    fuse_confidences = []
    gt_measurements = []
    for idx, voices, targets, _, _ in eval_loader:
        voices, targets = voices.cuda(), targets.cuda()
        targets = torch.unsqueeze(targets, -1)

        # voices = voices[:, :int(voices.size(1)*0.2)]
        preds, confs = head(bkb(voices))

        mean_preds = torch.mean(preds, dim=2, keepdim=True)
        fuse_preds = torch.sum(preds * confs, dim=2, keepdim=True)
        fuse_confs = torch.sum(confs, dim=2, keepdim=True)
        fuse_preds = fuse_preds / fuse_confs
        fuse_confs = torch.mean(confs, dim=2, keepdim=True)
        #fuse_confs, _ = torch.max(confs, dim=2, keepdim=True)
        
        mean_measurements.append(mean_preds)
        fuse_measurements.append(fuse_preds)
        fuse_confidences.append(fuse_confs)
        gt_measurements.append(targets)

    mean_measurements = torch.cat(mean_measurements, dim=0).flatten()
    fuse_measurements = torch.cat(fuse_measurements, dim=0).flatten()
    fuse_confidences = torch.cat(fuse_confidences, dim=0).flatten()
    gt_measurements = torch.cat(gt_measurements, dim=0).flatten()

    count = torch.numel(fuse_confidences)
    [_, conf_indices] = torch.sort(fuse_confidences.flatten())
    conf_indices_25 = conf_indices[-int(0.25 * count):]
    conf_indices_50 = conf_indices[-int(0.50 * count):]
    conf_indices_75 = conf_indices[-int(0.75 * count):]

    mean_errors = torch.square(mean_measurements - gt_measurements)
    fuse_errors = torch.square(fuse_measurements - gt_measurements)
    selected_errors_25 = fuse_errors[conf_indices_25]
    selected_errors_50 = fuse_errors[conf_indices_50]
    selected_errors_75 = fuse_errors[conf_indices_75]
    base_errors = torch.square(gt_measurements)

    mean_mse.append(mean_errors.mean().item())
    fuse_mse.append(fuse_errors.mean().item())
    selected_mse_25.append(selected_errors_25.mean().item())
    selected_mse_50.append(selected_errors_50.mean().item())
    selected_mse_75.append(selected_errors_75.mean().item())
    base_mse.append(base_errors.mean().item())

    print(', '.join([
        'mean_mse: {:8.5f}'.format(mean_mse[-1]),
        'fuse_mse: {:8.5f}'.format(fuse_mse[-1]),
        'selected_mse_25: {:8.5f}'.format(selected_mse_25[-1]),
        'selected_mse_50: {:8.5f}'.format(selected_mse_50[-1]),
        'selected_mse_75: {:8.5f}'.format(selected_mse_75[-1]),
        'baseline: {:8.5f}'.format(base_mse[-1]),
        ]))

# compute t-test statistics
t1, p1, CI1, mu1, std1 = paired_two_sample_t_test(base_mse, mean_mse)
t2, p2, CI2, mu2, std2 = paired_two_sample_t_test(base_mse, fuse_mse)
t3, p3, CI3, mu3, std3 = paired_two_sample_t_test(base_mse, selected_mse_25)
t4, p4, CI4, mu4, std4 = paired_two_sample_t_test(base_mse, selected_mse_50)
t5, p5, CI5, mu5, std5 = paired_two_sample_t_test(base_mse, selected_mse_75)
t6, p6, CI6, mu6, std6 = paired_two_sample_t_test(mean_mse, fuse_mse)

print('mean - t-score: {:8.5f}, p-val: {:8.5e}, CI-0.90: [{:8.5f}, {:8.5f}], mean: {:8.5f}, std: {:8.5f}'.format(t1, p1, CI1[0], CI1[1], mu1, std1))
print('fuse - t-score: {:8.5f}, p-val: {:8.5e}, CI-0.90: [{:8.5f}, {:8.5f}], mean: {:8.5f}, std: {:8.5f}'.format(t2, p2, CI2[0], CI2[1], mu2, std2))
print('se25 - t-score: {:8.5f}, p-val: {:8.5e}, CI-0.90: [{:8.5f}, {:8.5f}], mean: {:8.5f}, std: {:8.5f}'.format(t3, p3, CI3[0], CI3[1], mu3, std3))
print('se50 - t-score: {:8.5f}, p-val: {:8.5e}, CI-0.90: [{:8.5f}, {:8.5f}], mean: {:8.5f}, std: {:8.5f}'.format(t4, p4, CI4[0], CI4[1], mu4, std4))
print('se75 - t-score: {:8.5f}, p-val: {:8.5e}, CI-0.90: [{:8.5f}, {:8.5f}], mean: {:8.5f}, std: {:8.5f}'.format(t5, p5, CI5[0], CI5[1], mu5, std5))
print('mufu - t-score: {:8.5f}, p-val: {:8.5e}, CI-0.90: [{:8.5f}, {:8.5f}], mean: {:8.5f}, std: {:8.5f}'.format(t6, p6, CI6[0], CI6[1], mu6, std6))



