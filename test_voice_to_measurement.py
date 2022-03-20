import os
import os.path as osp
import numpy as np
import yaml
import copy
import math
import random

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


# hfn v0 v1 v2
meas_id = '51'
proj_dirs = glob("project/seed_666_adam/adam_l2_vfn_{}/2022*".format(meas_id))
print(proj_dirs[0])
random.shuffle(proj_dirs)
# proj_dirs.sort()
n_proj = len(proj_dirs)
mean_measurements = []
fuse_measurements = []
fuse_confidences = []
gt_measurements = []
with torch.no_grad():
    for idx, proj_dir in enumerate(proj_dirs):
        if idx % 10 == 0:
           print(idx)

        config_file = osp.join(proj_dir, 'configs.yml')
        with open(config_file, 'r') as f:
            configs = yaml.load(f, yaml.SafeLoader)
    
        # data 
        test_config = copy.deepcopy(configs['data']['eval'])
        test_config['dataset']['mode'] = 'test'
        test_config['dataset']['norm_mu_path'] = osp.join(proj_dir, 'norm_mu.txt')
        test_config['dataset']['norm_std_path'] = osp.join(proj_dir, 'norm_std.txt')
        test_loader = build_dataloader(test_config)
        norm_mu = test_loader.dataset.norm_mu.item()
        norm_std = test_loader.dataset.norm_std.item()
    
        # network
        bkb_config = copy.deepcopy(configs['model']['backbone']['net'])
        bkb = load_net(bkb_config, 'backbone')
        head_config = copy.deepcopy(configs['model']['head']['net'])
        head = load_net(head_config, 'head')
    
        # eval
        for _, voices, targets, _, _ in test_loader:
            voices, targets = voices.cuda(), targets.cuda()
            targets = torch.unsqueeze(targets, -1)
    
            preds, confs = head(bkb(voices))
    
            mean_preds = torch.mean(preds, dim=2, keepdim=True)
            fuse_preds = torch.sum(preds * confs, dim=2, keepdim=True)
            fuse_confs = torch.sum(confs, dim=2, keepdim=True)
            fuse_preds = fuse_preds / fuse_confs
            fuse_confs = torch.mean(confs, dim=2, keepdim=True)
            # [fuse_confs, _] = torch.max(confs, dim=2, keepdim=True)
    
            mean_measurements.append(norm_mu + mean_preds * norm_std)
            fuse_measurements.append(norm_mu + fuse_preds * norm_std)
            fuse_confidences.append(fuse_confs)
            gt_measurements.append(norm_mu + targets * norm_std)

    mean_measurements = torch.cat(mean_measurements, dim=0).view(n_proj, -1)
    fuse_measurements = torch.cat(fuse_measurements, dim=0).view(n_proj, -1)
    fuse_confidences = torch.cat(fuse_confidences, dim=0).view(n_proj, -1)
    gt_measurements = torch.cat(gt_measurements, dim=0).view(n_proj, -1)

    mean_measurements = torch.cumsum(mean_measurements, dim=0)
    mean_confidences = torch.cumsum(torch.ones_like(fuse_confidences), dim=0)
    mean_measurements /= mean_confidences
    fuse_measurements = torch.cumsum(fuse_measurements * fuse_confidences, dim=0)
    fuse_confidences = torch.cumsum(fuse_confidences, dim=0)
    fuse_measurements /= fuse_confidences

    count = fuse_confidences.size(1)
    [_, conf_indices] = torch.sort(fuse_confidences, dim=1)
    conf_indices_25 = conf_indices[:, -int(0.25 * count):]
    conf_indices_50 = conf_indices[:, -int(0.50 * count):]
    conf_indices_75 = conf_indices[:, -int(0.75 * count):]
    conf_indices_25_50 = conf_indices[:, -int(0.50 * count):-int(0.25 * count)]

    baseline = torch.var(gt_measurements)
    mean_errors = torch.square(mean_measurements - gt_measurements) / baseline
    fuse_errors = torch.square(fuse_measurements - gt_measurements) / baseline
    selected_errors_25 = torch.gather(fuse_errors, 1, conf_indices_25)
    selected_errors_50 = torch.gather(fuse_errors, 1, conf_indices_50)
    selected_errors_75 = torch.gather(fuse_errors, 1, conf_indices_75)

for idx in range(n_proj):
    print(', '.join([
        '{:2d}'.format(idx),
        'mean_mse: {:8.5f}'.format(mean_errors[idx].mean().item()),
        'fuse_mse: {:8.5f}'.format(fuse_errors[idx].mean().item()),
        'selected_mse_25: {:8.5f}'.format(selected_errors_25[idx].mean().item()),
        'selected_mse_50: {:8.5f}'.format(selected_errors_50[idx].mean().item()),
        'selected_mse_75: {:8.5f}'.format(selected_errors_75[idx].mean().item()),
        ]))

# save fuse measurements and confidences
results = np.array([
    fuse_measurements[-1].detach().cpu().numpy(),
    gt_measurements[-1].detach().cpu().numpy(),
    fuse_confidences[-1].detach().cpu().numpy(),
])
# np.savetxt('project/seed_666/' + meas_id + '.txt', results, fmt='%12.8f')
