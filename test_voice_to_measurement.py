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

def get_statistics(data):
    n = len(data)
    mu = data.mean()
    std = data.std()

    t_val = mu / std * np.sqrt(n)
    p_val = stats.t.sf(np.abs(t_val), n-1)
    CI = [mu - 1.676 * std / np.sqrt(n), mu + 1.676 * std / np.sqrt(n)]
    
    return t_val, p_val, CI, mu, std

# hfn v0 v1 v2
proj_dirs = glob("project/sgd_l2_vfn_02/2022*")
random.shuffle(proj_dirs)
# proj_dirs.sort()
model_mean_preds = []
model_fuse_preds = []
model_fuse_confs = []
model_targets = []
with torch.no_grad():
    for idx, proj_dir in enumerate(proj_dirs):
        config_file = osp.join(proj_dir, 'configs.yml')
        with open(config_file, 'r') as f:
            configs = yaml.load(f, yaml.SafeLoader)
    
        # data 
        test_config = copy.deepcopy(configs['data']['eval'])
        test_config['dataset']['mode'] = 'test'
        test_loader = build_dataloader(test_config)
    
        # network
        bkb_config = copy.deepcopy(configs['model']['backbone']['net'])
        bkb = load_net(bkb_config, 'backbone')
        head_config = copy.deepcopy(configs['model']['head']['net'])
        head = load_net(head_config, 'head')
    
        # eval
        sample_mean_preds = []
        sample_fuse_preds = []
        sample_fuse_confs = []
        sample_targets = []
        for idx, voices, targets, _, _ in test_loader:
            voices, targets = voices.cuda(), targets.cuda()
            targets = torch.unsqueeze(targets, -1)
    
            preds, confs = head(bkb(voices))
    
            mean_preds = torch.mean(preds, dim=2, keepdim=True)
            fuse_preds = torch.sum(preds * confs, dim=2, keepdim=True)
            fuse_confs = torch.sum(confs, dim=2, keepdim=True)
            fuse_preds = fuse_preds / fuse_confs
    
            sample_mean_preds.append(mean_preds)
            sample_fuse_preds.append(fuse_preds)
            sample_fuse_confs.append(fuse_confs)
            sample_targets.append(targets)
        
    
        sample_mean_preds = torch.cat(sample_mean_preds, dim=0)
        sample_fuse_preds = torch.cat(sample_fuse_preds, dim=0)
        sample_fuse_confs = torch.cat(sample_fuse_confs, dim=0)
        sample_targets = torch.cat(sample_targets, dim=0)
        
    
        model_mean_preds.append(sample_mean_preds)
        model_fuse_preds.append(sample_fuse_preds)
        model_fuse_confs.append(sample_fuse_confs)
        model_targets.append(sample_targets)
       
        
        model_mean_preds_ = torch.stack(model_mean_preds, dim=0)
        model_fuse_preds_ = torch.stack(model_fuse_preds, dim=0)
        model_fuse_confs_ = torch.stack(model_fuse_confs, dim=0)
        model_targets_ = torch.stack(model_targets, dim=0)
    
        model_mean_preds_ = torch.mean(model_mean_preds_, dim=0)
        model_fuse_preds_ = torch.sum(model_fuse_preds_ * model_fuse_confs_, dim=0) / torch.sum(model_fuse_confs_, dim=0)
        model_fuse_confs_ = torch.sum(model_fuse_confs_, dim=0)
        model_targets_ = torch.mean(model_targets_, dim=0)
    
        #print(model_mean_preds_.size(), model_fuse_preds_.size(),
        #        model_fuse_confs_.size(), model_targets_.size())
    
    
        model_mean_dists = torch.square(model_mean_preds_ - model_targets_)
        model_fuse_dists = torch.square(model_fuse_preds_ - model_targets_)
        model_baseline = torch.square(model_targets_)
    
        count = torch.numel(model_fuse_confs_)
        [_, conf_indices] = torch.sort(model_fuse_confs_.flatten())
        conf_indices_25 = conf_indices[-int(0.25 * count):]
        conf_indices_50 = conf_indices[-int(0.50 * count):]
        conf_indices_75 = conf_indices[-int(0.75 * count):]
    
     
        mean_mse = model_mean_dists.mean().item()
        fuse_mse = model_fuse_dists.mean().item()
        selected_mse_25 = model_fuse_dists.flatten()[conf_indices_25].mean().item()
        selected_mse_50 = model_fuse_dists.flatten()[conf_indices_50].mean().item()
        selected_mse_75 = model_fuse_dists.flatten()[conf_indices_75].mean().item()
        base_mse = model_baseline.mean().item()
        
    
        print(', '.join([
            'mean_dist: {:8.5f}'.format(mean_mse),
            'fuse_dist: {:8.5f}'.format(fuse_mse),
            'selected_dist_25: {:8.5f}'.format(selected_mse_25),
            'selected_dist_50: {:8.5f}'.format(selected_mse_50),
            'selected_dist_75: {:8.5f}'.format(selected_mse_75),
            'baseline: {:8.5f}'.format(base_mse),
            ]))

