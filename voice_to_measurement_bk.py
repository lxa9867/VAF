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


def fuse_prediction():

    return fuse_pred

# hfn v0 v1 v2
proj_dirs = glob("project/sgd_l2_vfn_12/2022*")
proj_dirs.sort()
for idx, proj_dir in enumerate(proj_dirs):
    config_file = osp.join(proj_dir, 'configs.yml')
    with open(config_file, 'r') as f:
        configs = yaml.load(f, yaml.SafeLoader)

    # data 
    val_config = copy.deepcopy(configs['data']['val'])
    val_loader = build_dataloader(val_config)
    # print(val_loader)
    eval_config = copy.deepcopy(configs['data']['eval'])
    eval_loader = build_dataloader(eval_config)
    # print(eval_loader)

    test_config = copy.deepcopy(configs['data']['eval'])
    test_config['dataset']['mode'] = 'test'
    test_loader = build_dataloader(test_config)

    # network
    bkb_config = copy.deepcopy(configs['model']['backbone']['net'])
    # print(bkb_config)
    bkb = load_net(bkb_config, 'backbone')
    head_config = copy.deepcopy(configs['model']['head']['net'])
    # print(head_config)
    head = load_net(head_config, 'head')

    # eval
    tot_mean_dist = 0.
    tot_fuse_dist = 0.
    tot_baseline_dist = 0.
    all_fuse_dist = []
    all_fuse_conf = []
    loss = 0.
    count = 0.
    for idx, voices, targets, _, _ in test_loader:
        voices, targets = voices.cuda(), targets.cuda()
        targets = torch.unsqueeze(targets, -1)

        preds, confs = head(bkb(voices))

        if test_loader.dataset.norm_type == 'l2':
            mean_preds = torch.mean(preds, dim=2, keepdim=True)
            mean_dist = torch.square(mean_preds - targets)
            fuse_preds = torch.sum(preds * confs, dim=2, keepdim=True)
            fuse_confs = torch.sum(confs, dim=2, keepdim=True)
            fuse_preds = fuse_preds / fuse_confs
            fuse_dist = torch.square(fuse_preds - targets)
            baseline_dist = torch.square(targets)
        else:
            error('unknown norm type')

        tot_mean_dist += torch.mean(mean_dist).item()
        tot_fuse_dist += torch.mean(fuse_dist).item()
        tot_baseline_dist += torch.mean(baseline_dist).item()
        all_fuse_dist.append(fuse_dist)
        fuse_confs = torch.mean(confs, dim=2, keepdim=True)
        all_fuse_conf.append(fuse_confs)
        
        count += 1

        ''' 
        save_dir = proj_dir + '/results/eval/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fileid = os.path.basename(eval_loader.dataset.data_info[idx]['voice_path'])[:-4]
        pred_path = os.path.join(save_dir, fileid + '_pred_{:.5f}.txt'.format(targets.item()))
        conf_path = os.path.join(save_dir, fileid + '_conf.txt')
        np.savetxt(pred_path, preds.flatten().detach().cpu().numpy(), fmt='%8.5f')
        np.savetxt(conf_path, confs.flatten().detach().cpu().numpy(), fmt='%8.5f')
        '''

    all_fuse_dist = torch.cat(all_fuse_dist, dim=0)
    all_fuse_conf = torch.cat(all_fuse_conf, dim=0)
    [_, conf_indices] = torch.sort(all_fuse_conf.flatten())
    conf_indices_25 = conf_indices[-int(0.25 * count):]
    conf_indices_50 = conf_indices[-int(0.50 * count):]
    conf_indices_75 = conf_indices[-int(0.75 * count):]
    
    print('mean_dist: {:8.5f}, fuse_dist: {:8.5f}, selected_dist_25: {:8.5f}, selected_dist_50: {:8.5f}, selected_dist_75: {:8.5f}, baseline: {:8.5f},'.format(
        tot_mean_dist / count, tot_fuse_dist / count, 
        all_fuse_dist.flatten()[conf_indices_25].mean().item(),
        all_fuse_dist.flatten()[conf_indices_50].mean().item(),
        all_fuse_dist.flatten()[conf_indices_75].mean().item(),
        tot_baseline_dist / count))

    # xxxx




