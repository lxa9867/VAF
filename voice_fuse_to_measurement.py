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
model_fuse_preds = []
model_fuse_confs = []
model_fuse_tgz = []
for idx, proj_dir in enumerate(proj_dirs):
    print(proj_dir)
    config_file = osp.join(proj_dir, 'configs.yml')
    with open(config_file, 'r') as f:
        configs = yaml.load(f, yaml.SafeLoader)

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
    all_fuse_pred = []
    all_fuse_conf = []
    all_fuse_tgz = []
    loss = 0.
    count = 0.
    for idx, voices, targets, _, _ in test_loader:
        voices, targets = voices.cuda(), targets.cuda()
        targets = torch.unsqueeze(targets, -1)

        preds, confs = head(bkb(voices))

        if test_loader.dataset.norm_type == 'l2':
            mean_preds = torch.mean(preds, dim=2, keepdim=True)
            fuse_preds = torch.sum(preds * confs, dim=2, keepdim=True)
            fuse_confs = torch.sum(confs, dim=2, keepdim=True)
            fuse_preds = fuse_preds / fuse_confs
        else:
            error('unknown norm type')

        all_fuse_pred.append(fuse_preds.item())
        all_fuse_conf.append(fuse_confs.item())
        all_fuse_tgz.append(targets.item())

    model_fuse_preds.append(all_fuse_pred)
    model_fuse_confs.append(all_fuse_conf)
    model_fuse_tgz.append(all_fuse_tgz)


model_fuse_preds = np.array(model_fuse_preds)
model_fuse_confs = np.array(model_fuse_confs)
model_fuse_tgz = np.array(model_fuse_tgz)
model_fuse_preds = np.sum(model_fuse_preds * model_fuse_confs, axis=0) / np.sum(model_fuse_confs, axis=0)
print(np.mean(np.square(model_fuse_preds - model_fuse_tgz[0])))






