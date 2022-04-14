import os
import os.path as osp
import numpy as np
import yaml
import copy
import math
import random

import torch

from glob import glob
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


@torch.no_grad()
def get_AM_from_dir(proj_dir):
    config_path = osp.join(proj_dir, 'configs.yml')
    with open(config_path, 'r') as f:
        configs = yaml.load(f, yaml.SafeLoader)

    # data 
    test_config = copy.deepcopy(configs['data']['eval'])
    test_loader = build_dataloader(test_config)
    norm_mu = test_loader.dataset.norm_mu.item()
    norm_std = test_loader.dataset.norm_std.item()

    # network
    bkb_config = copy.deepcopy(configs['model']['backbone']['net'])
    head_config = copy.deepcopy(configs['model']['head']['net'])
    bkb = load_net(bkb_config, 'backbone')
    head = load_net(head_config, 'head')

    # eval
    AM_confs = []
    for _, voices, targets, _, _ in test_loader:
        assert voices.size(0) == targets.size(0) == 1

        voices = voices.cuda()
        preds, confs = head(bkb(voices))
        fuse_confs = torch.mean(confs, dim=2, keepdim=True)
        # [fuse_confs, _] = torch.max(confs, dim=2, keepdim=True)

        AM_confs.append(fuse_confs.item())

    return np.array(AM_confs)

nnn = 0
meas_ids = range(0, 96, 1)
path = 'project/anth/sgd_666_l2_conf_f/measid_{}/2022*'
for n_id, meas_id in enumerate(meas_ids):

    proj_dirs = glob(path.format(meas_id))
    for n_dir, proj_dir in enumerate(proj_dirs):
        print('predicting: AM: {}/{}, proj_dir: {}/{}'.format(
            n_id+1, len(meas_ids), n_dir+1, len(proj_dirs)), end='\r')

        eval_confs = get_AM_from_dir(proj_dir)
        np.savetxt(
            osp.join(proj_dir, 'eval_confs.txt'), eval_confs,
        )
