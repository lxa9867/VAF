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
    test_config['dataset']['ann_path'] = osp.join(proj_dir, 'eval_list.txt')
    test_config['dataset']['split'] = [0,0,1,0]
    test_config['dataset']['norm_type'] = None
    test_loader = build_dataloader(test_config)
    #norm_mu = test_loader.dataset.norm_mu.item()
    #norm_std = test_loader.dataset.norm_std.item()

    print(len(test_loader.dataset))
    for _, voices, targets, _, _ in test_loader:
        #print(targets.item())
        pass
    #print(norm_mu, norm_std)
    asdasddas


    # network
    bkb_config = copy.deepcopy(configs['model']['backbone']['net'])
    head_config = copy.deepcopy(configs['model']['head']['net'])
    bkb = load_net(bkb_config, 'backbone')
    head = load_net(head_config, 'head')

    # eval
    AM_errors = []
    AM_confs = []
    base_errors = []
    for _, voices, targets, _, _ in test_loader:
        assert voices.size(0) == targets.size(0) == 1

        voices = voices.cuda()
        preds, confs = head(bkb(voices))

        fuse_preds = torch.sum(preds * confs, dim=2)
        fuse_confs = torch.sum(confs, dim=2)
        fuse_preds = fuse_preds / fuse_confs

        fuse_confs = torch.mean(confs, dim=2)
        # [fuse_confs, _] = torch.max(confs, dim=2, keepdim=True)

        AM_errors.append(torch.square(fuse_preds.cpu() - targets))
        AM_confs.append(fuse_confs.cpu())
        base_errors.append(torch.square(targets))

    AM_errors = torch.cat(AM_errors, dim=0)
    AM_confs = torch.cat(AM_confs, dim=0)
    base_error = torch.mean(torch.cat(base_errors, dim=0), dim=0)

    count = AM_errors.size(0)
    indices_100 = torch.argsort(AM_confs, dim=0)
    indices_75 = indices_100[-int(0.75*count):]
    indices_50 = indices_100[-int(0.50*count):]

    error_100 = torch.mean(torch.gather(AM_errors, 0, indices_100), dim=0)
    error_75 = torch.mean(torch.gather(AM_errors, 0, indices_75), dim=0)
    error_50 = torch.mean(torch.gather(AM_errors, 0, indices_50), dim=0)

    return torch.stack([error_100, error_75, error_50, base_error])

def get_statistics(data):
    n = len(data)
    mu = data.mean()
    std = data.std()
    t_val = stats.t.ppf(0.95, n-1)
    CI = [mu - t_val * std / np.sqrt(n), mu + t_val * std / np.sqrt(n)]

    return CI, mu, std


nnn = 0
meas_ids = range(16*nnn, 16*(nnn+1), 1)
meas_ids = [79, 89]
path = 'project/anth/sgd_666_l2_conf_f/measid_{}/20220409_22101*'
for n_id, meas_id in enumerate(meas_ids):

    errors = []
    proj_dirs = glob(path.format(meas_id))
    proj_dirs.sort()
    for n_dir, proj_dir in enumerate(proj_dirs):
        print('predicting: AM: {}/{}, proj_dir: {}/{}'.format(
            n_id+1, len(meas_ids), n_dir+1, len(proj_dirs)), end='\r')
        error = get_AM_from_dir(proj_dir)
        errors.append(error)
    errors = torch.stack(errors) # N x 4 x K
    #errors = errors[:, 0:3] / errors[:, 3:4]
    errors = errors[:, 0:3] / torch.mean(errors[:, 3:4], dim=0, keepdim=True)
    errors = errors.numpy()
    
    for k in range(errors.shape[2]):
        CI_100, mu_100, std_100 = get_statistics(errors[:, 0, k])
        CI_75, mu_75, std_75 = get_statistics(errors[:, 1, k])
        CI_50, mu_50, std_50 = get_statistics(errors[:, 2, k])

        print(
            'AM_ID: {:2d}'.format(meas_id),
            'mu100: {:6.4f}'.format(mu_100),
            'std100: {:6.4f}'.format(std_100),
            'CI100: [{:6.4f}, {:6.4f}]'.format(CI_100[0], CI_100[1]),
            'CIu100: {:6.4f}'.format(CI_100[1]),
            'mu75: {:6.4f}'.format(mu_75),
            'std75: {:6.4f}'.format(std_75),
            'CI75: [{:6.4f}, {:6.4f}]'.format(CI_75[0], CI_75[1]),
            'CIu75: {:6.4f}'.format(CI_75[1]),
            'mu50: {:6.4f}'.format(mu_50),
            'std50: {:6.4f}'.format(std_50),
            'CI50: [{:6.4f}, {:6.4f}]'.format(CI_50[0], CI_50[1]),
            'CIu50: {:6.4f}'.format(CI_50[1]),
        )
    print()
