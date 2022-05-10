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
    test_config['dataset']['mode'] = 'test'
    test_config['dataset']['ann_path'] = osp.join(proj_dir, 'test_list.txt')
    test_config['dataset']['split'] = [0,0,0,1]
    test_loader = build_dataloader(test_config)
    norm_mu = torch.from_numpy(test_loader.dataset.norm_mu).view(1, -1)
    norm_std = torch.from_numpy(test_loader.dataset.norm_std).view(1, -1)

    # network
    bkb_config = copy.deepcopy(configs['model']['backbone']['net'])
    head_config = copy.deepcopy(configs['model']['head']['net'])
    bkb = load_net(bkb_config, 'backbone')
    head = load_net(head_config, 'head')

    # eval
    AMs = []
    AM_confs = []
    AM_gts = []
    AM_means = []
    for _, voices, targets, _, _ in test_loader:
        assert voices.size(0) == targets.size(0) == 1

        voices = voices.cuda()
        preds, confs = head(bkb(voices))

        fuse_preds = torch.sum(preds * confs, dim=2)
        fuse_confs = torch.sum(confs, dim=2)
        fuse_preds = fuse_preds / fuse_confs

        fuse_confs = torch.mean(confs, dim=2)
        # [fuse_confs, _] = torch.max(confs, dim=2, keepdim=True)

        AMs.append(norm_mu + fuse_preds.cpu() * norm_std)
        AM_confs.append(fuse_confs.cpu())
        AM_gts.append(norm_mu + targets * norm_std)
        AM_means.append(norm_mu)

    AMs = torch.cat(AMs, dim=0)
    AM_confs = torch.cat(AM_confs, dim=0)
    AM_gts = torch.cat(AM_gts, dim=0)
    AM_means = torch.cat(AM_means, dim=0)

    return torch.stack([AMs, AM_confs, AM_gts, AM_means])

path = 'project/anths/sgd_10086_l2_conf_m/2022*'
proj_dirs = glob(path)
for n_dir, proj_dir in enumerate(proj_dirs):
    print('predicting: proj_dir: {}/{}'.format(n_dir+1, len(proj_dirs)), end='\r')
    prediction = get_AM_from_dir(proj_dir)
    np.save(osp.join(proj_dir, 'prediction'), prediction.numpy())

predictions = []
for n_dir, proj_dir in enumerate(proj_dirs):
    prediction = np.load(osp.join(proj_dir, 'prediction.npy'))
    predictions.append(prediction)
predictions = np.array(predictions) # #exp x 4 x #sample x #AM

errors_100 = []
errors_75 = []
errors_50 = []
for n in range(predictions.shape[0]):
    AMs = np.sum(predictions[:n+1, 0, :] * predictions[:n+1, 1, :], axis=0) / np.sum(predictions[:n+1, 1, :], axis=0)
    AM_confs = np.mean(predictions[:n+1, 1, :], axis=0)
    AM_gts = np.mean(predictions[:n+1, 2, :], axis=0)
    AM_means = np.mean(predictions[:n+1, 3, :], axis=0)

    count = AM_confs.shape[0]
    index_100 = np.argsort(np.mean(AM_confs, axis=1))
    index_75 = index_100[-int(0.75*count):]
    index_50 = index_100[-int(0.50*count):]

    err_base = np.mean(np.square(AM_means - AM_gts), axis=0)
    diff = np.square(AMs - AM_gts)
    err_100 = np.mean(diff[index_100], axis=0)
    err_75 = np.mean(diff[index_75], axis=0)
    err_50 = np.mean(diff[index_50], axis=0)
    #err_100 = np.mean(np.take_along_axis(diff, index_100, axis=0), axis=0)
    #err_75 = np.mean(np.take_along_axis(diff, index_75, axis=0), axis=0)
    #err_50 = np.mean(np.take_along_axis(diff, index_50, axis=0), axis=0)

    errors_100.append(err_100/err_base)
    errors_75.append(err_75/err_base)
    errors_50.append(err_50/err_base)

np.savetxt(
    osp.join(osp.dirname(path), 'errors_100.txt'),
    np.array(errors_100), fmt='%8.5f',
)
np.savetxt(
    osp.join(osp.dirname(path), 'errors_75.txt'),
    np.array(errors_75), fmt='%8.5f',
)
np.savetxt(
    osp.join(osp.dirname(path), 'errors_50.txt'),
    np.array(errors_50), fmt='%8.5f',
)
# 4 x #sample x #AM
np.save(
    osp.join(osp.dirname(path), 'predictions'),
    np.array([AMs, AM_confs, AM_gts, AM_means]),
)

print(path)
for k in range(len(errors_100[0])):
    print(
        'k: {:2d}'.format(k),
        'mse_100: {:8.5f}'.format(errors_100[-1][k].item()),
        'mse_75: {:8.5f}'.format(errors_75[-1][k].item()),
        'mse_50: {:8.5f}'.format(errors_50[-1][k].item()),
    )
