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
    norm_mu = test_loader.dataset.norm_mu.item()
    norm_std = test_loader.dataset.norm_std.item()

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

        fuse_preds = torch.sum(preds * confs, dim=2, keepdim=True)
        fuse_confs = torch.sum(confs, dim=2, keepdim=True)
        fuse_preds = fuse_preds / fuse_confs

        fuse_confs = torch.mean(confs, dim=2, keepdim=True)
        # [fuse_confs, _] = torch.max(confs, dim=2, keepdim=True)

        AMs.append(norm_mu + fuse_preds.item() * norm_std)
        AM_confs.append(fuse_confs.item())
        AM_gts.append(norm_mu + targets.item() * norm_std)
        AM_means.append(norm_mu)

    return np.array([AMs, AM_confs, AM_gts, AM_means])

nnn = 3
meas_ids = range(24*nnn, 24*(nnn+1), 1)
path = 'project/anth/sgd_666_l2_conf_f/measid_{}/2022*'
for n_id, meas_id in enumerate(meas_ids):

    proj_dirs = glob(path.format(meas_id))
    for n_dir, proj_dir in enumerate(proj_dirs):
        print('predicting: AM: {}/{}, proj_dir: {}/{}'.format(
            n_id+1, len(meas_ids), n_dir+1, len(proj_dirs)), end='\r')

        prediction = get_AM_from_dir(proj_dir)
        np.savetxt(
            osp.join(proj_dir, 'prediction.txt'), prediction.T,
        )

    predictions = []
    for n_dir, proj_dir in enumerate(proj_dirs):
        prediction = np.loadtxt(osp.join(proj_dir, 'prediction.txt')).T
        predictions.append(prediction)
    predictions = np.array(predictions)

    errors = []
    for i in range(predictions.shape[0]):
        AMs = np.sum(predictions[:i+1, 0, :] * predictions[:i+1, 1, :], axis=0) / np.sum(predictions[:i+1, 1, :], axis=0)
        AM_confs = np.mean(predictions[:i+1, 1, :], axis=0)
        AM_gts = np.mean(predictions[:i+1, 2, :], axis=0)
        AM_means = np.mean(predictions[:i+1, 3, :], axis=0)

        index = np.argsort(AM_confs)
        index_75 = index[-int(0.75*len(index)):]
        index_50 = index[-int(0.50*len(index)):]
        index_25 = index[-int(0.25*len(index)):]

        err_mean = np.mean(np.square(AM_means - AM_gts))
        err = np.mean(np.square(AMs - AM_gts))
        err_75 = np.mean(np.square(AMs[index_75] - AM_gts[index_75]))
        err_50 = np.mean(np.square(AMs[index_50] - AM_gts[index_50]))
        err_25 = np.mean(np.square(AMs[index_25] - AM_gts[index_25]))
        errors.append([err/err_mean, err_75/err_mean, err_50/err_mean, err_25/err_mean])

    np.savetxt(
        osp.join(osp.dirname(path.format(meas_id)), 'errors.txt'),
        np.array(errors), fmt='%8.5f',
    )
    np.savetxt(
        osp.join(osp.dirname(path.format(meas_id)), 'predictions.txt'),
        np.array([AMs, AM_confs, AM_gts, AM_means]).T,
    )

    print(path.format(meas_id))
    print(
        'AM_ID: {:2d}'.format(meas_id),
        'mse: {:8.5f}'.format(errors[-1][0].item()),
        'mse_75: {:8.5f}'.format(errors[-1][1].item()),
        'mse_50: {:8.5f}'.format(errors[-1][2].item()),
        'mse_25: {:8.5f}'.format(errors[-1][3].item()),
    )
    print()
