import os
import yaml
import argparse
import collections
import time
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from datetime import datetime
from pprint import pprint
from copy import deepcopy
from runner import IterRunner
from builder import build_dataloader, build_model


def parse_args():
    parser = argparse.ArgumentParser(
            description='A PyTorch project for 3D face reconstruction from voice.')
    parser.add_argument('--config', 
            help='train config file path')
    parser.add_argument('--proj_dir', 
            help='the dir to save logs and models')
    parser.add_argument('--start_time', 
            help='time to start training')
    parser.add_argument('--measure_indices', nargs='+',
            help='time to start training')
    args = parser.parse_args()

    return args


def merge(dict1, dict2):
    ''' Return a new dictionary by merging two dictionaries recursively.
    '''
    result = deepcopy(dict1)
    for key, value in dict2.items():
        if isinstance(value, collections.abc.Mapping):
            result[key] = merge(result.get(key, {}), value)
        else:
            result[key] = deepcopy(dict2[key])
    return result


def fill_config(configs):
    #configs = copy.deepcopy(configs)
    base_cfg = configs.pop('base', {})
    for sub, sub_cfg in configs.items():
        if isinstance(sub_cfg, dict):
            configs[sub] = merge(base_cfg, sub_cfg)
        elif isinstance(sub_cfg, list):
            configs[sub] = [merge(base_cfg, c) for c in sub_cfg]
    return configs

def main_worker(configs):
    # init dataloader
    train_loader = build_dataloader(configs['data']['train'])

    # ,eam and std
    norm_mu_path = os.path.join(
            configs['project']['proj_dir'], 'norm_mu.txt')
    norm_std_path = os.path.join(
            configs['project']['proj_dir'], 'norm_std.txt')
    np.savetxt(norm_mu_path,
            train_loader.dataset.norm_mu, fmt='%.4f')
    np.savetxt(norm_std_path,
            train_loader.dataset.norm_std, fmt='%.4f')
    configs['data']['val']['dataset']['norm_mu_path'] = norm_mu_path
    configs['data']['val']['dataset']['norm_std_path'] = norm_std_path
    configs['data']['eval']['dataset']['norm_mu_path'] = norm_mu_path
    configs['data']['eval']['dataset']['norm_std_path'] = norm_std_path

    val_loader = build_dataloader(configs['data']['val'])
    eval_loader = build_dataloader(configs['data']['eval'])

    # init model
    feat_dim = configs['model']['backbone']['net']['feat_dim']
    configs['model']['head']['net']['input_dim'] = feat_dim
    model = build_model(configs['model'])

    # init runner and run
    runner = IterRunner(configs, train_loader, val_loader, eval_loader, model)
    runner.run()

if __name__ == '__main__':
    # get arguments and configs
    args = parse_args()

    with open(args.config, 'r') as f:
        configs = yaml.load(f, yaml.SafeLoader)

    if args.measure_indices:
        measure_indices = [int(idx) for idx in args.measure_indices]
        configs['data']['base']['dataset']['measure_indices'] = measure_indices

    if configs['data']['base']['dataset']['seed'] < 0:
        configs['data']['base']['dataset']['seed'] = np.random.randint(20000)

    configs['data'] = fill_config(configs['data'])
    configs['model'] = fill_config(configs['model'])

    if args.start_time:
        yy, mm, dd, h, m, s = args.start_time.split('-')
        yy, mm, dd = int(yy), int(mm), int(dd)
        h, m, s = int(h), int(m), int(s)
        start_time = datetime(yy, mm, dd, h, m, s)
        while datetime.now() < start_time:
            print(datetime.now())
            time.sleep(600)

    # project directory
    if args.proj_dir:
        configs['project']['proj_dir'] = args.proj_dir
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    proj_dir = os.path.join(
            configs['project']['proj_dir'], timestamp)
    if not os.path.exists(proj_dir):
        os.makedirs(proj_dir)
    configs['project']['proj_dir'] = proj_dir

    # start multiple processes
    main_worker(configs)
