import os
import copy
import warnings

import torch
import torch.nn as nn

from dataset.utils import get_collate_fn
from importlib import import_module


def build_from_cfg(cfg, module):
    """Build a module from config dict.
    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.
    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        raise KeyError(f'`cfg` must contain the key "type", but got {cfg}')

    args = cfg.copy()
    obj_type = args.pop('type')
    if not isinstance(obj_type, str):
        raise TypeError(f'type must be a str, but got {type(obj_type)}')
    else:
        obj_cls = getattr(import_module(module), obj_type)
        if obj_cls is None:
            raise KeyError(f'{obj_type} is not in the {module} module')

    return obj_cls(**args)


def build_dataloader(cfg):
    """
    Args:
        the type of `cfg` could also be a dict for a dataloader,
        or a list or a tuple of dicts for multiple dataloader,
    Returns:
        PyTorch dataloader(s)
    """
    if isinstance(cfg, (list, tuple)):
        return [build_dataloader(c) for c in cfg]
    else:
        if 'dataset' not in cfg:
            raise KeyError(f'`cfg` must contain the key "dataset", but got {cfg}')
        dataset = build_from_cfg(cfg['dataset'], 'dataset')
 
        if 'dataloader' not in cfg:
            raise KeyError(f'`cfg` must contain the key "dataloader", but got {cfg}')
        loader_cfg = copy.deepcopy(cfg['dataloader'])
        loader_cfg['dataset'] = dataset
        if dataset.mode == 'train':
            loader_cfg['collate_fn'] = get_collate_fn(
                    dataset.duration, dataset.sample_rate)
 
        dataloader = build_from_cfg(loader_cfg, 'torch.utils.data')
    
        return dataloader


def build_module(cfg, module):
    if 'net' not in cfg:
        raise KeyError(f'`cfg` must contain the key "net", but got {cfg}')

    net = build_from_cfg(cfg['net'], module)
    net = net.cuda()

    if 'optimizer' not in cfg:
        raise KeyError(f'`cfg` must contain the key "optimizer", but got {cfg}')
    optim_cfg = copy.deepcopy(cfg['optimizer'])
    optim_cfg['params'] = net.parameters()
    optimizer = build_from_cfg(optim_cfg, 'torch.optim')

    if 'scheduler' not in cfg:
        raise KeyError(f'`cfg` must contain the key "dataloader", but got {cfg}')
    sched_cfg = copy.deepcopy(cfg['scheduler'])
    sched_cfg['optimizer'] = optimizer
    scheduler = build_from_cfg(sched_cfg, 'torch.optim.lr_scheduler')
    
    return {'net': net, 'optimizer': optimizer, 'scheduler': scheduler}


def build_model(cfg):
    if 'backbone' not in cfg:
        raise KeyError(f'`cfg` must contain the key "backbone", but got {cfg}')
    if 'head' not in cfg:
        raise KeyError(f'`cfg` must contain the key "head", but got {cfg}')

    model = {}
    for module in cfg:
        model[module] = build_module(cfg[module], f'model.{module}')
    return model

