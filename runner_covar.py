import os
import os.path as osp
import time
import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import IterLoader, LoggerBuffer

from torch import distributed as dist
from torch.nn.utils import clip_grad_norm_

#torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.deterministic = True


class IterRunner():
    def __init__(self, configs, train_loader, val_loader, eval_loader, model):
        self.configs = configs
        print('len of dataset:', len(train_loader.dataset),
            len(val_loader.dataset), len(eval_loader.dataset))
        self.train_loader = IterLoader(train_loader)
        self.val_loader = val_loader
        self.eval_loader = eval_loader
        self.model = model
        print('len of dataloader:', len(self.train_loader),
            len(self.val_loader), len(self.eval_loader))

        # meta variables of a new runner
        proj_cfg = configs['project']
        self._iter = 0
        self._max_iters = [max(cfg['scheduler']['milestones']) 
                for cfg in configs['model'].values()]
        self._max_iters = max(self._max_iters)
        self.val_intvl = proj_cfg['val_intvl']
        self.eval_intvl = proj_cfg['eval_intvl']
        self.save_iters = proj_cfg['save_iters']

        # project directory
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        proj_dir = proj_cfg['proj_dir']
        proj_dir = osp.join(proj_dir, timestamp)
        if not osp.exists(proj_dir):
            os.makedirs(proj_dir)
        proj_cfg['proj_dir'] = proj_dir

        # model directory
        self.model_dir = osp.join(proj_dir, proj_cfg['model_dir'])
        if not osp.exists(self.model_dir):
            os.makedirs(self.model_dir)
        proj_cfg['model_dir'] = self.model_dir

        # logger
        train_log_cfg = proj_cfg['train_log']
        train_log_cfg['path'] = osp.join(
                proj_dir, train_log_cfg['path'])
        self.train_buffer = LoggerBuffer(name='train', **train_log_cfg)

        val_log_cfg = proj_cfg['val_log']
        val_log_cfg['path'] = osp.join(
                proj_dir, val_log_cfg['path'])
        self.val_buffer = LoggerBuffer(name='val', **val_log_cfg)

        eval_log_cfg = proj_cfg['eval_log']
        eval_log_cfg['path'] = osp.join(
                proj_dir, eval_log_cfg['path'])
        self.eval_buffer = LoggerBuffer(name='eval', **eval_log_cfg)


        #to avoid duplicated logging info in PyTorch >1.9
        #import logging
        #self.train_buffer.logger.setLevel(logging.WARNING)
        #self.val_buffer.logger.setLevel(logging.WARNING)
        #for name in logging.root.manager.loggerDict:
        #    logger = logging.getLogger(name)
        #    print(name, logger, logger.handlers)
        #xxxx


        # save configs to proj_dir
        config_path = osp.join(proj_dir, proj_cfg['cfg_fname'])
        with open(config_path, 'w') as f:
            yaml.dump(configs, f, sort_keys=False, default_flow_style=None)


    def set_model(self, test_mode):
        for module in self.model:
            if test_mode:
                self.model[module]['net'].eval()
            else:
                self.model[module]['net'].train()
                self.model[module]['optimizer'].zero_grad()

    def update_model(self):
        lrs = []
        for module in self.model:
            self.model[module]['optimizer'].step()
            self.model[module]['scheduler'].step()
            lrs.extend(self.model[module]['scheduler'].get_last_lr())

        if getattr(self, 'current_lrs', None) != lrs:
            self.current_lrs = lrs
            lr_msg = ', '.join(
                    ['{:3.5f}'.format(lr) for lr in self.current_lrs])
            self.train_buffer.logger.info(
                    'Lrs are changed to {}'.format(lr_msg))

    def save_model(self):
        for module in self.model:
            model_name = '{}_{}.pth'.format(str(module), str(self._iter))
            model_path = osp.join(self.model_dir, model_name)
            torch.save(self.model[module]['net'].state_dict(), model_path)

    def train(self):
        self.set_model(test_mode=False)

        # data
        idx, voices, faces, genders, _ = next(self.train_loader)
        voices, faces = voices.cuda(), faces.cuda()
        faces = torch.unsqueeze(faces, -1)
        genders = genders.cuda()

        # forward
        feats = self.model['backbone']['net'](voices)
        outs = self.model['head']['net'](feats)
        loss = F.cross_entropy(outs, genders)
        preds = outs.argmax(dim=1, keepdim=True)
        acc = preds.eq(genders.view_as(preds)).sum() / genders.size(0)

        loss.backward()
        b_grad = clip_grad_norm_(
                self.model['backbone']['net'].parameters(),
                max_norm=1., norm_type=2)
        h_grad = clip_grad_norm_(
                self.model['head']['net'].parameters(),
                max_norm=1., norm_type=2)
        self.update_model()

        # logging and update meters
        msg = {
            'Iter': self._iter,
            'Loss': loss.item(),
            'Acc': acc.item(),
            'bkb_grad': b_grad,
            'head_grad': h_grad,
        }
        self.train_buffer.update(msg)

    @torch.no_grad()
    def val(self,):
        self.set_model(test_mode=True)

        count = 0.
        loss = 0.
        correct = 0.
        for idx, voices, faces, genders, _ in self.val_loader:
            voices, faces = voices.cuda(), faces.cuda()
            faces = torch.unsqueeze(faces, -1)
            genders = genders.cuda()

            feats = self.model['backbone']['net'](voices)
            outs = self.model['head']['net'](feats)
            preds = outs.argmax(dim=1, keepdim=True)
            count += voices.size(0)
            loss += F.cross_entropy(outs, genders)
            correct += preds.eq(genders.view_as(preds)).sum()

        # logging and update meters
        msg = {
            'Iter': self._iter,
            'Loss': loss.item() / count,
            'Acc': correct.item() / count,
        }
        self.val_buffer.update(msg)

    @torch.no_grad()
    def eval(self,):
        self.set_model(test_mode=True)

        count = 0.
        loss = 0.
        correct = 0.
        for idx, voices, faces, genders, _ in self.eval_loader:
            voices, faces = voices.cuda(), faces.cuda()
            faces = torch.unsqueeze(faces, -1)
            genders = genders.cuda()

            feats = self.model['backbone']['net'](voices)
            outs = self.model['head']['net'](feats)
            preds = outs.argmax(dim=1, keepdim=True)
            count += voices.size(0)
            loss += F.cross_entropy(outs, genders)
            correct += preds.eq(genders.view_as(preds)).sum()

        # logging and update meters
        msg = {
            'Iter': self._iter,
            'Loss': loss / count,
            'Acc': correct / count,
        }
        self.eval_buffer.update(msg)

    def run(self):
        while self._iter <= self._max_iters:
            # train step
            if self._iter % self.val_intvl == 0:
                self.val()
            if self._iter % self.eval_intvl == 0:
                self.eval()

            if self._iter in self.save_iters:
                self.save_model()

            self.train()
            self._iter += 1

