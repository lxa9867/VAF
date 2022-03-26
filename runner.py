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
        self.val_errs = []
        self.lowest_err = 100.

        # model directory
        proj_dir = proj_cfg['proj_dir']
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
            model_name = '{}.pth'.format(str(module))
            model_path = osp.join(self.model_dir, model_name)
            torch.save(self.model[module]['net'].state_dict(), model_path)

    def train(self):
        idx, voices, targets, _, _ = next(self.train_loader)
        voices, targets = voices.cuda(), targets.cuda()
        targets = torch.unsqueeze(targets, -1)

        # forward
        self.set_model(test_mode=False)
        feats = self.model['backbone']['net'](voices)
        preds, confs = self.model['head']['net'](feats)

        if self.train_loader.dataset.norm_type == 'l1':
            dist = torch.abs(preds - targets)
            mean_preds = torch.mean(preds, dim=2, keepdim=True)
            mean_dist = torch.abs(mean_preds - targets)
            baseline_dist = torch.abs(targets)
        elif self.train_loader.dataset.norm_type == 'l2':
            dist = torch.square(preds - targets)
            mean_preds = torch.mean(preds, dim=2, keepdim=True)
            mean_dist = torch.square(mean_preds - targets)
            baseline_dist = torch.square(targets)
        else:
            error('unknown norm type')

        with torch.no_grad():
            baseline_dist = torch.mean(baseline_dist)
            mean_dist = torch.mean(mean_dist)

        loss = torch.mean(dist * confs - torch.log(confs))

        # backward abd update model
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
            'Baseline': baseline_dist.item(),
            'Mean_Dist': mean_dist.item(),
            'bkb_grad': b_grad,
            'head_grad': h_grad,
        }
        self.train_buffer.update(msg)

    @torch.no_grad()
    def val(self,):
        self.set_model(test_mode=True)

        all_baseline_dist = []
        all_fuse_dist = []
        all_fuse_conf = []
        loss = 0.
        count = 0.
        for idx, voices, targets, _, _ in self.val_loader:
            voices, targets = voices.cuda(), targets.cuda()
            targets = torch.unsqueeze(targets, -1)

            feats = self.model['backbone']['net'](voices)
            preds, confs = self.model['head']['net'](feats)

            fuse_preds = torch.sum(preds * confs, dim=2, keepdim=True)
            fuse_confs = torch.sum(confs, dim=2, keepdim=True)
            fuse_preds = fuse_preds / fuse_confs
            if self.val_loader.dataset.norm_type == 'l1':
                dist = torch.abs(preds - targets)
                baseline_dist = torch.abs(targets)
                fuse_dist = torch.abs(fuse_preds - targets)
            elif self.val_loader.dataset.norm_type == 'l2':
                dist = torch.square(preds - targets)
                baseline_dist = torch.square(targets)
                fuse_dist = torch.square(fuse_preds - targets)
            else:
                error('unknown norm type')

            all_baseline_dist.append(baseline_dist)
            all_fuse_dist.append(fuse_dist)
            all_fuse_conf.append(fuse_confs)

            loss += torch.mean(dist * confs - torch.log(confs)).item()
            count += voices.size(0)

        all_baseline_dist = torch.cat(all_baseline_dist, dim=0)
        all_fuse_dist = torch.cat(all_fuse_dist, dim=0)
        all_fuse_conf = torch.cat(all_fuse_conf, dim=0)
        [_, conf_indices] = torch.sort(all_fuse_conf.flatten())
        conf_indices = conf_indices[int(torch.numel(all_fuse_conf)//2):]

        # logging and update meters
        msg = {
            'Iter': self._iter,
            'Loss': loss / count,
            'Baseline': all_baseline_dist.mean().item(),
            'Fuse_Dist': all_fuse_dist.mean().item(),
            'Selected': all_fuse_dist.flatten()[conf_indices].mean().item(),
        }
        self.val_buffer.update(msg)
        self.val_errs.append(all_fuse_dist.mean().item())

        #
        proj_dir = self.configs['project']['proj_dir']
        line = (torch.mean(torch.squeeze(all_baseline_dist, 2), 0).tolist()
                + torch.mean(torch.squeeze(all_fuse_dist, 2), 0).tolist()
                + torch.mean(torch.squeeze(all_fuse_conf, 2), 0).tolist())
        line = ['{:5d}'.format(self._iter)] + ['{:.5f}'.format(digit) for digit in line]
        with open(osp.join(proj_dir, 'val_pred.txt'), 'a') as f:
            f.write(' '.join(line) + '\n')


    @torch.no_grad()
    def eval(self,):
        self.set_model(test_mode=True)

        all_baseline_dist = []
        all_fuse_dist = []
        all_fuse_conf = []
        loss = 0.
        count = 0.
        for idx, voices, targets, _, _ in self.eval_loader:
            voices, targets = voices.cuda(), targets.cuda()
            targets = torch.unsqueeze(targets, -1)

            feats = self.model['backbone']['net'](voices)
            preds, confs = self.model['head']['net'](feats)

            fuse_preds = torch.sum(preds * confs, dim=2, keepdim=True)
            fuse_confs = torch.sum(confs, dim=2, keepdim=True)
            fuse_preds = fuse_preds / fuse_confs
            if self.val_loader.dataset.norm_type == 'l1':
                dist = torch.abs(preds - targets)
                baseline_dist = torch.abs(targets)
                fuse_dist = torch.abs(fuse_preds - targets)
            elif self.val_loader.dataset.norm_type == 'l2':
                dist = torch.square(preds - targets)
                baseline_dist = torch.square(targets)
                fuse_dist = torch.square(fuse_preds - targets)
            else:
                error('unknown norm type')

            all_baseline_dist.append(baseline_dist)
            all_fuse_dist.append(fuse_dist)
            all_fuse_conf.append(fuse_confs)

            loss += torch.mean(dist * confs - torch.log(confs)).item()
            count += voices.size(0)

        all_baseline_dist = torch.cat(all_baseline_dist, dim=0)
        all_fuse_dist = torch.cat(all_fuse_dist, dim=0)
        all_fuse_conf = torch.cat(all_fuse_conf, dim=0)
        [_, conf_indices] = torch.sort(all_fuse_conf.flatten())
        conf_indices = conf_indices[int(torch.numel(all_fuse_conf)//2):]

        # logging and update meters
        msg = {
            'Iter': self._iter,
            'Loss': loss / count,
            'Baseline': all_baseline_dist.mean().item(),
            'Fuse_Dist': all_fuse_dist.mean().item(),
            'Selected': all_fuse_dist.flatten()[conf_indices].mean().item(),
        }
        self.eval_buffer.update(msg)

        #
        proj_dir = self.configs['project']['proj_dir']
        line = (torch.mean(torch.squeeze(all_baseline_dist, 2), 0).tolist()
                + torch.mean(torch.squeeze(all_fuse_dist, 2), 0).tolist()
                + torch.mean(torch.squeeze(all_fuse_conf, 2), 0).tolist())
        line = ['{:5d}'.format(self._iter)] + ['{:.5f}'.format(digit) for digit in line]
        with open(osp.join(proj_dir, 'eval_pred.txt'), 'a') as f:
            f.write(' '.join(line) + '\n')


    def run(self):
        while self._iter <= self._max_iters:
            # train step
            if self._iter % self.val_intvl == 0:
                self.val()
            if self._iter % self.eval_intvl == 0:
                self.eval()

            #print(curr_err, self.lowest_err)
            if len(self.val_errs) > 5 and self.val_errs[-1] < self.lowest_err:
                self.lowest_err = self.val_errs[-1]
                # self.save_model()
            
            self.train()
            self._iter += 1

