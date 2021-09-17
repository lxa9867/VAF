import os
import os.path as osp
import numpy as np
import yaml
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from glob import glob
from dataset import PenstateDataset
from builder import build_dataloader
from importlib import import_module

# hfn v0 v1 v2
proj_dirs = [
    #'project/20210914_064008',
    #'project/20210914_064016',
    #'project/20210914_064451',
    #'project/20210914_064527',
    #'project/20210914_064557',
    #'project/20210914_100700',
    #'project/20210914_134047',
    #'project/20210914_134614',
    #'project/20210914_151858',
    #'project/20210914_154146',

    'project/20210915_004420',
    'project/20210915_012235',
    'project/20210915_012620',
    'project/20210915_012649',
]

#proj_dirs = glob('project/ms1m/sfn/20210*')
with torch.no_grad():
    for proj_dir in proj_dirs:
        print(proj_dir)

        config_file = osp.join(proj_dir, 'configs.yml')
        with open(config_file, 'r') as f:
            configs = yaml.load(f, yaml.SafeLoader)
    
        # dataloader
        configs['data']['val']['dataset']['duration'] = [16, 18]
        val_loader = build_dataloader(configs['data']['val'])
        
        # network
        save_iters = configs['project']['save_iters']
        bkb_paths = [
                osp.join(proj_dir, 'models/backbone_{}.pth'.format(str(save_iter)))
                for save_iter in save_iters]
         
        bkb_config = configs['model']['backbone']['net']
        bkb_type = bkb_config.pop('type')
        bkb_cls = getattr(import_module('model.backbone'), bkb_type)
        head_config = configs['model']['head']['net']
        head_type = head_config.pop('type')
        head_cls = getattr(import_module('model.head'), head_type)
    
        bkbs = []
        heads = []
        for save_iter in save_iters:
            bkb_path = osp.join(proj_dir, 'models/backbone_{}.pth'.format(str(save_iter)))
            bkb = bkb_cls(**bkb_config)
            bkb.load_state_dict(torch.load(bkb_path))
            bkb.cuda()
            bkb.eval()
            bkbs.append(bkb)
    
            head_path = osp.join(proj_dir, 'models/head_{}.pth'.format(str(save_iter)))
            head = head_cls(**head_config)
            head.load_state_dict(torch.load(head_path))
            head.cuda()
            head.eval()
            heads.append(head)
    
        result_dir = osp.join(proj_dir, 'results')
        mean_path = osp.join(result_dir, 'mean.qls')
        std_path = osp.join(result_dir, 'std.qls')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        np.savetxt(mean_path, val_loader.dataset.face_mean, fmt='%.5f')
        np.savetxt(std_path, val_loader.dataset.face_std, fmt='%.5f')
    
        for bkb, head, save_iter in zip(bkbs, heads, save_iters):
            count = 0.
            loss = 0.
            for idx, voices, faces in val_loader:
                face_path = val_loader.dataset.test_info[idx]['face_path']
                voices, faces = voices.cuda(), faces.cuda()
                faces = torch.unsqueeze(faces, -1)
                preds, probs = head(bkb(voices))
                square_dist = torch.sum(torch.square(preds - faces), dim=1)
                count += 1
                loss += torch.mean(square_dist / probs + torch.log(probs)).item()

                # save gt and pred
                pd_dir = osp.join(result_dir, 'pd_' + str(save_iter))
                pd_path = osp.join(pd_dir, osp.basename(face_path))
                if not os.path.exists(pd_dir):
                    os.makedirs(pd_dir)
                pred = torch.squeeze(preds).cpu().numpy()
                np.savetxt(pd_path, pred, fmt='%.5f')
    
                gt_dir = osp.join(result_dir, 'gt_' + str(save_iter))
                gt_path = osp.join(gt_dir, osp.basename(face_path))
                if not os.path.exists(gt_dir):
                    os.makedirs(gt_dir)
                face = torch.squeeze(faces).cpu().numpy()
                np.savetxt(gt_path, face, fmt='%.5f')
    
            print('save_iter:', save_iter, 'loss:', loss / count, 'count:', count)
