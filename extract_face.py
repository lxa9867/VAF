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
    'project/20210921_233055',
    'project/20210921_234659',
    'project/20210921_234714',
    'project/20210921_234728',
    'project/20210921_235532',
]

with torch.no_grad():
    for proj_dir in proj_dirs:
        print(proj_dir)

        config_file = osp.join(proj_dir, 'configs.yml')
        with open(config_file, 'r') as f:
            configs = yaml.load(f, yaml.SafeLoader)

        # dataloader
        configs['data']['val']['dataset']['duration'] = [16, 18]
        configs['data']['eval']['dataset']['duration'] = [36, 38]
        train_loader = build_dataloader(configs['data']['train'])
        val_loader = build_dataloader(configs['data']['val'])
        eval_loader = build_dataloader(configs['data']['eval'])

        # gender template
        male_template = []
        female_template = []
        for idx, voices, faces, genders, _ in train_loader:
            for face, gender in zip(faces, genders):
                if gender.item() == 0:
                    male_template.append(face.numpy())
                else:
                    female_template.append(face.numpy())
        male_template = np.array(male_template)
        female_template = np.array(female_template)
        male_template = np.mean(male_template, axis=0)
        female_template = np.mean(female_template, axis=0)

        # network
        file_paths = glob(osp.join(proj_dir, 'models/backbone_*.pth'))
        save_iters = []
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            save_iter = os.path.splitext(filename)[0].split('_')[1]
            save_iters.append(int(save_iter))
        save_iter = str(max(save_iters))

        bkb_path = osp.join(proj_dir, 'models/backbone_{}.pth'.format(str(save_iter)))
        bkb_config = configs['model']['backbone']['net']
        bkb_type = bkb_config.pop('type')
        bkb_cls = getattr(import_module('model.backbone'), bkb_type)
        bkb = bkb_cls(**bkb_config)
        bkb.load_state_dict(torch.load(bkb_path))
        bkb.cuda()
        bkb.eval()

        head_path = osp.join(proj_dir, 'models/head_{}.pth'.format(str(save_iter)))
        head_config = configs['model']['head']['net']
        head_type = head_config.pop('type')
        head_cls = getattr(import_module('model.head'), head_type)
        head = head_cls(**head_config)
        head.load_state_dict(torch.load(head_path))
        head.cuda()
        head.eval()

        result_dir = osp.join(proj_dir, 'result')
        print(result_dir)
        count = 0.
        loss = 0.
        error_map = 0.
        error_map_ = 0.
        for idx, voices, faces, genders, _ in eval_loader:
            voices, faces = voices.cuda(), faces.cuda()
            faces = torch.unsqueeze(faces, -1)
            preds, probs = head(bkb(voices))
            square_dist = torch.sum(torch.square(preds - faces), dim=1)
            count += 1
            loss += torch.mean(square_dist / probs + torch.log(probs)).item()

            # save gt and pred
            face_path = eval_loader.dataset.data_info[idx]['face_path']
            pd_dir = osp.join(result_dir, 'pd_' + str(save_iter))
            pd_path = osp.join(pd_dir, osp.basename(face_path))
            if not os.path.exists(pd_dir):
                os.makedirs(pd_dir)
            pred = torch.squeeze(preds).cpu().numpy()
            np.savetxt(pd_path, pred, fmt='%.4f')

            gt_dir = osp.join(result_dir, 'gt_' + str(save_iter))
            gt_path = osp.join(gt_dir, osp.basename(face_path))
            if not os.path.exists(gt_dir):
                os.makedirs(gt_dir)
            face = torch.squeeze(faces).cpu().numpy()
            np.savetxt(gt_path, face, fmt='%.4f')


            if genders.item() == 0:
                error_map += np.sum(np.square(face - male_template), axis=0)
                error_map_ += np.sum(np.square(face - female_template), axis=0)
            else:
                error_map += np.sum(np.square(face - female_template), axis=0)
                error_map_ += np.sum(np.square(face - male_template), axis=0)

        error_map = error_map / count
        error_map_ = error_map_ / count

        np.savetxt(osp.join(proj_dir, 'error_map.txt'), error_map, fmt='%.4f')
        print('save_iter:', save_iter, 'loss:', loss / count, 'count:', count)
        print(error_map.mean(), error_map_.mean())
