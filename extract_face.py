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
    #'project/20211008_233706',
    #'project/20211008_233719',
    #'project/20211008_233739',
    #'project/20211010_084913',
    #'project/20211010_091147',
    #'project/20211010_091627',
    #'project/20211010_103442',
    #'project/20211010_103453',
    #'project/20211010_104522',
    'project/20211011_213008',
    'project/20211011_213021',
    'project/20211011_213032',
    'project/20211011_214310',
    'project/20211011_214316',
    'project/20211011_220832',
]

with torch.no_grad():
    for proj_dir in proj_dirs:

        config_file = osp.join(proj_dir, 'configs.yml')
        with open(config_file, 'r') as f:
            configs = yaml.load(f, yaml.SafeLoader)
        print(proj_dir,
              configs['data']['train']['dataset']['seed'],
              configs['model']['head']['net'],
              configs['model']['head']['optimizer'])

        # dataloader
        configs['data']['val']['dataset']['duration'] = [34, 35]
        configs['data']['eval']['dataset']['duration'] = [34, 35]
        #configs['data']['eval']['dataset']['duration'] = [16, 18]
        configs['data']['train']['dataloader'] = {
            'type': 'DataLoader',
            'batch_size': 1,
            'num_workers': 1, 
            'pin_memory': False,
            'shuffle': False,
            'drop_last': False,
        }
        train_loader = build_dataloader(configs['data']['train'])
        val_loader = build_dataloader(configs['data']['val'])
        eval_loader = build_dataloader(configs['data']['eval'])

        # gender template
        male_template = []
        female_template = []
        for idx, voices, faces, genders, _ in train_loader:
            for face, gender in zip(faces, genders):
                if gender.item() == 1:
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
        count = 0.
        loss = 0.
        loss_w_conf = 0.
        error_map = 0.
        error_map_ = 0.
        errors = []
        confidences = []
        for idx, voices, faces, genders, _ in eval_loader:
            count += 1

            voices, faces = voices.cuda(), faces.cuda()
            faces = torch.unsqueeze(faces, -1)
            preds, confs = head(bkb(voices))
            fuse_confs = torch.sum(confs, dim=3, keepdim=True)
            fuse_preds = torch.sum(preds * confs, dim=3, keepdim=True) / fuse_confs
            square_dist = torch.sum(torch.square(fuse_preds - faces), dim=1)
            loss += torch.mean(square_dist).item()
            index = torch.argsort(fuse_confs.flatten())
            loss_w_conf += torch.mean(square_dist.flatten()[index[-1000:]]).item()

            error = torch.sum(torch.square(preds - faces), dim=1)
            errors.append(torch.squeeze(error).t())
            confidences.append(torch.squeeze(confs).t())

            # save gt and pred
            face_path = eval_loader.dataset.data_info[idx]['face_path']
            pd_dir = osp.join(result_dir, 'pd_' + str(save_iter))
            pd_path = osp.join(pd_dir, osp.basename(face_path))
            if not os.path.exists(pd_dir):
                os.makedirs(pd_dir)
            fuse_pred = torch.squeeze(fuse_preds).cpu().numpy()
            np.savetxt(pd_path, fuse_pred.T, fmt='%.4f')

            gt_dir = osp.join(result_dir, 'gt_' + str(save_iter))
            gt_path = osp.join(gt_dir, osp.basename(face_path))
            if not os.path.exists(gt_dir):
                os.makedirs(gt_dir)
            face = torch.squeeze(faces).cpu().numpy()
            np.savetxt(gt_path, face.T, fmt='%.4f')

            conf_dir = osp.join(result_dir, 'conf_' + str(save_iter))
            conf_path = osp.join(conf_dir, osp.basename(face_path))
            if not os.path.exists(conf_dir):
                os.makedirs(conf_dir)
            fuse_confs = torch.squeeze(fuse_confs).cpu().numpy()
            np.savetxt(conf_path, fuse_confs.T, fmt='%.4f')


            if genders.item() == 1:
                error_map += np.sum(np.square(face - male_template), axis=0)
                error_map_ += np.sum(np.square(face - female_template), axis=0)
            else:
                error_map += np.sum(np.square(face - female_template), axis=0)
                error_map_ += np.sum(np.square(face - male_template), axis=0)

        errors = torch.cat(errors, dim=0).t()
        confidences = torch.cat(confidences, dim=0).t()
        print(errors.size(), confidences.size())
        errors = errors - torch.mean(errors, dim=1, keepdim=True)
        confidences = confidences - torch.mean(confidences, dim=1, keepdim=True)
        corr = torch.sum(errors * confidences, dim=1)
        corr = corr / torch.sqrt(torch.sum(errors * errors, dim=1) * torch.sum(confidences * confidences, dim=1))
        # sorted_corr, indices = torch.sort(corr)
        corr = corr.detach().cpu().numpy()
        np.savetxt(osp.join(proj_dir, 'corr_coeff.txt'), corr, fmt='%.4f')
        

        error_map = error_map / count
        error_map_ = error_map_ / count
        np.savetxt(osp.join(proj_dir, 'error_map.txt'), error_map, fmt='%.4f')
        print('save_iter:', save_iter,
              'loss:', loss / count, 
              'loss_w_conf', loss_w_conf / count, 
              'count:', count)
        print(error_map.mean(), error_map_.mean())


        count = 0.
        loss = 0.
        loss_w_conf = 0.
        error_map = 0.
        error_map_ = 0.
        errors = []
        confidences = []
        for idx, voices, faces, genders, _ in val_loader:
            count += 1

            voices, faces = voices.cuda(), faces.cuda()
            faces = torch.unsqueeze(faces, -1)
            preds, confs = head(bkb(voices))
            fuse_confs = torch.sum(confs, dim=3, keepdim=True)
            fuse_preds = torch.sum(preds * confs, dim=3, keepdim=True) / fuse_confs
            square_dist = torch.sum(torch.square(fuse_preds - faces), dim=1)
            loss += torch.mean(square_dist).item()
            index = torch.argsort(fuse_confs.flatten())
            loss_w_conf += torch.mean(square_dist.flatten()[index[-1000:]]).item()

            error = torch.sum(torch.square(preds - faces), dim=1)
            errors.append(torch.squeeze(error).t())
            confidences.append(torch.squeeze(confs).t())

            # save gt and pred
            face_path = val_loader.dataset.data_info[idx]['face_path']
            pd_dir = osp.join(result_dir, 'pd_' + str(save_iter))
            pd_path = osp.join(pd_dir, osp.basename(face_path))
            if not os.path.exists(pd_dir):
                os.makedirs(pd_dir)
            fuse_pred = torch.squeeze(fuse_preds).cpu().numpy()

            gt_dir = osp.join(result_dir, 'gt_' + str(save_iter))
            gt_path = osp.join(gt_dir, osp.basename(face_path))
            if not os.path.exists(gt_dir):
                os.makedirs(gt_dir)
            face = torch.squeeze(faces).cpu().numpy()

            conf_dir = osp.join(result_dir, 'conf_' + str(save_iter))
            conf_path = osp.join(conf_dir, osp.basename(face_path))
            if not os.path.exists(conf_dir):
                os.makedirs(conf_dir)
            fuse_confs = torch.squeeze(fuse_confs).cpu().numpy()


            if genders.item() == 1:
                error_map += np.sum(np.square(face - male_template), axis=0)
                error_map_ += np.sum(np.square(face - female_template), axis=0)
            else:
                error_map += np.sum(np.square(face - female_template), axis=0)
                error_map_ += np.sum(np.square(face - male_template), axis=0)

        errors = torch.cat(errors, dim=0).t()
        confidences = torch.cat(confidences, dim=0).t()
        print(errors.size(), confidences.size())
        errors = errors - torch.mean(errors, dim=1, keepdim=True)
        confidences = confidences - torch.mean(confidences, dim=1, keepdim=True)
        corr = torch.sum(errors * confidences, dim=1)
        corr = corr / torch.sqrt(torch.sum(errors * errors, dim=1) * torch.sum(confidences * confidences, dim=1))
        # sorted_corr, indices = torch.sort(corr)
        corr = corr.detach().cpu().numpy()
        np.savetxt(osp.join(proj_dir, 'val_corr_coeff.txt'), corr, fmt='%.4f')
        

        error_map = error_map / count
        error_map_ = error_map_ / count
        print('save_iter:', save_iter,
              'loss:', loss / count, 
              'loss_w_conf', loss_w_conf / count, 
              'count:', count)
        print(error_map.mean(), error_map_.mean())
