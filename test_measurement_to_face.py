import os
import os.path as osp
import numpy as np
import yaml
import copy
import random

import torch
import torch.nn.functional as F

from glob import glob
from dataset import PenstateDataset
from builder import build_dataloader

# np.set_printoptions(precision=3, suppress=True)


''' load measurements and faces from training data
'''
# meas_ids = list(range(0, 96, 1))
# meas_ids = [88, 51, 50, 16, 87, 21, 57, 72, 12, 13, 70, 73, 71, 68, 58, 33, 20, 92, 15, 56]
meas_ids = [88, 51, 50, 87, 12, 13, 57, 58, 16, 70, 72, 95,  2, 14, 93, 21, 71, 20, 68, 33]
#meas_ids = meas_ids[0:10]
path = 'project/anth/sgd_666_l2_conf_f/measid_{}/2022*'
proj_dir = glob(path.format(meas_ids[0]))[0]
print(proj_dir)
config_path = osp.join(proj_dir, 'configs.yml')
with open(config_path, 'r') as f:
    configs = yaml.load(f, yaml.SafeLoader)
data_config = copy.deepcopy(configs['data']['val'])
data_config['dataset']['split'] = [0, 9, 0, 1]
data_config['dataset']['norm_type'] = None
data_config['dataset']['measure_indices'] = meas_ids
data_config['dataloader']['batch_size'] = 1
data_config['dataloader']['shuffle'] = False
train_loader = build_dataloader(data_config)
train_faces = [item['face'].flatten()
        for item in train_loader.dataset.data_items]
train_AMs = [item['target']
        for item in train_loader.dataset.data_items]
train_faces = np.array(train_faces)
train_AMs = np.array(train_AMs)
print(train_faces.shape)
print(train_AMs.shape)

''' load faces from testing data, only for comparison
'''
data_config['dataset']['mode'] = 'test'
test_loader = build_dataloader(data_config)
test_faces = [item['face'].flatten()
        for item in test_loader.dataset.data_items]
test_faces = np.array(test_faces)
print(test_faces.shape)

''' compute PCA parameters
'''
pca_dim = 20
mu = np.mean(train_faces, axis=0, keepdims=True)
train_faces = train_faces - mu
cov_mtx = np.matmul(train_faces, train_faces.T)
np.savetxt('temp.txt', cov_mtx, fmt='%6.3f')
print(cov_mtx.shape)
w, proj_mtx = np.linalg.eig(cov_mtx)
print('111111')
proj_mtx = proj_mtx.real

proj_mtx = np.matmul(train_faces.T, proj_mtx[:, :pca_dim])
print('111111')
col_norm = np.sqrt(np.sum(proj_mtx * proj_mtx, axis=0, keepdims=True))
print('111111')
proj_mtx = proj_mtx / col_norm
print('111111')
proj_mtx = proj_mtx.astype(np.float32)
mu = mu.astype(np.float32).T
print('111111')

c = np.matmul(train_faces, proj_mtx)
s = np.sqrt(np.mean(c * c, axis=0))
print('avg power of coeffs', (s * s).mean().item())


pred_AMs = []
pred_confs = []
gt_AMs = []
for meas_id in meas_ids:
    predictions = np.loadtxt(
        osp.join(osp.dirname(path).format(meas_id), 'predictions.txt')
    )
    pred_AMs.append(predictions[:, 0].flatten())
    pred_confs.append(predictions[:, 1].flatten())
    gt_AMs.append(predictions[:, 2].flatten())
pred_AMs = np.array(pred_AMs).astype(np.float32).T
pred_confs = np.array(pred_confs).astype(np.float32).T
gt_AMs = np.array(gt_AMs).astype(np.float32).T

''' compute normalization parameters
'''
face_indices = train_loader.dataset.face_indices
measurement_info = train_loader.dataset.measurement_info
for idx, meas_id in enumerate(meas_ids):
    info = measurement_info[meas_id]
    mtype = info[0]
    if mtype == 'dist' or mtype == 'prop':
        train_AMs[:, idx] = np.square(train_AMs[:, idx])
        pred_AMs[:, idx] = np.square(pred_AMs[:, idx])
        gt_AMs[:, idx] = np.square(gt_AMs[:, idx])
    elif mtype == 'angle':
        train_AMs[:, idx] = np.cos(train_AMs[:, idx] / 180. * np.pi)
        pred_AMs[:, idx] = np.cos(pred_AMs[:, idx] / 180. * np.pi)
        gt_AMs[:, idx] = np.cos(gt_AMs[:, idx] / 180. * np.pi)
AM_mu = np.mean(train_AMs, axis=0, keepdims=True)
AM_std = np.std(train_AMs, axis=0, keepdims=True)
pred_AMs = (pred_AMs - AM_mu) / AM_std
gt_AMs = (gt_AMs - AM_mu) / AM_std


''' measurement computating operations in PyTorch
'''
# pytorch - auto_grad
def get_distance(face, index1, index2):
    distance = face[index1, :] - face[index2, :]
    distance = torch.sum(distance * distance)

    return distance

def get_proportion(face, index1, index2, index3, index4):
    distance1 = get_distance(face, index1, index2)
    distance2 = get_distance(face, index3, index4)

    return distance1 / distance2

def get_cos_angle(face, index1, index2, index3):
    v1 = face[index1, :] - face[index2, :]
    v2 = face[index3, :] - face[index2, :]
    cos_angle = F.cosine_similarity(v1, v2, dim=0)

    return cos_angle


''' optimization-based reconstruction
'''
mu = torch.from_numpy(mu)
proj_mtx = torch.from_numpy(proj_mtx)
s = torch.from_numpy(s).view(-1, 1)

pred_AMs = torch.from_numpy(pred_AMs)
gt_AMs = torch.from_numpy(gt_AMs)
AM_mu = torch.from_numpy(AM_mu.flatten())
AM_std = torch.from_numpy(AM_std.flatten())


lr = 0.01
wd = 0.0000
for n_sample in range(pred_AMs.size(0)):
    coeffs = torch.zeros(pca_dim, 1, requires_grad=True)
    for i in range(2000):
        face = mu + torch.matmul(proj_mtx, s * coeffs)
        face = face.view(-1, 3)

        loss = wd * torch.mean(coeffs * coeffs)
        diff = []
        ams = []
        gts = []
        for idx, meas_id in enumerate(meas_ids):
            info = measurement_info[meas_id]
            index1 = face_indices[info[1]]
            index2 = face_indices[info[2]]
            if info[0] == 'dist':
                cur_AM = get_distance(face, index1, index2)
            elif info[0] == 'prop':
                index3 = face_indices[info[3]]
                index4 = face_indices[info[4]]
                cur_AM = get_proportion(face, index1, index2, index3, index4)
            elif info[0] == 'angle':
                index3 = face_indices[info[3]]
                cur_AM = get_cos_angle(face, index1, index2, index3)
            else:
                raise ValueError('undefined measurement type.')
            cur_AM = (cur_AM - AM_mu[idx].item()) / AM_std[idx].item()
            err = cur_AM - gt_AMs[n_sample, idx].item()
            loss += (1. * torch.square(err))
            diff.append(torch.square(err).item())
            ams.append(cur_AM)
            gts.append(gt_AMs[n_sample, idx].item())
        loss.backward()

        with torch.no_grad():
            if i % 100 == 0:
                print(
                    'iter: {:5d}'.format(i),
                    'loss: {:10.6f}'.format(loss.item()),
                    'diff: {:10.6f}'.format(sum(diff)),
                    'coeffs: {:6.3f}'.format(torch.mean(s * s * coeffs * coeffs).item()),
                    '[{:6.3f}, {:6.3f}, {:6.3f}, {:6.3f}]'.format(ams[0], ams[1], ams[2], ams[3]),
                    '[{:6.3f}, {:6.3f}, {:6.3f}, {:6.3f}]'.format(gts[0], gts[1], gts[2], gts[3]),
                )
                       
            coeffs -= lr * coeffs.grad
            coeffs.grad.zero_()

    save_dir = osp.join(osp.dirname(osp.dirname(path)), 'face_100_{}'.format(len(meas_ids)))
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    np.savetxt(
        osp.join(save_dir, 'gt_{}.txt'.format(n_sample)),
        test_faces[n_sample].reshape((-1, 3)),
    )
    np.savetxt(
        osp.join(save_dir, 'pd_{}.txt'.format(n_sample)),
        face.detach().numpy(),
    )


