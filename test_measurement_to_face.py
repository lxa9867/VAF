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
from scipy.io import savemat

# direction
#path = 'project/anths/sgd_10086_l2_conf_m/2022*'
path = 'project/anths/sgd_10086_l2_conf_f/2022*'


''' load measurements and faces from training data
'''
proj_dir = glob(path)[0]
print(proj_dir)
config_path = osp.join(proj_dir, 'configs.yml')
with open(config_path, 'r') as f:
    configs = yaml.load(f, yaml.SafeLoader)
data_config = copy.deepcopy(configs['data']['eval'])
data_config['dataset']['mode'] = 'test'
data_config['dataset']['split'] = [0,0,0,1]
data_config['dataset']['norm_type'] = None

data_config['dataset']['ann_path'] = osp.join(proj_dir, 'train_list.txt')
train_loader = build_dataloader(data_config)
data_config['dataset']['ann_path'] = osp.join(proj_dir, 'val_list.txt')
val_loader = build_dataloader(data_config)
data_config['dataset']['ann_path'] = osp.join(proj_dir, 'eval_list.txt')
eval_loader = build_dataloader(data_config)

data_items = (train_loader.dataset.data_items
        + val_loader.dataset.data_items
        + eval_loader.dataset.data_items)

train_faces = [item['face'].flatten() for item in data_items]
train_AMs = [item['target'] for item in data_items]
train_faces = np.array(train_faces)
train_AMs = np.array(train_AMs)
print(train_faces.shape)
print(train_AMs.shape)

save_dir = osp.dirname(path)
statistics = {
    'mu': np.mean(train_faces, axis=0),
    'std': np.std(train_faces, axis=0),
    'AM_mu': np.mean(train_AMs, axis=0),
    'AM_std': np.std(train_AMs, axis=0),
}
savemat(osp.join(save_dir, 'statistics.mat'), statistics)

''' load faces from testing data, only for comparison
'''
data_config['dataset']['ann_path'] = osp.join(proj_dir, 'test_list.txt')
test_loader = build_dataloader(data_config)
test_faces = [item['face'].flatten()
        for item in test_loader.dataset.data_items]
test_faces = np.array(test_faces)
print(test_faces.shape)

''' compute PCA parameters
'''
pca_dim = 100
mu = np.mean(train_faces, axis=0, keepdims=True)
train_faces = train_faces - mu
cov_mtx = np.matmul(train_faces, train_faces.T)
[proj_mtx, w, _] = np.linalg.svd(cov_mtx)
#w, proj_mtx = np.linalg.eig(cov_mtx)
proj_mtx = proj_mtx.real

proj_mtx = np.matmul(train_faces.T, proj_mtx[:, :pca_dim])
col_norm = np.sqrt(np.sum(proj_mtx * proj_mtx, axis=0, keepdims=True))
proj_mtx = proj_mtx / col_norm
proj_mtx = proj_mtx.astype(np.float32)
mu = mu.astype(np.float32).T

c = np.matmul(train_faces, proj_mtx)
s = np.sqrt(np.mean(c * c, axis=0))
print('avg power of coeffs', (s * s).mean().item())


''' load predictions
'''
# 4 x #sample x #AM
# AM = min(AM, 180)
predictions = np.load(osp.join(osp.dirname(path), 'predictions.npy'))
savemat(osp.join(save_dir, 'predictions.mat'), {'predictions': predictions})
pred_AMs = predictions[0]
pred_confs = predictions[1]
gt_AMs = predictions[2]


''' compute normalization parameters
'''
#check if AM > 180
face_indices = test_loader.dataset.face_indices
measurement_info = test_loader.dataset.measurement_info
for idx, info in enumerate(measurement_info):
    mtype = info[0]
    if mtype == 'dist' or mtype == 'prop':
        train_AMs[:, idx] = np.square(train_AMs[:, idx])
        pred_AMs[:, idx] = np.square(pred_AMs[:, idx])
        gt_AMs[:, idx] = np.square(gt_AMs[:, idx])
    elif mtype == 'angle':
        train_AMs[:, idx] = np.minimum(train_AMs[:, idx], 180.-1e-5)
        train_AMs[:, idx] = np.cos(train_AMs[:, idx] / 180. * np.pi)
        pred_AMs[:, idx] = np.minimum(pred_AMs[:, idx], 180.-1e-5)
        pred_AMs[:, idx] = np.cos(pred_AMs[:, idx] / 180. * np.pi)
        gt_AMs[:, idx] = np.minimum(gt_AMs[:, idx], 180.-1e-5)
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


#AM_IDs = list(range(0, 96, 1))
AM_IDs = [87, 88, 50, 12, 13, 16, 51, 57, 15, 20, 21, 58, 72, 70, 69, 68, 22, 38, 71, 14] # for 10086 female
# AM_IDs = [88, 12, 87, 50, 13, 51, 57, 16, 58, 72, 70, 21, 14, 68, 69, 95, 71, 15, 20, 73] # for 666 female
#AM_IDs = [90, 25, 48, 88, 16, 15, 39, 20, 76, 21, 80, 50, 40, 84, 87, 12, 75, 91, 66, 92] # for 10086 male
# AM_IDs = [88, 50, 95, 87, 51, 33, 12, 89, 38, 22, 13, 16, 76, 90, 84, 57, 48, 75, 21, 2] # for 666 male
AM_IDs = AM_IDs[0:10]

# last
#AM_IDs = [83,44,78,34,86,59,41,36,35,3] # 10086 male
#AM_IDs = [35,60,40,59,46,30,6,41,36,5] # 10086 female
#AM_IDs = AM_IDs[0:5]

lr = 0.01
wd = 0.0000
for n_sample in range(pred_AMs.size(0)):
    coeffs = torch.zeros(pca_dim, 1, requires_grad=True)
    print(n_sample)
    for i in range(2000):
        face = mu + torch.matmul(proj_mtx, s * coeffs)
        face = face.view(-1, 3)

        loss = wd * torch.mean(coeffs * coeffs)
        diff = []
        ams = []
        gts = []
        for AM_ID in AM_IDs:
            info = measurement_info[AM_ID]
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
            cur_AM = (cur_AM - AM_mu[AM_ID].item()) / AM_std[AM_ID].item()
            #err = torch.square(cur_AM - pred_AMs[n_sample, AM_ID].item())
            err = torch.square(cur_AM - pred_AMs[n_sample, AM_ID].item())
            loss += (1. * err)
            diff.append(err.item())
            ams.append(cur_AM)
            gts.append(pred_AMs[n_sample, AM_ID].item())
        loss.backward()

        with torch.no_grad():
            if i % 100 == 0:
                print(
                    'iter: {:5d}'.format(i),
                    'loss: {:10.6f}'.format(loss.item()),
                    'diff: {:10.6f}'.format(sum(diff)),
                    'coeffs: {:6.3f}'.format(torch.mean(s * s * coeffs * coeffs).item()),
                    '[{:6.3f}, {:6.3f}, {:6.3f}, {:6.3f}]'.format(diff[0], diff[1], diff[2], diff[3]),
                    '[{:6.3f}, {:6.3f}, {:6.3f}, {:6.3f}]'.format(ams[0], ams[1], ams[2], ams[3]),
                    '[{:6.3f}, {:6.3f}, {:6.3f}, {:6.3f}]'.format(gts[0], gts[1], gts[2], gts[3]),
                )
            coeffs -= lr * coeffs.grad
            coeffs.grad.zero_()

    save_dir = osp.join(osp.dirname(path), 'face_{}_{}'.format(pca_dim, len(AM_IDs)))
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


