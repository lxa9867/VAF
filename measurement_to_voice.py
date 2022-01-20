import os
import os.path as osp
import numpy as np
import yaml
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from glob import glob
from scipy import stats
from dataset import PenstateDataset
from builder import build_dataloader
from importlib import import_module
from data.penstate.measurement import facial_indices, measurement

np.set_printoptions(precision=3, suppress=True)

def load_net(net_config, module):
    obj_type = net_config.pop('type')
    obj_cls = getattr(import_module('model.{}'.format(module)), obj_type)
    net = obj_cls(**net_config)

    model_path = osp.join(proj_dir, 'models/{}.pth'.format(module))
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()

    return net


meas_dirs = glob("project/sgd_l2_vfn_*")
meas_dirs.sort()
meas_preds =[]
meas_targets = []
meas_indices = []
meas_mu = []
meas_std = []
with torch.no_grad():
    for meas_dir in meas_dirs:
        meas_index = int(meas_dir[-2:])
        meas_indices.append(meas_index)

        proj_dir = glob(os.path.join(meas_dir, "2022*"))[0]
        config_file = osp.join(proj_dir, 'configs.yml')
        with open(config_file, 'r') as f:
            configs = yaml.load(f, yaml.SafeLoader)

        # data 
        test_config = copy.deepcopy(configs['data']['eval'])
        test_config['dataset']['mode'] = 'test'
        test_loader = build_dataloader(test_config)

        meas_mu.append(np.loadtxt(test_config['dataset']['norm_mu_path']))
        meas_std.append(np.loadtxt(test_config['dataset']['norm_std_path']))

        # network
        bkb_config = copy.deepcopy(configs['model']['backbone']['net'])
        bkb = load_net(bkb_config, 'backbone')
        head_config = copy.deepcopy(configs['model']['head']['net'])
        head = load_net(head_config, 'head')

        # eval
        sample_fuse_preds = []
        sample_targets = []
        for idx, voices, targets, _, _ in test_loader:
            voices, targets = voices.cuda(), targets.cuda()
            targets = torch.unsqueeze(targets, -1)
    
            preds, confs = head(bkb(voices))
    
            fuse_preds = torch.sum(preds * confs, dim=2, keepdim=True)
            fuse_confs = torch.sum(confs, dim=2, keepdim=True)
            fuse_preds = fuse_preds / fuse_confs
    
            sample_fuse_preds.append(fuse_preds)
            sample_targets.append(targets)
            
        sample_fuse_preds = torch.cat(sample_fuse_preds, dim=0)
        sample_targets = torch.cat(sample_targets, dim=0)
    
        meas_preds.append(sample_fuse_preds.flatten().detach().cpu().numpy())
        meas_targets.append(sample_targets.flatten().detach().cpu().numpy())

meas_preds = np.array(meas_preds, dtype=np.float32)
meas_targets = np.array(meas_targets, dtype=np.float32)


# get faces and pca basis
meas_dir = meas_dirs[0]
proj_dir = glob(os.path.join(meas_dir, "2022*"))[0]
config_file = osp.join(proj_dir, 'configs.yml')
with open(config_file, 'r') as f:
    configs = yaml.load(f, yaml.SafeLoader)

train_config = copy.deepcopy(configs['data']['train'])
train_config['dataloader']['batch_size'] = 1
train_config['dataloader']['shuffle'] = False
train_config['dataloader']['drop_last'] = False
train_loader = build_dataloader(train_config)
training_faces = []
for _, _, _, _, faces in train_loader:
    training_faces.append(faces.flatten())

training_faces = torch.stack(training_faces, dim=0)
training_faces = training_faces.numpy().astype(np.float32)
print(training_faces.shape)


test_config = copy.deepcopy(configs['data']['eval'])
test_config['dataset']['mode'] = 'test'
test_loader = build_dataloader(test_config)
testing_faces = []
for _, _, _, _, faces in test_loader:
    testing_faces.append(faces.flatten())

testing_faces = torch.stack(testing_faces, dim=0)
testing_faces = testing_faces.numpy().astype(np.float32)
print(testing_faces.shape)


pca_dim = 100
mu = np.mean(training_faces, axis=0, keepdims=True)
training_faces = training_faces - mu
cov_mtx = np.matmul(training_faces, training_faces.T)
w, proj_mtx = np.linalg.eig(cov_mtx)
proj_mtx = np.matmul(training_faces.T, proj_mtx[:, :pca_dim])
col_norm = np.sqrt(np.sum(proj_mtx * proj_mtx, axis=0, keepdims=True))
proj_mtx = proj_mtx / col_norm
mu = mu.astype(np.float32).T
proj_mtx = proj_mtx.astype(np.float32)

c = np.matmul(training_faces, proj_mtx)
s = np.sqrt(np.mean(c * c, axis=0))
s = torch.from_numpy(s.astype(np.float32))
s = s.view(-1, 1)
# s = s * 0. + 1.


# auto_grad
def get_distance(face, index1, index2):
    distance = face[index1, :] - face[index2, :]
    distance = torch.sum(distance * distance)

    return distance

meas_preds = torch.from_numpy(meas_preds)
meas_targets = torch.from_numpy(meas_targets)

mu = torch.from_numpy(mu)
proj_mtx = torch.from_numpy(proj_mtx)



lr = 0.1
wd = 0.000
for n in range(meas_preds.size(1)):
    coeffs = torch.zeros(pca_dim, 1, requires_grad=True)
    for i in range(20000):
        face = mu + torch.matmul(proj_mtx, s * coeffs)
        face = face.view(-1, 3)
        loss = wd * torch.mean(coeffs * coeffs)
        pred = []
        tgz = []
        for idx, meas_index in enumerate(meas_indices):
            terms = measurement[meas_index]
            if terms[0] == 'dist':
                index1 = facial_indices[terms[1]]
                index2 = facial_indices[terms[2]]
                # m = meas_preds[idx, 22] * meas_std[idx] + meas_mu[idx]
                m = meas_targets[idx, n] * meas_std[idx] + meas_mu[idx]
                loss += (500. * torch.square(get_distance(face, index1, index2) - m**2))
                pred.append(get_distance(face, index1, index2).item()**0.5 * 100)
                tgz.append(m.item() * 100)
        
        loss.backward()
        with torch.no_grad():
            if i % 100 == 0:
                print('iter: {:5d}'.format(i),
                      'loss: {:10.6f}'.format(loss.item()),
                      #'dist: [{:6.3f}, {:6.3f}, {:6.3f}, {:6.3f}]'.format(pred[0], pred[1], pred[2], pred[3]),
                      #'tgz: [{:6.3f}, {:6.3f}, {:6.3f}, {:6.3f}]'.format(tgz[0], tgz[1], tgz[2], tgz[3]),
                      'diff: [{:6.3f}, {:6.3f}, {:6.3f}, {:6.3f}, {:6.3f}, {:6.3f}, {:6.3f}, {:6.3f}]'.format(
                          np.abs(pred[0] - tgz[0]), np.abs(pred[1] - tgz[1]),
                          np.abs(pred[2] - tgz[2]), np.abs(pred[3] - tgz[3]),
                          np.abs(pred[4] - tgz[4]), np.abs(pred[5] - tgz[5]),
                          np.abs(pred[6] - tgz[6]), np.abs(pred[7] - tgz[7])),
                      'coeffs: {:6.3f}'.format(torch.mean(s * s * coeffs * coeffs).item()))
                      
                # print(coeffs.data.T.detach().numpy(), coeffs.grad.T.detach().numpy())
    
            coeffs -= lr * coeffs.grad
            coeffs.grad.zero_()

    np.savetxt('gt/' + str(n) + '.txt', testing_faces[n].reshape((-1, 3)), fmt='%6.3f')
    np.savetxt('pd/' + str(n) + '.txt', face.detach().numpy(), fmt='%6.3f')


