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
from data.penstate.measurement import facial_indices, measurement

np.set_printoptions(precision=3, suppress=True)


''' load predicted measurements
'''
used_meas_ids = [12, 13, 16, 50, 51, 57, 58, 87, 88]
result_paths = glob("project/seed_666/*.txt")
result_paths.sort()
meas_ids = []
preds = []
gts = []
confs = []
for result_path in result_paths:
    result = np.loadtxt(result_path)
    meas_id = int(osp.basename(result_path)[:-4])
    if meas_id not in used_meas_ids:
        continue
    meas_ids.append(meas_id)
    preds.append(result[0])
    gts.append(result[1])
    confs.append(result[2])

tgz_measurements = np.array(preds).T
# tgz_measurements = np.array(gts).T
confs = np.array(confs).T

preds = np.array(preds).T
gts = np.array(gts).T
#mu = np.mean(gts, axis=0, keepdims=True)
#std = np.std(gts, axis=0, keepdims=True)
#preds = (preds - mu) / std
#gts = (gts - mu) / std
np.savetxt('project/seed_666/results/preds.txt', preds, fmt='%9.6f')
np.savetxt('project/seed_666/results/gts.txt', gts, fmt='%9.6f')
np.savetxt('project/seed_666/results/confs.txt', confs, fmt='%9.6f')
np.savetxt('project/seed_666/results/errs.txt', np.abs(preds-gts), fmt='%9.6f')
xxxxxxx


''' load measurements and faces from training data
'''
proj_dir = glob("project/seed_666/sgd_*/2022*")[0]
print(proj_dir)
config_file = osp.join(proj_dir, 'configs.yml')
with open(config_file, 'r') as f:
    configs = yaml.load(f, yaml.SafeLoader)

training_faces = []
training_measurements = []

train_config = copy.deepcopy(configs['data']['train'])
train_config['dataset']['measure_indices'] = meas_ids
train_config['dataloader']['batch_size'] = 1
train_config['dataloader']['shuffle'] = False
train_config['dataloader']['drop_last'] = False
train_loader = build_dataloader(train_config)
for _, _, targets, _, faces in train_loader:
    assert faces.size(0) == 1
    training_faces.append(faces.flatten())
    targets = targets.numpy().flatten()
    targets = targets * train_loader.dataset.norm_std + train_loader.dataset.norm_mu
    training_measurements.append(targets)

val_config = copy.deepcopy(configs['data']['val'])
val_config['dataset']['measure_indices'] = meas_ids
val_config['dataset']['norm_mu_path'] = osp.join(proj_dir, 'norm_mu.txt')
val_config['dataset']['norm_std_path'] = osp.join(proj_dir, 'norm_std.txt')
val_loader = build_dataloader(val_config)
for _, _, targets, _, faces in val_loader:
    assert faces.size(0) == 1
    training_faces.append(faces.flatten())
    targets = targets.numpy().flatten()
    targets = targets * val_loader.dataset.norm_std + val_loader.dataset.norm_mu
    training_measurements.append(targets)

eval_config = copy.deepcopy(configs['data']['eval'])
eval_config['dataset']['measure_indices'] = meas_ids
eval_config['dataset']['norm_mu_path'] = osp.join(proj_dir, 'norm_mu.txt')
eval_config['dataset']['norm_std_path'] = osp.join(proj_dir, 'norm_std.txt')
eval_loader = build_dataloader(eval_config)
for _, _, targets, _, faces in eval_loader:
    assert faces.size(0) == 1
    training_faces.append(faces.flatten())
    targets = targets.numpy().flatten()
    targets = targets * eval_loader.dataset.norm_std + eval_loader.dataset.norm_mu
    training_measurements.append(targets)

training_faces = torch.stack(training_faces, dim=0)
training_faces = training_faces.numpy().astype(np.float32)
training_measurements = np.array(training_measurements)
print(training_faces.shape)
print(training_measurements.shape)

''' load faces from testing data, only for comparing results
'''
test_config = copy.deepcopy(configs['data']['eval'])
test_config['dataset']['mode'] = 'test'
test_config['dataset']['norm_mu_path'] = osp.join(proj_dir, 'norm_mu.txt')
test_config['dataset']['norm_std_path'] = osp.join(proj_dir, 'norm_std.txt')
test_loader = build_dataloader(test_config)
testing_faces = []
for _, _, _, _, faces in test_loader:
    testing_faces.append(faces.flatten())

testing_faces = torch.stack(testing_faces, dim=0)
testing_faces = testing_faces.numpy().astype(np.float32)
print(testing_faces.shape)

''' compute normalization parameters
'''
for idx, meas_id in enumerate(meas_ids):
    terms = measurement[meas_id]
    if terms[0] == 'dist' or terms[0] == 'prop':
        training_measurements[:, idx] = np.square(training_measurements[:, idx])
        tgz_measurements[:, idx] = np.square(tgz_measurements[:, idx])
    elif terms[0] == 'angle':
        training_measurements[:, idx] = np.cos(training_measurements[:, idx] / 180. * np.pi)
        tgz_measurements[:, idx] = np.cos(tgz_measurements[:, idx] / 180. * np.pi)
meas_mu = np.mean(training_measurements, axis=0)
meas_std = np.std(training_measurements, axis=0)

''' compute PCA parameters
'''
pca_dim = 10
mu = np.mean(training_faces, axis=0, keepdims=True)
training_faces = training_faces - mu
cov_mtx = np.matmul(training_faces, training_faces.T)
w, proj_mtx = np.linalg.eig(cov_mtx)
proj_mtx = proj_mtx.real

proj_mtx = np.matmul(training_faces.T, proj_mtx[:, :pca_dim])
col_norm = np.sqrt(np.sum(proj_mtx * proj_mtx, axis=0, keepdims=True))
proj_mtx = proj_mtx / col_norm
mu = mu.astype(np.float32).T
proj_mtx = proj_mtx.astype(np.float32)

c = np.matmul(training_faces, proj_mtx)
s = np.sqrt(np.mean(c * c, axis=0))
s = torch.from_numpy(s.astype(np.float32))
s = s.view(-1, 1)
print('avg power of coeffs', (s * s).mean().item())

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
# preparing
tgz_measurements = torch.from_numpy(tgz_measurements)
meas_mu = torch.from_numpy(meas_mu)
meas_std = torch.from_numpy(meas_std)
mu = torch.from_numpy(mu)
proj_mtx = torch.from_numpy(proj_mtx)
np.savetxt('project/seed_666/results/avg_face.txt',
        mu.view(-1, 3).numpy(), fmt='%6.3f')
xxxx

lr = 0.01
wd = 0.0000
for n_sample in range(tgz_measurements.size(0)):
    coeffs = torch.zeros(pca_dim, 1, requires_grad=True)
    for i in range(2000):
        face = mu + torch.matmul(proj_mtx, s * coeffs)
        face = face.view(-1, 3)

        loss = wd * torch.mean(coeffs * coeffs)
        diff = []
        for n_meas, meas_id in enumerate(meas_ids):
            terms = measurement[meas_id]
            index1 = facial_indices[terms[1]]
            index2 = facial_indices[terms[2]]
            if terms[0] == 'dist':
                estimate = get_distance(face, index1, index2)
            elif terms[0] == 'prop':
                index3 = facial_indices[terms[3]]
                index4 = facial_indices[terms[4]]
                estimate = get_proportion(face, index1, index2, index3, index4)
            elif terms[0] == 'angle':
                index3 = facial_indices[terms[3]]
                estimate = get_cos_angle(face, index1, index2, index3)
            else:
                raise ValueError('undefined measurement type.')
            estimate = (estimate - meas_mu[n_meas]) / meas_std[n_meas]
            tgz_measurement = (tgz_measurements[n_sample, n_meas] - meas_mu[n_meas]) / meas_std[n_meas]
            err = estimate - tgz_measurement
            loss += (1. * torch.square(err))
            diff.append(torch.square(err).item())
        loss.backward()

        with torch.no_grad():
            if i % 100 == 0:
                print('iter: {:5d}'.format(i),
                      'loss: {:10.6f}'.format(loss.item()),
                      'diff: {:10.6f}'.format(sum(diff)),
                      'coeffs: {:6.3f}'.format(torch.mean(s * s * coeffs * coeffs).item()))
            coeffs -= lr * coeffs.grad
            coeffs.grad.zero_()

    np.savetxt('project/seed_666/results/gt_' + str(n_sample) + '.txt',
            testing_faces[n_sample].reshape((-1, 3)), fmt='%6.3f')
    np.savetxt('project/seed_666/results/pd_' + str(n_sample) + '.txt', 
            face.detach().numpy(), fmt='%6.3f')


