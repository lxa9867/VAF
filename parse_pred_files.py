import os
import os.path as osp
import numpy as np
import yaml
import copy
import math

from glob import glob
from scipy import stats

def parse_pred(pred_path):
    preds = np.loadtxt(pred_path)
    n_pred = preds.shape[1] - 1
    stride = n_pred // 3

    iters = preds[:, 0].astype(np.int64)
    baseline_dists = preds[:, 0*stride+1:1*stride+1]
    fuse_dists = preds[:, 1*stride+1:2*stride+1]
    confs = preds[:, 2*stride+1:3*stride+1]

    return iters, baseline_dists, fuse_dists, confs

def get_statistics(data):
    n = len(data)
    mu = data.mean()
    std = data.std()
    t_val = mu / std * np.sqrt(n)
    p_val = stats.t.sf(t_val, n-1)
    #CI = [mu - 1.697 * std / np.sqrt(n), mu + 1.697 * std / np.sqrt(n)]
    CI = [mu - 2.042 * std / np.sqrt(n), mu + 2.042 * std / np.sqrt(n)]

    return t_val, p_val, CI, mu, std


# hfn v0 v1 v2
paths = [
#    'project/raw_vertex/sgd_l2_avg/2022*',
#    'project/raw_vertex/sgd_l2_avg_m/2022*',
#    'project/raw_vertex/sgd_l2_avg_f/2022*',
    'project/pca/sgd_l2_avg_2_m/2022*',
    'project/pca/sgd_l2_avg_2_f/2022*',
    'project/pca/sgd_l2_avg_5_m/2022*',
    'project/pca/sgd_l2_avg_5_f/2022*',
    'project/pca/sgd_l2_avg_10_m/2022*',
    'project/pca/sgd_l2_avg_10_f/2022*',
    'project/pca/sgd_l2_avg_20_m/2022*',
    'project/pca/sgd_l2_avg_20_f/2022*',
]
for path in paths:
    proj_dirs = glob(path)
    #    ddd
    proj_dirs.sort()
    x = []
    norm_baseline = []
    for idx, proj_dir in enumerate(proj_dirs):
        config_file = osp.join(proj_dir, 'configs.yml')
        with open(config_file, 'r') as f:
            configs = yaml.load(f, yaml.SafeLoader)

        val_path = proj_dir + '/val_pred.txt'
        val_iters, val_baseline_dists, val_fuse_dists, val_confs = parse_pred(val_path)

        eval_path = proj_dir + '/eval_pred.txt'
        eval_iters, eval_baseline_dists, eval_fuse_dists, eval_confs = parse_pred(eval_path)

        
        index = np.argmin(val_fuse_dists, axis=0)
        #e_baseline_dist = np.array([val_baseline_dists[index[i], i] for i in range(len(index))])
        #eval_fuse_dist = np.array([val_fuse_dists[index[i], i] for i in range(len(index))])
        e_baseline_dist = np.array([eval_baseline_dists[index[i], i] for i in range(len(index))])
        eval_fuse_dist = np.array([eval_fuse_dists[index[i], i] for i in range(len(index))])

        x.append(1. - eval_fuse_dist/e_baseline_dist)


    # critical value
    print(path, len(proj_dirs))
    x = np.array(x)
    for i in range(x.shape[1]):
        t, p, CI, mu, std = get_statistics(x[:, i])
        print('t-score: {:8.5f}, p-value: {:6.3e}, CI-0.95: [{:8.5f}, {:8.5f}], mean: {:8.5f}, std: {:8.5f}'.format(t, p, CI[0], CI[1], mu, std))
