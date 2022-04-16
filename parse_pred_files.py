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
    t_val = stats.t.ppf(0.95, n-1)
    CI = [mu - t_val * std / np.sqrt(n), mu + t_val * std / np.sqrt(n)]

    return CI, mu, std


# hfn v0 v1 v2
paths = [
#    'project/raw_vertex/sgd_l2_avg/2022*',
#    'project/raw_vertex/sgd_l2_avg_m/2022*',
#    'project/raw_vertex/sgd_l2_avg_f/2022*',
    'project/anth/sgd_666_l2_conf_f/measid_12/2022*',
#    'project/anths/sgd_666_l2_avg_m/2022*',
#    'project/anths/sgd_666_l2_avg_f/2022*',
#    'project/anths/sgd_666_l2_conf_f/2022*',
#    'project/anths/sgd_10086_l2_conf_f/2022*',
#    'project/anths/sgd_10086_l2_conf_f_wd001/2022*',
]

for path in paths:
    proj_dirs = glob(path)
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


        #index = np.argmin(val_fuse_dists, axis=0)
        index = np.argmin(val_fuse_dists.mean(axis=1, keepdims=True) + val_fuse_dists*0., axis=0)

        #e_baseline_dist = np.array([val_baseline_dists[index[i], i] for i in range(len(index))])
        #eval_fuse_dist = np.array([val_fuse_dists[index[i], i] for i in range(len(index))])
        e_baseline_dist = np.array([eval_baseline_dists[index[i], i] for i in range(len(index))])
        eval_fuse_dist = np.array([eval_fuse_dists[index[i], i] for i in range(len(index))])

        x.append(eval_fuse_dist/e_baseline_dist)


    # critical value
    print(path, len(proj_dirs))
    x = np.array(x)
    for i in range(x.shape[1]):
        CI, mu, std = get_statistics(x[:, i])
        print('CI-0.95: [{:8.5f}, {:8.5f}], mean: {:8.5f}, std: {:8.5f}'.format(CI[0], CI[1], mu, std))
