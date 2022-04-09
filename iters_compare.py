import os
import os.path as osp
import numpy as np
import yaml
import copy
import math

from glob import glob
from scipy import stats

def parse_log(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()
    iters = []
    baseline_dists = []
    fuse_dists = []
    selected_dists = []
    for line in lines:
        terms = line.rstrip().split(',')
        iters.append(float(terms[1].split(': ')[-1]))
        baseline_dists.append(float(terms[3].split(': ')[-1]))
        fuse_dists.append(float(terms[4].split(': ')[-1]))
        selected_dists.append(float(terms[5].split(': ')[-1]))

    iters = np.array(iters)
    baseline_dists = np.array(baseline_dists)
    fuse_dists = np.array(fuse_dists)
    selected_dists = np.array(selected_dists)

    return iters, baseline_dists, fuse_dists, selected_dists

def get_mvavg(data, wsize):
    avg_data = []
    for i in range(len(data)):
        st = max(0, i-wsize//2)
        ed = min(len(data), i+wsize//2+1)
        avg_data.append(sum(data[st:ed]) / len(data[st:ed]))

    return avg_data

def get_statistics(data):
    n = len(data)
    mu = data.mean()
    std = data.std()
    t_val = mu / std * np.sqrt(n)
    p_val = stats.t.sf(t_val, n-1)
    CI = [mu - 1.697 * std / np.sqrt(n), mu + 1.697 * std / np.sqrt(n)]

    return t_val, p_val, CI, mu, std


# hfn v0 v1 v2
paths = [
    #'project/anth/sgd_666_l2_conf_f_6_8/measid_12/2022*',
    #'project/anth/sgd_666_l2_conf_f_6_8/measid_13/2022*',
    #'project/anth/sgd_666_l2_conf_f_6_8/measid_16/2022*',
    #'project/anth/sgd_666_l2_conf_f_6_8/measid_50/2022*',
    #'project/anth/sgd_666_l2_conf_f_6_8/measid_51/2022*',
    #'project/anth/sgd_666_l2_conf_f_6_8/measid_57/2022*',

    'project/anth/sgd_666_l2_conf_f/measid_12/2022*',
]
for path in paths:
    proj_dirs = glob(path)
    proj_dirs.sort()
    wsize = 1
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    x5 = []
    x6 = []
    x7 = []
    x8 = []
    for idx, proj_dir in enumerate(proj_dirs):
        config_file = osp.join(proj_dir, 'configs.yml')
        with open(config_file, 'r') as f:
            configs = yaml.load(f, yaml.SafeLoader)

        val_path = proj_dir + '/val.log'
        val_iters, val_baseline_dists, val_fuse_dists, val_selected_dists = parse_log(val_path)
        val_fuse_dists_ma = get_mvavg(val_fuse_dists, wsize)
        val_selected_dists_ma = get_mvavg(val_selected_dists, wsize)

        eval_path = proj_dir + '/eval.log'
        eval_iters, eval_baseline_dists, eval_fuse_dists, eval_selected_dists = parse_log(eval_path)
        e_baseline_dist = eval_baseline_dists[0].item()

        index = np.argmin(val_fuse_dists_ma[0:250])
        e_fuse_dist_0 = eval_fuse_dists[index].item()
        e_selected_dist_0 = eval_selected_dists[index].item()

        index = np.argmin(val_fuse_dists_ma[0:200])
        e_fuse_dist_1 = eval_fuse_dists[index].item()
        e_selected_dist_1 = eval_selected_dists[index].item()

        index = np.argmin(val_fuse_dists_ma[0:150])
        e_fuse_dist_2 = eval_fuse_dists[index].item()
        e_selected_dist_2 = eval_selected_dists[index].item()
        
        index = np.argmin(val_fuse_dists_ma[0:100])
        e_fuse_dist_3 = eval_fuse_dists[index].item()
        e_selected_dist_3 = eval_selected_dists[index].item()

        index = np.argmin(val_fuse_dists_ma[0:50])
        e_fuse_dist_4 = eval_fuse_dists[index].item()
        e_selected_dist_4 = eval_selected_dists[index].item()


        x1.append(e_fuse_dist_0 - e_fuse_dist_1)
        x2.append(e_selected_dist_0 - e_selected_dist_1)
        x3.append(e_fuse_dist_0 - e_fuse_dist_2)
        x4.append(e_selected_dist_0 - e_selected_dist_2)
        x5.append(e_fuse_dist_0 - e_fuse_dist_3)
        x6.append(e_selected_dist_0 - e_selected_dist_3)
        x7.append(e_fuse_dist_0 - e_fuse_dist_4)
        x8.append(e_selected_dist_0 - e_selected_dist_4)

        
        x1[-1] = x1[-1] / e_baseline_dist
        x2[-1] = x2[-1] / e_baseline_dist
        x3[-1] = x3[-1] / e_baseline_dist
        x4[-1] = x4[-1] / e_baseline_dist
        x5[-1] = x5[-1] / e_baseline_dist
        x6[-1] = x6[-1] / e_baseline_dist
        x7[-1] = x7[-1] / e_baseline_dist
        x8[-1] = x8[-1] / e_baseline_dist
        

    print()
    print(path)
    for i, (f, s) in enumerate([(x1, x2), (x3, x4), (x5, x6), (x7, x8)]):
        f = np.array(f)
        t1, p1, CI1, mu1, std1 = get_statistics(f)
        s = np.array(s)
        t2, p2, CI2, mu2, std2 = get_statistics(s)
        print(200 - 50*i)
        print('t-score: {:8.5f}, p-value: {:6.3e}, CI-0.90: [{:8.5f}, {:8.5f}], mean: {:8.5f}, std: {:8.5f}'.format(t1, p1, CI1[0], CI1[1], mu1, std1))
        print('t-score: {:8.5f}, p-value: {:6.3e}, CI-0.90: [{:8.5f}, {:8.5f}], mean: {:8.5f}, std: {:8.5f}'.format(t2, p2, CI2[0], CI2[1], mu2, std2))
