import os
import os.path as osp
import random
import yaml
import copy

from glob import glob



meas_ids = range(0, 96, 1)
path = 'project/anth/sgd_666_l2_conf_f/measid_{}/2022*'
g = 'F'

for n_id, meas_id in enumerate(meas_ids):
    print(meas_id)
    proj_dirs = glob(path.format(meas_id))
    proj_dirs.sort()

    for n_dir, proj_dir in enumerate(proj_dirs):
 
        config_path = osp.join(proj_dir, 'configs.yml')
        with open(config_path, 'r') as f:
            configs = yaml.load(f, yaml.SafeLoader)

        test_config = copy.deepcopy(configs['data']['eval'])
        seed = test_config['dataset']['seed']
        train_seed = test_config['dataset']['train_seed']
        split = [s / sum([7,1,1,1]) for s in [7,1,1,1]]

        with open('data/penstate/data_16k.lst', 'r') as f:
            lines = f.readlines()

        data_items = []
        for line in lines:
            pid, gender, voice_path, face_path = line.rstrip().split(' ')
            item = {
                'pid': pid,
                'gender': gender,
                'voice_path': voice_path,
                'face_path': face_path,
            }
            if g is None or g == gender:
                data_items.append(item)

        # split
        pt1 = int(len(data_items) * sum(split[0:1]))
        pt2 = int(len(data_items) * sum(split[0:2]))
        pt3 = int(len(data_items) * sum(split[0:3]))

        random.Random(seed).shuffle(data_items)
        train_items = data_items[:pt3]
        test_items = data_items[pt3:]
        random.Random(train_seed).shuffle(train_items)


        if not osp.exists(osp.join(proj_dir, 'train_list.txt')):
            with open(osp.join(proj_dir, 'train_list.txt'), 'w') as f:
                for item in train_items[:pt1]:
                    f.write(' '.join([item['pid'], item['gender'], item['voice_path'], item['face_path']]) + '\n')
        if not osp.exists(osp.join(proj_dir, 'val_list.txt')):
            with open(osp.join(proj_dir, 'val_list.txt'), 'w') as f:
                for item in train_items[pt1:pt2]:
                    f.write(' '.join([item['pid'], item['gender'], item['voice_path'], item['face_path']]) + '\n')
        if not osp.exists(osp.join(proj_dir, 'eval_list.txt')):
            with open(osp.join(proj_dir, 'eval_list.txt'), 'w') as f:
                for item in train_items[pt2:pt3]:
                    f.write(' '.join([item['pid'], item['gender'], item['voice_path'], item['face_path']]) + '\n')
        if not osp.exists(osp.join(proj_dir, 'test_list.txt')):
            with open(osp.join(proj_dir, 'test_list.txt'), 'w') as f:
                for item in test_items:
                    f.write(' '.join([item['pid'], item['gender'], item['voice_path'], item['face_path']]) + '\n')
