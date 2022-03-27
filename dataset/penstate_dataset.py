import numpy as np
import random
import torchaudio

from torch.utils.data import Dataset
from datetime import datetime


class PenstateDataset(Dataset):
    """ An example of creating a dataset from a given data_items.
    """
    def __init__(self, ann_path, gender,
                 seed, split, mode,
                 sample_rate, duration,
                 pca_dims=[], pca_mu_path=None, pca_proj_mtx_path=None,
                 norm_type=None, norm_mu_path=None, norm_std_path=None):
        super(PenstateDataset, self).__init__()
        self.ann_path = ann_path
        self.gender = gender
        self.seed = seed
        self.split = split
        self.mode = mode
        self.sample_rate = sample_rate
        self.duration = duration
        self.pca_dims = pca_dims
        self.norm_type = norm_type

        # get data
        self.data_items = self.get_data()

        # get target
        if len(pca_dims) > 0:
            if mode=='train':
                self.pca_mu, self.pca_proj_mtx = self.get_pca_param()
            else:
                self.pca_mu = np.loadtxt(pca_mu_path)
                self.pca_proj_mtx = np.loadtxt(pca_proj_mtx_path)
            self.pca_projection()

        # normalize target
        if norm_type is not None:
            if mode=='train':
                self.norm_mu, self.norm_std = self.get_normalizer()
            else:
                self.norm_mu = np.loadtxt(norm_mu_path)
                self.norm_std = np.loadtxt(norm_std_path)
            self.normalize()

        self.data_indices = np.random.permutation(len(self.data_items)).tolist()
        if self.mode == 'train':
            for _ in range(1000):
                self.data_indices.extend(
                    np.random.permutation(len(self.data_items)).tolist()
                )

    def get_data(self,):
        split = [s / sum(self.split) for s in self.split]
        with open(self.ann_path, 'r') as f:
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
            if self.gender is None or self.gender == gender:
                data_items.append(item)

        # split
        random.Random(self.seed).shuffle(data_items)
        pt1 = int(len(data_items) * sum(split[0:1]))
        pt2 = int(len(data_items) * sum(split[0:2]))
        if self.mode == 'train':
            selected_items = data_items[:pt1]
        elif self.mode == 'val':
            selected_items = data_items[pt1:pt2]
        elif self.mode == 'eval':
            selected_items = data_items[pt2:]

        # load face data
        for item in selected_items:
            voice_path = item['voice_path']
            voice, _ = torchaudio.load(voice_path)
            item['voice'] = voice[0]
            face_path = item['face_path']
            item['face'] = np.loadtxt(face_path, dtype=np.float32)
            item['target'] = item['face'].flatten()

        return selected_items


    def get_pca_param(self):
        # check
        assert self.mode == 'train' and len(self.pca_dims) > 0

        targets = [item['target'] for item in self.data_items]
        targets = np.array(targets, dtype=np.float32)
        mu = np.mean(targets, axis=0, keepdims=True)
        targets = targets - mu
        cov_mtx = np.matmul(targets, targets.T)
        w, proj_mtx = np.linalg.eig(cov_mtx)
        proj_mtx = np.matmul(targets.T, proj_mtx[:, self.pca_dims])
        col_norm = np.sqrt(np.sum(proj_mtx * proj_mtx, axis=0, keepdims=True))
        proj_mtx = proj_mtx / col_norm

        return mu, proj_mtx

    def pca_projection(self,):
        for item in self.data_items:
            target = np.expand_dims(item['target'], axis=0)
            item['target'] = np.matmul(target - self.pca_mu, self.pca_proj_mtx)

    def get_normalizer(self,):
        # compute mean and std
        assert self.mode == 'train' and self.norm_type is not None
        targets = [item['target'] for item in self.data_items]
        targets = np.array(targets, dtype=np.float32)
        mu = np.mean(targets, axis=0)
        delta = targets - np.expand_dims(mu, axis=0)
        if self.norm_type == 'l1':
            std = np.mean(np.abs(delta), axis=0)
        elif self.norm_type == 'l2':
            std = np.sqrt(np.mean(np.square(delta), axis=0))
        else:
            raise ValueError('unknown norm type {}'.format(self.norm_type))

        return mu, std

    def normalize(self):
        for item in self.data_items:
            item['target'] = (item['target'] - self.norm_mu) / self.norm_std
            item['target'] = item['target'].flatten()

    def prepare(self, idx):
        idx = self.data_indices[idx]
        item = self.data_items[idx]

        # voice
        voice_path = item['voice_path']
        voice_info = torchaudio.info(voice_path)
        assert self.sample_rate == voice_info.sample_rate
        num_frames = voice_info.num_frames

        if self.mode == 'train':
            max_num_frames = self.duration[1] * self.sample_rate
            assert num_frames >= max_num_frames
            frame_offset = np.random.randint(
                num_frames - max_num_frames, size=1)
            frame_offset = np.asscalar(frame_offset)
            voice = item['voice'][frame_offset:frame_offset+max_num_frames]
        else:
            voice = item['voice']

        target = item['target']
        gender = item['gender']
        face = item['face']

        return idx, voice, target, gender, face

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        return self.prepare(idx)
