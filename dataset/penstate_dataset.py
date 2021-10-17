import numpy as np
import random
import torchaudio

from torch.utils.data import Dataset

class PenstateDataset(Dataset):
    """ An example of creating a dataset from a given data_info.
    """
    def __init__(self,
                 ann_file=None,
                 seed=None,
                 split=None,
                 mode=None,
                 sample_rate=None,
                 duration=None,
                 pca_dims=None,
                 pca_mu_path=None,
                 pca_proj_mtx_path=None,
                 norm_type=None,
                 norm_mu_path=None,
                 norm_std_path=None,
                 gender=None):
        super(PenstateDataset, self).__init__()
        self.ann_file = ann_file
        self.seed = seed
        self.split = split
        self.mode = mode
        self.sample_rate = sample_rate
        self.duration = duration
        self.pca_dims = pca_dims
        self.norm_type = norm_type

        # target: face
        self.data_info = self.get_data()

        if mode=='train':
            self.pca_mu, self.pca_proj_mtx = self.get_pca_param()
        else:
            self.pca_mu = np.loadtxt(pca_mu_path)
            self.pca_proj_mtx = np.loadtxt(pca_proj_mtx_path)
        
        # target principle component
        if pca_dims:
            self.pca_projection()

        if mode=='train':
            self.norm_mu, self.norm_std = self.get_normalizer()
        else:
            self.norm_mu = np.loadtxt(norm_mu_path)
            self.norm_std = np.loadtxt(norm_std_path)

        # target: normalized pc
        if norm_type:
            self.normalize()

    def get_data(self,):
        ann_file = self.ann_file
        seed = self.seed
        split = [s / sum(self.split) for s in self.split]
        mode = self.mode
          
        with open(ann_file, 'r') as f:
            lines = f.readlines()

        data_info = []
        for line in lines:
            pid, gender, ancestry, voice_path, face_path = line.rstrip().split(' ')
            info = {'pid': pid, 'gender': gender,
                    'ancestry': ancestry,
                    'voice_path': voice_path,
                    'face_path': face_path}
            if gender == 'F':
                data_info.append(info)

        # split
        random.Random(seed).shuffle(data_info)
        pt1 = int(len(data_info) * sum(split[0:1]))
        pt2 = int(len(data_info) * sum(split[0:2]))
        if mode == 'train':
            data_info = data_info[:pt1]
            #data_info = data_info[:64]
        elif mode == 'val':
            data_info = data_info[pt1:pt2]
        elif mode == 'eval':
            data_info = data_info[pt2:]

        # load face data
        for info in data_info:
            face_path = info['face_path']
            info['face'] = np.loadtxt(face_path, dtype=np.float32)
            info['target'] = info['face'].flatten()

        return data_info

    def get_pca_param(self,):
        assert self.mode == 'train'
        targets = [info['target'] for info in self.data_info]
        targets = np.array(targets, dtype=np.float32)
        mu = np.mean(targets, axis=0, keepdims=True)
        targets = targets - mu
        cov_mtx = np.matmul(targets, targets.T)
        w, proj_mtx = np.linalg.eig(cov_mtx)
        proj_mtx = np.matmul(targets.T, proj_mtx[:, 0:5])
        col_norm = np.sqrt(np.sum(proj_mtx * proj_mtx, axis=0, keepdims=True))
        proj_mtx = proj_mtx / col_norm

        '''
        sample = targets[0:1, :]
        reconst_face = np.matmul(np.matmul(sample, proj_mtx), proj_mtx.T)
        print(reconst_face.shape, sample.shape)
        print(np.sum(np.abs(reconst_face - sample)))
        print(np.sum(np.abs(sample)))
        xxxxxxx
        print(reconst_face[0, 0:10])
        print(sample[0, 0:10])
        xxxxxxx
        '''
        return mu, proj_mtx
    
    def pca_projection(self,):
        for info in self.data_info:
            target = np.expand_dims(info['target'], axis=0)
            info['target'] = np.matmul(target - self.pca_mu, self.pca_proj_mtx)

    def get_normalizer(self,):
        # compute mean and std
        assert self.mode == 'train'
        targets = [info['target'] for info in self.data_info]
        targets = np.array(targets, dtype=np.float32)
        mu = np.mean(targets, axis=0)
        delta = targets - np.expand_dims(mu, axis=0)
        if self.norm_type == 'l1':
            std = np.mean(np.abs(delta), axis=0)
        elif self.norm_type == 'l2':
            std = np.sqrt(np.mean(np.square(delta), axis=0))
        else:
            error('unknown norm type')

        return mu, std

    def normalize(self,):
        for info in self.data_info:
            info['target'] = (info['target'] - self.norm_mu) / self.norm_std
            info['target'] = info['target'].flatten()

    def prepare(self, idx):
        info = self.data_info[idx]

        # voice
        voice_path = info['voice_path']
        voice_info = torchaudio.info(voice_path)
        num_frames = voice_info.num_frames
        max_num_frames = self.duration[1] * self.sample_rate

        assert self.sample_rate == voice_info.sample_rate
        assert num_frames >= max_num_frames

        frame_offset = np.random.randint(
                num_frames - max_num_frames, size=1)
        frame_offset = np.asscalar(frame_offset)

        voice, _ = torchaudio.load(
                voice_path, frame_offset, max_num_frames)
        #c = np.random.randint(2)
        voice = voice[0]

        # face
        target = info['target']

        # gender ancestry
        gender = info['gender']
        ancestry = info['ancestry']

        return idx, voice, target, gender, ancestry

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        return self.prepare(idx)

