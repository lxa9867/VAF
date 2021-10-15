import numpy as np
import random
import torchaudio

from torch.utils.data import Dataset

class PenstateDatasetPC(Dataset):
    """ An example of creating a dataset from a given data_info.
    """
    def __init__(self,
                 ann_file=None,
                 seed=None,
                 split=None,
                 mode=None,
                 sample_rate=None,
                 duration=None,
                 mu_path=None,
                 proj_mtx_path=None,
                 pc_mean_path=None,
                 pc_std_path=None):
        super(PenstateDatasetPC, self).__init__()
        self.ann_file = ann_file
        self.seed = seed
        self.split = split
        self.mode = mode
        self.sample_rate = sample_rate
        self.duration = duration

        self.data_info = self.get_data()
        if mu_path is None or proj_mtx_path is None:
            self.mu, self.proj_mtx = self.get_PCA_param()
        else:
            self.mu = np.loadtxt(mu_path)
            self.proj_mtx = np.loadtxt(proj_mtx_path)

        self.face_to_pc()

        if pc_mean_path is None or pc_std_path is None:
            self.pc_mean, self.pc_std = self.get_pc_normalizer()
        else:
            self.pc_mean = np.loadtxt(pc_mean_path)
            self.pc_std = np.loadtxt(pc_std_path)

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
            face = np.loadtxt(face_path).T
            #face = np.random.randn(3, 6790)
            info = {'pid': pid, 'gender': gender,
                    'ancestry': ancestry,
                    'voice_path': voice_path,
                    'face_path': face_path, 'face': face}
            if gender == 'F':
                data_info.append(info)

        # covariate
        genders = {info['gender'] for info in data_info}
        ancestries = {info['ancestry'] for info in data_info}

        # split
        random.Random(seed).shuffle(data_info)
        pt1 = int(len(data_info) * sum(split[0:1]))
        pt2 = int(len(data_info) * sum(split[0:2]))

        # to label
        gender2label = {'F': 0, 'M': 1}
        ancestry2label = dict(zip(list(ancestries), range(len(ancestries))))
        for info in data_info:
            gender = info['gender']
            info['gender'] = gender2label[gender]
            ancestry = info['ancestry']
            info['ancestry'] = ancestry2label[ancestry]

        if mode == 'train':
            data_info = data_info[:pt1]
        elif mode == 'val':
            data_info = data_info[pt1:pt2]
        elif mode == 'eval':
            data_info = data_info[pt2:]

        return data_info

    def get_PCA_param(self,):
        assert self.mode == 'train'
        faces = [info['face'].flatten() for info in self.data_info]
        faces = np.array(faces, dtype=np.float32)
        mu = np.mean(faces, axis=0, keepdims=True)
        faces = faces - mu
        cov_mtx = np.matmul(faces, faces.T)
        w, proj = np.linalg.eig(cov_mtx)
        proj = np.matmul(faces.T, proj[:, 0:1])
        col_norm = np.sqrt(np.sum(proj * proj, axis=0, keepdims=True))
        proj = proj / col_norm

        #reconst_face = np.matmul(np.matmul(faces[0:1, :], proj), proj.T)
        #print(reconst_face.shape, faces[0:1, :].shape)
        #print(np.sum(np.abs(reconst_face - faces[0:1, :])))
        #print(reconst_face[0:10], faces[0:1, 0:10])

        return mu, proj

    def face_to_pc(self,):
        for info in self.data_info:
            face = info['face'].reshape(1, -1)
            pc = np.matmul(face - self.mu, self.proj_mtx)
            info['pc'] = pc.flatten()

    def get_pc_normalizer(self,):
        # compute mean and std
        assert self.mode == 'train'
        pcs = [info['pc'] for info in self.data_info]
        pcs = np.array(pcs, dtype=np.float32)
        pc_mean = np.mean(pcs, axis=0).flatten()
        pc_std = np.std(pcs, axis=0).flatten()

        return pc_mean, pc_std

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
        voice = voice[0]

        # face
        pc = info['pc']
        pc = (pc - self.pc_mean) / self.pc_std

        # gender ancestry
        gender = info['gender']
        ancestry = info['ancestry']

        return idx, voice, pc, gender, ancestry

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        return self.prepare(idx)

