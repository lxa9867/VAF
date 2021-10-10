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
                 face_mean_path=None,
                 face_std_path=None):
        super(PenstateDataset, self).__init__()
        self.ann_file = ann_file
        self.seed = seed
        self.split = split
        self.mode = mode
        self.sample_rate = sample_rate
        self.duration = duration

        self.data_info = self.get_data()
        if face_mean_path is None or face_std_path is None:
            self.face_mean, self.face_std = self.get_face_normalizer()
        else:
            self.face_mean = np.loadtxt(face_mean_path)
            self.face_std = np.loadtxt(face_std_path)

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
        #assert {info['gender'] for info in data_info[:pt1]} == genders
        #assert {info['ancestry'] for info in data_info[:pt1]} == ancestries

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
            #assert len({info['gender'] for info in data_info}) == 2
        elif mode == 'val':
            data_info = data_info[pt1:pt2]
        elif mode == 'eval':
            data_info = data_info[pt2:]

        return data_info

    def get_face_normalizer(self,):
        # compute mean and std
        assert self.mode == 'train'
        faces = [info['face'] for info in self.data_info]
        faces = np.array(faces, dtype=np.float32)
        face_mean = np.mean(faces, axis=0, keepdims=False)
        face_delta = faces - np.expand_dims(face_mean, axis=0)
        face_dist = np.sum(np.square(face_delta), axis=1, keepdims=True)
        face_std = np.sqrt(np.mean(face_dist, axis=0, keepdims=False))
        #face_std = np.mean(np.abs(face_delta), axis=0)

        return face_mean, face_std

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
        face = info['face']
        face = (face - self.face_mean) / self.face_std

        # gender ancestry
        gender = info['gender']
        ancestry = info['ancestry']

        return idx, voice, face, gender, ancestry

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        return self.prepare(idx)

