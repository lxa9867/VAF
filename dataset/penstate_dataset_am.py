import numpy as np
import random
import torchaudio

from torch.utils.data import Dataset
from datetime import datetime


class PenstateDatasetAM(Dataset):
    """ An example of creating a dataset from a given data_info.
    """
    def __init__(self, ann_path, gender,
                 seed, train_seed, split, mode, 
                 sample_rate, duration, measure_indices, 
                 norm_type=None, norm_mu_path=None, norm_std_path=None):
        super(PenstateDatasetAM, self).__init__()
        self.ann_path = ann_path
        self.gender = gender
        self.seed = seed
        self.train_seed = train_seed
        self.split = split
        self.mode = mode
        self.sample_rate = sample_rate
        self.duration = duration
        self.measure_indices = measure_indices
        self.norm_type = norm_type

        self.face_indices = [
            5732, 5471, 5120, 4662, 4212, 
            2462, 2084, 1686, 1352, 1106,
            5366, 5062, 4884, 4711, 4256, 4720, 4842, 5078,
            2415, 2066, 1904, 1718, 1389, 1728, 1839, 2065,
            3358, 3362, 3418, 3389, 3371, 
            4272, 4049, 3706, 3378, 3094, 2701, 2478, 
            4680, 4336, 3759, 3397, 3091, 2700, 
            2214, 2588, 2926, 3393, 3887, 4311, 
            3388, 3411, 3459, 3493, 4721,
            6489, 6769, 6778, 6512, 3246,
            186, 0, 57, 437, 2096,
        ]

        self.measurement_info = [
            # 0 - 24
            ['dist', 10, 14],
            ['dist', 18, 22],
            ['dist', 14, 18],
            ['dist', 10, 22],
            ['dist', 4, 5],
            ['dist', 0, 9],
            ['dist', 2, 7],
            ['dist', 58, 60],
            ['dist', 57, 61],
            ['dist', 56, 62],
            ['dist', 55, 63],
            ['dist', 54, 64],
            ['dist', 31, 37], # picked
            ['dist', 32, 36], # picked
            ['dist', 33, 35],
            ['dist', 48, 46],
            ['dist', 40, 42], # picked
            ['dist', 54, 53],
            ['dist', 53, 64],
            ['dist', 38, 44],
            ['dist', 48, 46],
            ['dist', 39, 43],
            ['dist', 41, 47],
            ['dist', 12, 16],
            ['dist', 20, 24],

            # 25 - 49
            ['dist', 59, 27],
            ['dist', 59, 30],
            ['dist', 59, 34],
            ['dist', 59, 50],
            ['dist', 59, 51],
            ['dist', 59, 53],
            ['dist', 27, 30],
            ['dist', 26, 50],
            ['dist', 30, 34],
            ['dist', 30, 51],
            ['dist', 30, 52],
            ['dist', 30, 53],
            ['dist', 34, 50],
            ['dist', 41, 47],
            ['dist', 51, 52],
            ['dist', 52, 53],
            ['dist', 50, 53],
            ['dist', 54, 55],
            ['dist', 55, 56],
            ['dist', 56, 57],
            ['dist', 57, 58],
            ['dist', 60, 61],
            ['dist', 61, 62],
            ['dist', 62, 63],
            ['dist', 63, 64],

            # 50 - 63
            ['prop', 31, 37, 27, 30],
            ['prop', 32, 36, 27, 30],
            ['prop', 57, 61, 27, 30],
            ['prop', 56, 62, 27, 30],
            ['prop', 55, 63, 27, 30],
            ['prop', 54, 64, 27, 30],
            ['prop', 38, 44, 27, 30],
            ['prop', 31, 37, 59, 53],
            ['prop', 32, 36, 59, 53],
            ['prop', 57, 61, 59, 53],
            ['prop', 56, 62, 59, 53],
            ['prop', 55, 63, 59, 53],
            ['prop', 54, 64, 59, 53],
            ['prop', 38, 44, 59, 53],

            # 64 - 77
            ['prop', 58, 60, 57, 61],
            ['prop', 57, 61, 56, 62],
            ['prop', 56, 62, 55, 63],
            ['prop', 55, 63, 54, 64],
            ['prop', 57, 61, 31, 37],
            ['prop', 56, 62, 31, 37],
            ['prop', 55, 63, 31, 37],
            ['prop', 58, 60, 31, 37],
            ['prop', 54, 64, 31, 37],
            ['prop', 38, 44, 31, 37],
            ['prop', 59, 53, 27, 30],
            ['prop', 51, 52, 52, 53],
            ['prop', 50, 51, 50, 53],
            ['prop', 27, 30, 27, 53],

            # 78 - 86
            ['angle', 56, 57, 58, -0.1011, -0.3078,  0.9344],
            ['angle', 55, 56, 57, -0.9009, -0.1629,  0.3678],
            ['angle', 54, 55, 56, -0.7863, -0.1348,  0.5986],
            ['angle', 53, 54, 55, -0.3654,  0.5084,  0.7771],
            ['angle', 64, 53, 54, -0.0262,  0.9601,  0.2719],
            ['angle', 63, 64, 53,  0.2780,  0.6064,  0.7425],
            ['angle', 62, 63, 64,  0.8007, -0.1164,  0.5832],
            ['angle', 61, 62, 63,  0.9095, -0.1439,  0.3239],
            ['angle', 60, 61, 62,  0.4377, -0.2226,  0.8568],

            # 87 - 95
            ['angle', 31, 29, 37, -0.0054, -0.6474,  0.7596],
            ['angle', 31, 30, 37, -0.0098, -0.9611,  0.2697],
            ['angle', 27, 30, 34, -0.9997, -0.0218, -0.0046],
            ['angle', 51, 52, 53, -0.9893,  0.0432, -0.0409],
            ['angle', 27, 30, 31, -0.7108,  0.4236,  0.5553],
            ['angle', 37, 30, 27,  0.7083,  0.4411,  0.5448],
            ['angle', 37, 50, 31,  0.0048,  0.2632,  0.9614],
            ['angle', 63, 53, 55,  0.0039,  0.8142,  0.5791],
            ['angle', 29, 30, 34, -0.9987, -0.0466,  0.0199],
        ]


        # get data and target
        self.data_items = self.get_data()
        self.get_measurements()

        # target: normalized measurements
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
        pt1 = int(len(data_items) * sum(split[0:1]))
        pt2 = int(len(data_items) * sum(split[0:2]))
        pt3 = int(len(data_items) * sum(split[0:3]))

        random.Random(self.seed).shuffle(data_items)
        train_items = data_items[:pt3]
        test_items = data_items[pt3:]
        random.Random(self.train_seed).shuffle(train_items)
        
        if self.mode == 'train':
            selected_items = train_items[:pt1]
        elif self.mode == 'val':
            selected_items = train_items[pt1:pt2]
        elif self.mode == 'eval':
            selected_items = train_items[pt2:]
        elif self.mode == 'test':
            selected_items = test_items

        
        # load face data
        for items in selected_items:
            face_path = items['face_path']
            items['face'] = np.loadtxt(face_path, dtype=np.float32)

        return selected_items

    def get_measurements(self):

        def get_distance(face, idx1, idx2):
            diff = face[idx1, :] - face[idx2, :]
            distance = np.sqrt(np.sum(diff * diff))
            return distance
        
        def get_proportion(face, idx1, idx2, idx3, idx4):
            dist1 = get_distance(face, idx1, idx2)
            dist2 = get_distance(face, idx3, idx4)
            return dist1 / dist2

        def get_angle(face, idx1, idx2, idx3, rv=[0, 0, 1]):
            v1 = face[idx1, :] - face[idx2, :]
            v2 = face[idx3, :] - face[idx2, :]
            normal = [
                v1[1] * v2[2] - v1[2] * v2[1],
                v1[2] * v2[0] - v1[0] * v2[2],
                v1[0] * v2[1] - v1[1] * v2[0],
            ]
            angle_sign = np.dot(np.array(rv), np.array(normal))
            cos_angle = np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
            angle = np.arccos(cos_angle)
            angle = angle / np.pi * 180.
            if angle_sign < 0:
                angle = 360. - angle
            return angle

        def get_measurement(face, info):
            mtype = info[0]
            idx1 = self.face_indices[info[1]]
            idx2 = self.face_indices[info[2]]
            if mtype == 'dist':
                measurement = get_distance(face, idx1, idx2)
            elif mtype == 'prop':
                idx3 = self.face_indices[info[3]]
                idx4 = self.face_indices[info[4]]
                measurement = get_proportion(face, idx1, idx2, idx3, idx4)
            elif mtype == 'angle':
                idx3 = self.face_indices[info[3]]
                measurement = get_angle(face, idx1, idx2, idx3)
            else:
                raise ValueError('unknown type {}'.format(mtype))

            return measurement

        for item in self.data_items:
            target = item['face']
            target[4842, :] = 0.5 * (target[4842, :] + target[4955, :])
            target[1839, :] = 0.5 * (target[1839, :] + target[1950, :])

            measurements = [
                get_measurement(target, info) for info in self.measurement_info]
            measurements = [measurements[idx] for idx in self.measure_indices]
            item['target'] = np.array(measurements, dtype=np.float32).flatten()


    def get_normalizer(self,):
        # compute mean and std
        assert self.mode == 'train'
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
        else:
            max_num_frames = -1
            frame_offset = 0

        voice, _ = torchaudio.load(
                voice_path, frame_offset, max_num_frames)
        voice = voice[0]

        target = item['target']
        gender = item['gender']
        face = item['face']

        return idx, voice, target, gender, face

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        return self.prepare(idx)

