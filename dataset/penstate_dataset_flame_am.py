import numpy as np
import random
import torchaudio

from torch.utils.data import Dataset


class PenstateDatasetFlameAM(Dataset):
    """ An example of creating a dataset from a given data_info.
    """
    def __init__(self,
                 ann_file=None,
                 seed=None,
                 split=None,
                 mode=None,
                 sample_rate=None,
                 duration=None,
                 measure_indices=None,
                 norm_type=None,
                 norm_mu_path=None,
                 norm_std_path=None,
                 gender=None,
                 ancestry=None):
        super(PenstateDatasetFlameAM, self).__init__()
        self.ann_file = ann_file
        self.seed = seed
        self.split = split
        self.mode = mode
        self.sample_rate = sample_rate
        self.measure_indices = measure_indices
        self.duration = duration
        self.norm_type = norm_type

        # target: face
        self.data_info = self.get_data()
        self.get_measurements()

        mat = []
        for info in self.data_info:
            mat.append(info['target'])
        np.savetxt('mat.txt', np.array(mat), fmt='%.5f')
        xxxx

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
        '''
        if mode == 'train':
            data_info = data_info[:pt1]
            #data_info = data_info[:64]
        elif mode == 'val':
            data_info = data_info[pt1:pt2]
        elif mode == 'eval':
            data_info = data_info[pt2:]
        '''
        # load face data
        for info in data_info:
            face_path = info['face_path']
            info['face'] = np.loadtxt(face_path, dtype=np.float32)
            info['target'] = info['face']

        return data_info

    def get_measurements(self):
        index = [2570, 2569, 2564, 3675, 2582,
                 1445, 3848, 1427, 1432, 1433,
                 2428, 2451, 2495, 2471, 3638, 2276, 2355, 2359,
                 3835, 1292, 1344, 1216, 1154, 999, 991, 4046,
                 3704, 3553, 3561, 3501, 3564,
                 2747, 1613, 1392, 3527, 471, 480, 1611,
                 3797, 2864, 2811, 3543, 1694, 1749,
                 3920, 2881, 2905, 1802, 1774, 3503,
                 3515, 3502, 3401, 3399, 3393,
                 3385, 1962, 3381, 3063, 3505,
                 3595, 3581, 3577, 2023, 567]

        def get_distance(target, idx1, idx2):
            diff = target[idx1, :] - target[idx2, :]
            distance = np.sqrt(np.sum(diff * diff))
            return distance
        
        def get_proportion(target, idx1, idx2, idx3, idx4):
            dist1 = get_distance(target, idx1, idx2)
            dist2 = get_distance(target, idx3, idx4)
            return dist1 / dist2

        def get_angle(target, idx1, idx2, idx3, rv=[0, 0, 1]):
            v1 = target[idx1, :] - target[idx2, :]
            v2 = target[idx3, :] - target[idx2, :]
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

        for info in self.data_info:
            target = info['target']
            target[4842, :] = 0.5 * (target[4842, :] + target[4955, :])
            target[1839, :] = 0.5 * (target[1839, :] + target[1950, :])
            measurements = [
                # 0-21
                get_distance(target, index[10], index[14]),
                get_distance(target, index[18], index[22]),
                get_distance(target, index[14], index[18]),
                get_distance(target, index[10], index[22]),
                get_distance(target, index[4], index[5]),
                get_distance(target, index[0], index[9]),
                get_distance(target, index[2], index[7]),
                get_distance(target, index[58], index[60]),
                get_distance(target, index[57], index[61]),
                get_distance(target, index[56], index[62]),
                get_distance(target, index[55], index[63]),
                get_distance(target, index[54], index[64]),
                get_distance(target, index[31], index[37]),
                get_distance(target, index[32], index[36]),
                get_distance(target, index[33], index[35]),
                get_distance(target, index[48], index[46]),
                get_distance(target, index[40], index[42]),
                get_distance(target, index[54], index[53]),
                get_distance(target, index[53], index[64]),
                get_distance(target, index[38], index[44]),
                get_distance(target, index[48], index[46]),
                get_distance(target, index[39], index[43]),

                # 22-49
                get_distance(target, index[41], index[47]),
                get_distance(target, index[12], index[16]),
                get_distance(target, index[20], index[24]),
                get_distance(target, index[59], index[27]),
                get_distance(target, index[59], index[30]),
                get_distance(target, index[59], index[34]),
                get_distance(target, index[59], index[50]),
                get_distance(target, index[59], index[51]),
                get_distance(target, index[59], index[53]),
                get_distance(target, index[27], index[30]),
                get_distance(target, index[26], index[50]),
                get_distance(target, index[30], index[34]),
                get_distance(target, index[30], index[51]),
                get_distance(target, index[30], index[52]),
                get_distance(target, index[30], index[53]),
                get_distance(target, index[34], index[50]),
                get_distance(target, index[41], index[47]),
                get_distance(target, index[51], index[52]),
                get_distance(target, index[52], index[53]),
                get_distance(target, index[50], index[53]),
                get_distance(target, index[54], index[55]),
                get_distance(target, index[55], index[56]),
                get_distance(target, index[56], index[57]),
                get_distance(target, index[57], index[58]),
                get_distance(target, index[60], index[61]),
                get_distance(target, index[61], index[62]),
                get_distance(target, index[62], index[63]),
                get_distance(target, index[63], index[64]),

                # 50-63
                get_proportion(target, index[31], index[37], index[27], index[30]),
                get_proportion(target, index[32], index[36], index[27], index[30]),
                get_proportion(target, index[57], index[61], index[27], index[30]),
                get_proportion(target, index[56], index[62], index[27], index[30]),
                get_proportion(target, index[55], index[63], index[27], index[30]),
                get_proportion(target, index[54], index[64], index[27], index[30]),
                get_proportion(target, index[38], index[44], index[27], index[30]),
                get_proportion(target, index[31], index[37], index[59], index[53]),
                get_proportion(target, index[32], index[36], index[59], index[53]),
                get_proportion(target, index[57], index[61], index[59], index[53]),
                get_proportion(target, index[56], index[62], index[59], index[53]),
                get_proportion(target, index[55], index[63], index[59], index[53]),
                get_proportion(target, index[54], index[64], index[59], index[53]),
                get_proportion(target, index[38], index[44], index[59], index[53]),

                # 64-77
                get_proportion(target, index[58], index[60], index[57], index[61]),
                get_proportion(target, index[57], index[61], index[56], index[62]),
                get_proportion(target, index[56], index[62], index[55], index[63]),
                get_proportion(target, index[55], index[63], index[54], index[64]),
                get_proportion(target, index[57], index[61], index[31], index[37]),
                get_proportion(target, index[56], index[62], index[31], index[37]),
                get_proportion(target, index[55], index[63], index[31], index[37]),
                get_proportion(target, index[58], index[60], index[31], index[37]),
                get_proportion(target, index[54], index[64], index[31], index[37]),
                get_proportion(target, index[38], index[44], index[31], index[37]),
                get_proportion(target, index[59], index[53], index[27], index[30]),
                get_proportion(target, index[51], index[52], index[52], index[53]),
                get_proportion(target, index[50], index[51], index[50], index[53]),
                get_proportion(target, index[27], index[30], index[27], index[53]),

                # 78-86
                get_angle(target, index[56], index[57], index[58], [ 0.8323, -0.0272, -0.5424]),
                get_angle(target, index[55], index[56], index[57], [ 0.2037, -0.2008, -0.8719]),
                get_angle(target, index[54], index[55], index[56], [-0.7313,  0.0418,  0.6758]),
                get_angle(target, index[53], index[54], index[55], [-0.2549,  0.7273,  0.6255]),
                get_angle(target, index[64], index[53], index[54], [-0.2546,  0.7300,  0.6207]),
                get_angle(target, index[63], index[64], index[53], [ 0.7108, -0.0163,  0.7008]),
                get_angle(target, index[62], index[63], index[64], [ 0.8742,  0.0477,  0.4653]),
                get_angle(target, index[61], index[62], index[63], [-0.7094, -0.0442, -0.6913]),
                get_angle(target, index[60], index[61], index[62], [-0.7399, -0.0322, -0.6663]),
                
                # 87-95
                get_angle(target, index[31], index[29], index[37], [ 0.0176, -0.6810,  0.7289]),
                get_angle(target, index[31], index[30], index[37], [ 0.0200, -0.9561,  0.2843]),
                get_angle(target, index[27], index[30], index[34], [-0.9999, -0.0077,  0.0116]),
                get_angle(target, index[51], index[52], index[53], [-0.9937, -0.0307,  0.0117]),
                get_angle(target, index[27], index[30], index[31], [-0.7941,  0.3594,  0.4849]),
                get_angle(target, index[37], index[30], index[27], [ 0.7950,  0.3753,  0.4710]),
                get_angle(target, index[37], index[50], index[31], [-0.0076,  0.6438,  0.7614]),
                get_angle(target, index[63], index[53], index[55], [-0.3671,  0.6032,  0.7010]),
                get_angle(target, index[29], index[30], index[34], [-0.9999,  0.0001,  0.0037]),
            ]
            # measurements = [measurements[idx] for idx in self.measure_indices]
            info['target'] = np.array(measurements, dtype=np.float32).flatten()

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

