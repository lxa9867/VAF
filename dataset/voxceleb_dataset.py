import numpy as np
import random
import torchaudio

from torch.utils.data import Dataset

class VoxcelebDataset(Dataset):
    """ An example of creating a dataset from a given data_info.
    """
    def __init__(self,
                 ann_file=None,
                 seed=None,
                 mode=None,
                 split=None,
                 sample_rate=None,
                 duration=None):
        super(VoxcelebDataset, self).__init__()
        self.ann_file = ann_file
        self.sample_rate = sample_rate
        self.duration = duration

        self.pid_list, self.data_info = self.get_data()

    def get_data(self,):
        ann_file = self.ann_file
          
        with open(ann_file, 'r') as f:
            lines = f.readlines()

        pids = set()
        genders = set()
        data_info = {}
        for line in lines:
            pid, gender, voice_path = line.rstrip().split(' ')
            pids.add(pid)
            genders.add(gender)
            if pid not in data_info:
                data_info[pid] = []
            info = {
                'voice_path': voice_path,
                'gender': gender,
            }
            data_info[pid].append(info)

        pid_list = list(pids)

        # to label
        gender2label = dict(zip(list(genders), range(len(genders))))
        gender2label = {'m': 0, 'f': 1}
        for pid in data_info:
            for info in data_info[pid]:
                gender = info['gender']
                info['gender'] = gender2label[gender]
        print('voxceleb:', gender2label)
        genders = [info['gender'] for pid in data_info for info in data_info[pid]]
        print('vox:', genders.count(0), genders.count(1))

        return pid_list, data_info

    def prepare(self, idx):
        pid = self.pid_list[idx]
        idx1, idx2 = np.random.choice(
                len(self.data_info[pid]), 2, replace=False)

        # voice
        voice_path1 = self.data_info[pid][idx1]['voice_path']
        voice_info1 = torchaudio.info(voice_path1)
        voice_path2 = self.data_info[pid][idx2]['voice_path']
        voice_info2 = torchaudio.info(voice_path2)
        assert self.sample_rate == voice_info1.sample_rate
        assert self.sample_rate == voice_info2.sample_rate

        if True:#self.mode == 'train':
            max_num_frames = self.duration[1] * self.sample_rate
            assert voice_info1.num_frames >= max_num_frames
            assert voice_info2.num_frames >= max_num_frames

            frame_offset1 = np.random.randint(
                    voice_info1.num_frames - max_num_frames, size=1)
            frame_offset1 = np.asscalar(frame_offset1)
            frame_offset2 = np.random.randint(
                    voice_info2.num_frames - max_num_frames, size=1)
            frame_offset2 = np.asscalar(frame_offset2)
        else:
            max_num_frames = num_frames
            frame_offset = 0

        voice1, _ = torchaudio.load(
                voice_path1, frame_offset1, max_num_frames)
        voice1 = voice1[0]
        voice2, _ = torchaudio.load(
                voice_path2, frame_offset2, max_num_frames)
        voice2 = voice2[0]


        # gender ancestry
        gender = self.data_info[pid][idx1]['gender']
        assert self.data_info[pid][idx2]['gender'] == gender

        return voice1, voice2, gender

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        return self.prepare(idx)

