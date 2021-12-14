import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
from librosa.util import fix_length
from transform import aug
from utils import audioProcess


class humDataset(Dataset):
    def __init__(self, df, config, has_transform=True, sub='train'):
        super(humDataset, self).__init__()

        self.df = df
        self.config = config
        self.has_transform = has_transform
        if has_transform:
            self.transform = aug(sub)
        self.sub = sub

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pos_path, hum_path = self.df.loc[idx, ['song_path', 'hum_path']]
        neg_path = np.random.choice(self.df.loc[idx, 'list_negative'])

        pos = librosa.load(pos_path, sr=self.config.sr)[0]
        neg = librosa.load(neg_path, sr=self.config.sr)[0]
        hum = librosa.load(hum_path, sr=self.config.sr)[0]

        max_duration = self.config.max_time * self.config.sr

        # pos
        if pos.shape[0] >= max_duration:
            duration = np.random.randint(0, pos.shape[0] + 1 - max_duration)
            pos = pos[duration:duration + max_duration]
        else:
            pos = fix_length(pos, max_duration)

        # neg
        if neg.shape[0] >= max_duration:
            duration = np.random.randint(0, neg.shape[0] + 1 - max_duration)
            neg = neg[duration:duration + max_duration]
        else:
            neg = fix_length(neg, max_duration)

        # hum
        if hum.shape[0] >= max_duration:
            duration = np.random.randint(0, hum.shape[0] + 1 - max_duration)
            hum = hum[duration:duration + max_duration]
        else:
            hum = fix_length(hum, max_duration)

        if self.has_transform:
            pos = self.transform(data=pos)['data']
            neg = self.transform(data=neg)['data']
            hum = self.transform(data=hum)['data']

        pos_mel = audioProcess(pos, self.config, alg=self.config.alg)
        neg_mel = audioProcess(neg, self.config, alg=self.config.alg)
        hum_mel = audioProcess(hum, self.config, alg=self.config.alg)

        return hum_mel, pos_mel, neg_mel
