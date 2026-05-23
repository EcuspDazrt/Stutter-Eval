import torch

from torch.utils.data import Dataset
from util.normalizer import load_normalizer, normalize

import numpy as np
import util.paths as p

class SpeechDataset(Dataset):
    def __init__(self, features_path=p.PROCESSED_DIR / 'train.npy', window_size=10):
        self.data = np.load(features_path, allow_pickle=True)
        self.window_size = window_size

        self.clip_frames = [clip.shape[1] - window_size for clip in self.data]
        self.cumulative = np.cumsum([0] + self.clip_frames)

        if not p.normalizer_exists():
            raise Exception('Normalizer does not exist; generation failed.')

        self.mean, self.std = load_normalizer()

    def __len__(self):
        return int(self.cumulative[-1])

    def __getitem__(self, idx):
        clip_idx = np.searchsorted(self.cumulative[1:], idx, side='right')
        frame_idx = self.window_size + (idx - self.cumulative[clip_idx])

        clip = self.data[clip_idx].T
        window = clip[frame_idx - self.window_size:frame_idx]
        window = normalize(window, self.mean, self.std)
        return torch.tensor(window, dtype=torch.float32)