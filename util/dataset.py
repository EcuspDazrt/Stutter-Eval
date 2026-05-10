import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import util.paths as p

class SpeechDataset(Dataset):
    def __init__(self, features_path=p.PROCESSED_DIR / 'train.npy', window_size=10):
        self.data = np.load(features_path, allow_pickle=True)
        self.window_size = window_size
        self.index = []

        for clip_idx, clip in enumerate(tqdm(self.data, total=len(self.data), desc='Speech Loader Process')):
            num_frames = clip.shape[1]
            for frame_idx in range(window_size, num_frames):
                self.index.append((clip_idx, frame_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        clip_idx, frame_idx = self.index[idx]
        clip = self.data[clip_idx].T
        window = clip[frame_idx - self.window_size:frame_idx]
        return torch.tensor(window, dtype=torch.float32)