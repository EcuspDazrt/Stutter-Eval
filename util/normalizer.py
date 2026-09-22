import numpy as np
from pathlib import Path
import util.paths as p

def compute_normalizer(features_dir):
    clip_paths = sorted(Path(features_dir).glob('*.npy'))

    mean = np.zeros(39)
    M2 = np.zeros(39)
    count = 0

    for path in clip_paths:
        clip = np.load(path, mmap_mode='r').T  # (n_frames, 39)
        for frame in clip:
            count += 1
            delta = frame - mean
            mean += delta / count
            M2 += delta * (frame - mean)

    std = np.sqrt(M2 / (count - 1))
    return mean, std

def save_normalizer(mean, std):
    np.save(p.ARTIFACTS_DIR / 'norm_mean.npy', mean)
    np.save(p.ARTIFACTS_DIR / 'norm_std.npy', std)

def load_normalizer():
    mean = np.load(p.ARTIFACTS_DIR / 'norm_mean.npy')
    std = np.load(p.ARTIFACTS_DIR / 'norm_std.npy')
    return mean, std

def normalize(features, mean, std):
    return (features - mean) / (std + 1e-8)

if __name__ == "__main__":
    mean, std = compute_normalizer(p.TRAIN_FEATURES_DIR)
    save_normalizer(mean, std)