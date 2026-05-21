import json
import numpy as np
import util.paths as p
from tqdm import tqdm

def compute_thresholds(validation_data, model, percentile=95):
    from model.inference import infer
    mfcc_scores = []
    delta_scores = []
    delta2_scores = []

    for clip in tqdm(validation_data, total=len(validation_data), desc='Computing thresholds'):
        deviations, smoothed = infer(clip, model)

        mfcc_scores.extend(smoothed['mfcc'])
        delta_scores.extend(smoothed['delta'])
        delta2_scores.extend(smoothed['delta2'])

    return {
        'mfcc': np.percentile(mfcc_scores, percentile),
        'delta': np.percentile(delta_scores, percentile),
        'delta2': np.percentile(delta2_scores, percentile),
    }

# noinspection PyShadowingNames
def save_thresholds(thresholds):
    with open(p.THRESHOLDS_PATH, 'w') as f:
        json.dump(thresholds, f)


# noinspection PyShadowingNames
def load_thresholds():
    with open(p.THRESHOLDS_PATH) as f:
        thresholds = json.load(f)
    return thresholds

if __name__ == '__main__':
    from model.definition import load_lstm
    lstm = load_lstm()
    valid_data = np.load(p.PROCESSED_DIR / 'validation.npy', allow_pickle=True)

    thresholds = compute_thresholds(valid_data, lstm)
    save_thresholds(thresholds)