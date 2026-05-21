import numpy as np
from model.thresholds import load_thresholds

HOP_LENGTH = 512
SR = 16000

def compute_anomaly_scores(predicted, actual):
    mfcc_err = np.mean((predicted[:13] - actual[:13]) ** 2)
    delta_err = np.mean((predicted[13:26] - actual[13:26]) ** 2)
    delta2_err = np.mean((predicted[26:] - actual[26:]) ** 2)

    return {
        "mfcc": mfcc_err,  # unusual spectral content
        "delta": delta_err,  # unusual rate of change
        "delta2": delta2_err,  # unusual acceleration
    }

def smooth_deviations(deviations, window_size=5):
    mfcc_scores = np.array([d['mfcc'] for d in deviations])
    delta_scores = np.array([d['delta'] for d in deviations])
    delta2_scores = np.array([d['delta2'] for d in deviations])

    kernel = np.ones(window_size) / window_size

    return {
        'mfcc': np.convolve(mfcc_scores, kernel, mode='same'),
        'delta': np.convolve(delta_scores, kernel, mode='same'),
        'delta2': np.convolve(delta2_scores, kernel, mode='same'),
    }

def find_anomalies(smoothed):
    flagged = []
    threshold = load_thresholds()
    for i in range(len(smoothed['mfcc'])):
        if (smoothed['mfcc'][i] > threshold['mfcc'] or
            smoothed['delta'][i] > threshold['delta'] or
            smoothed['delta2'][i] > threshold['delta2']):
            flagged.append(i)
    return flagged

def frame_to_time(frame_idx):
    return (frame_idx * HOP_LENGTH) / SR