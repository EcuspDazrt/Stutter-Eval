import numpy as np
import util.paths as p

from torch import tensor, float32
from definition import StutterAnomalyLSTM

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
        'mfcc': np.convolve(mfcc_scores, kernal, mode='same'),
        'delta': np.convolve(delta_scores, kernel, mode='same'),
        'delta2': np.convolve(delta2_scores, kernel, mode='same'),
    }


def infer(clip): # clip is in the form (n_frames, features)
    model = StutterAnomalyLSTM()
    model.load_state_dict(torch.load(p.MODEL_WEIGHTS_PATH))
    model.eval()

    predicted_frames = []
    actual_frames = []
    with torch.no_grad():
        for i in range(10, len(clip)):
            window = clip[i-10:i]
            window = tensor(window, dtype=float32).unsqueeze(0)

            prediction = model(window)
            predicted_next = prediction[0, -1, :].detach().numpy()
            actual = clip[i]

            predicted_frames.append(predicted_next)
            actual_frames.append(actual)

    deviations = []
    for x, y in zip(predicted_frames, actual_frames):
        deviation = compute_anomaly_scores(x, y)
        smooth_deviation = smooth_deviations(deviation)

        deviations.append((deviation, smooth_deviation))

    return deviations