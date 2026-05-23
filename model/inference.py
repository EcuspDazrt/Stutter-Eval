import torch

from model.compute_anomalies import compute_anomaly_scores, smooth_deviations
from util.normalizer import load_normalizer, normalize


def infer(clip, model):
    mean, std = load_normalizer()
    model.eval()

    clip = clip.T                     # (39, n_frames) -> (n_frames, 39)
    clip = normalize(clip, mean, std) # (n_frames, 39)

    predicted_frames = []
    actual_frames = []
    with torch.no_grad():
        for i in range(9, len(clip)):
            window = clip[i-9:i]
            window = torch.tensor(window, dtype=torch.float32).unsqueeze(0)

            prediction = model(window)
            predicted_next = prediction[0, -1, :].detach().numpy()
            actual = clip[i]

            predicted_frames.append(predicted_next)
            actual_frames.append(actual)

    deviations = []
    for x, y in zip(predicted_frames, actual_frames):
        deviation = compute_anomaly_scores(x, y)
        deviations.append(deviation)

    smoothed = smooth_deviations(deviations)

    return deviations, smoothed