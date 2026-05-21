import torch

from model.compute_anomalies import compute_anomaly_scores, smooth_deviations


def infer(clip, model): # clip is in the form (n_frames, features)
    model.eval()

    clip = clip.T

    predicted_frames = []
    actual_frames = []
    with torch.no_grad():
        for i in range(10, len(clip)):
            window = clip[i-10:i]
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