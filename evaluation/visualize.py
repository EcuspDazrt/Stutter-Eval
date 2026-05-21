import matplotlib.pyplot as plt
import numpy as np

from model.compute_anomalies import frame_to_time

def plot_anomalies(audio_array: np.ndarray, smoothed: dict, flagged_frames: list, sr=16000):
    audio_times = np.linspace(0, len(audio_array)/sr, len(audio_array))
    frame_times = np.array([frame_to_time(i) for i in range(len(smoothed['mfcc']))])

    fig, axes = plt.subplots(4, 1, figsize=(14,10), sharex=True)

    axes[0].plot(audio_times, audio_array, color='steelblue', linewidth=0.5)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Waveform')

    for ax, key, color in zip(axes[1:], ['mfcc', 'delta', 'delta2'], ['orange', 'green', 'red']):
        ax.plot(frame_times, smoothed[key], color=color)
        ax.set_ylabel(f'{key} error')

    for i in flagged_frames:
        start = frame_to_time(i)
        end = frame_to_time(i+1)
        for ax in axes:
            ax.axvspan(start, end, alpha=0.3, color='red')


    axes[-1].set_xlabel('Time (seconds)')
    plt.tight_layout()
    plt.savefig('anomalies.png', dpi=150)
    plt.show()
