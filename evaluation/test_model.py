from model.definition import load_lstm
from model.inference import infer
from model.compute_anomalies import find_anomalies
from data_processing.extract_features import extract_features
from evaluation.visualize import plot_anomalies

from pydub import AudioSegment
import io, librosa
import util.paths as p

def test_model(audio_sample):
    clip = extract_features(audio_sample)
    lstm = load_lstm()

    _, smoothed = infer(clip, lstm)
    flagged = find_anomalies(smoothed)

    audio_array = audio_sample['audio']['array']
    plot_anomalies(audio_array, smoothed, flagged)



def load_m4a(file_path):
    audio = AudioSegment.from_file(file_path, format='m4a')

    buffer = io.BytesIO()
    audio.export(buffer, format='wav')
    buffer.seek(0)

    audio_array, sr = librosa.load(buffer, sr=16000)

    return {
        'audio': {
            'array': audio_array,
            'sampling_rate': sr,
        }
    }


if __name__ == '__main__':
    audio_sample = load_m4a(p.TEST_PATH)
    test_model(audio_sample)