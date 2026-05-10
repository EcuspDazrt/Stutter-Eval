import numpy as np
import pandas as pd
import librosa, glob, io
import util.paths as p
from tqdm import tqdm

SR = 16000 # samples per second

def extract_features(sample):
    audio_array = sample["audio"]["array"]
    original_sr = sample["audio"]["sampling_rate"]

    # ensure audio is at 16kHz
    if original_sr != SR:
        audio_array = librosa.resample(audio_array, orig_sr=original_sr, target_sr=SR)

    mfcc = librosa.feature.mfcc(y=audio_array, sr=SR, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta_delta = librosa.feature.delta(mfcc, order=2)

    features = np.concatenate([mfcc, delta, delta_delta], axis=0)
    return features

def extract_commonvoice_split(df):
    features = []
    for _, row in df.iterrows():
        mp3_path = str(p.RAW_COMMONVOICE_DIR / 'en' / 'clips' / row['path'])
        audio_array, sr = librosa.load(mp3_path, sr=16000)
        sample = {'audio':
            {
                'array': audio_array,
                'sampling_rate': sr,
            }
        }

        features.append(extract_features(sample))

    return features

def extract_commonvoice():
    print('Extracting commonvoice...')
    df = pd.read_csv(p.RAW_COMMONVOICE_DIR / 'en' / 'train.tsv', sep='\t')
    train_features = extract_commonvoice_split(df)

    df = pd.read_csv(p.RAW_COMMONVOICE_DIR / 'en' / 'dev.tsv', sep='\t')
    validation_features = extract_commonvoice_split(df)

    df = pd.read_csv(p.RAW_COMMONVOICE_DIR / 'en' / 'test.tsv', sep='\t')
    test_features = extract_commonvoice_split(df)

    print('Done')
    return train_features, validation_features, test_features


def extract_librispeech_split(raw_dir):
    features = []
    audio_files = glob.glob(f'{raw_dir}/*.parquet')
    for parquet in tqdm(audio_files, total=len(audio_files), desc=raw_dir.name):
        df = pd.read_parquet(parquet)

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f'{raw_dir.name} process', leave=False):
            audio_bytes = row["audio"]["bytes"]
            audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)

            sample = {
                "audio": {
                    "array": audio_array,
                    "sampling_rate": sr
                }
            }

            features.append(extract_features(sample))

    return features

def extract_librispeech():
    print('Extracting Librispeech...')
    source_dir = p.RAW_LIBROSA_DIR / 'clean'

    train_100_dir = source_dir / 'train.100'
    train_100_features = extract_librispeech_split(train_100_dir)

    train_360_dir = source_dir / 'train.360'
    train_360_features = extract_librispeech_split(train_360_dir)

    test_dir = source_dir / 'test'
    test_features = extract_librispeech_split(test_dir)

    validation_dir = source_dir / 'validation'
    validation_features = extract_librispeech_split(validation_dir)

    train_features = train_100_features + train_360_features

    print('Done')
    return train_features, validation_features, test_features

def save_features(path, features):
    arr = np.empty(len(features), dtype=object)
    for i, item in enumerate(features):
        arr[i] = item
    np.save(path, arr)

def process_features(exclude_commonvoice=False):
    lib_train, lib_validation, lib_test = extract_librispeech()

    if exclude_commonvoice:
        save_features(p.PROCESSED_DIR / 'train.npy', lib_train)
        save_features(p.PROCESSED_DIR / 'test.npy', lib_test)
        save_features(p.PROCESSED_DIR / 'validation.npy', lib_validation)
        return

    com_train, com_validation, com_test = extract_commonvoice()

    train_features = lib_train + com_train
    validation_features = lib_validation + com_validation
    test_features = lib_test + com_test

    save_features(p.PROCESSED_DIR / 'train.npy', train_features)
    save_features(p.PROCESSED_DIR / 'test.npy', test_features)
    save_features(p.PROCESSED_DIR / 'validation.npy', validation_features)


if __name__ == '__main__':
    process_features(exclude_commonvoice=True)