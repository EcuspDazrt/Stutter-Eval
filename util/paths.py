from pathlib import Path
import glob

# <---------- Path variables ---------->
BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / 'datasets'

PROCESSED_DIR = DATASETS_DIR / 'processed'
TRAIN_FEATURES_DIR = PROCESSED_DIR / 'train'
TEST_FEATURES_DIR = PROCESSED_DIR / 'test'
VALIDATION_FEATURES_DIR = PROCESSED_DIR / 'validation'
RAW_COMMONVOICE_DIR = DATASETS_DIR / 'raw' / 'commonvoice'
RAW_LIBRISPEECH_DIR = DATASETS_DIR / 'raw' / 'librosa'
ARTIFACTS_DIR = DATASETS_DIR / 'artifacts'

MODEL_WEIGHTS_PATH = ARTIFACTS_DIR / 'LSTM_weights100.pt'
THRESHOLDS_PATH = ARTIFACTS_DIR / 'thresholds.json'

TEST_PATH = BASE_DIR / 'test' / 'Stutter_Eval_3.m4a'

ALL_DIRS = [
    DATASETS_DIR,
    PROCESSED_DIR,
    TRAIN_FEATURES_DIR,
    TEST_FEATURES_DIR,
    VALIDATION_FEATURES_DIR,
    RAW_COMMONVOICE_DIR,
    RAW_LIBRISPEECH_DIR,
    ARTIFACTS_DIR,
]


# <---------- Misc Variables ---------->
NUM_RAW_LIBRISPEECH_PARQUETS = 64
NUM_RAW_COMMONVOICE_MP3S = 0 # placeholder for now


# <---------- Retrieve paths ---------->
def get_raw_librispeech_paths():
    return glob.glob(f'{RAW_LIBRISPEECH_DIR}/**/*.parquet', recursive=True)

def get_raw_commonvoice_paths():
    return glob.glob(f'{RAW_COMMONVOICE_DIR}/**/*.mp3', recursive=True)

def get_processed_train_paths():
    return glob.glob(f'{TRAIN_FEATURES_DIR}/*.npy')

def get_processed_test_paths():
    return glob.glob(f'{TEST_FEATURES_DIR}/*.npy')

def get_processed_validation_paths():
    return glob.glob(f'{VALIDATION_FEATURES_DIR}/*.npy')



# <---------- Pathing checks ---------->
def librispeech_downloaded() -> bool:
    raw_librosa_paths = get_raw_librispeech_paths()

    if len(raw_librosa_paths) > NUM_RAW_LIBRISPEECH_PARQUETS:
        print('Extra Librosa Paths Detected')

    return len(raw_librosa_paths) >= NUM_RAW_LIBRISPEECH_PARQUETS

def commonvoice_downloaded() -> bool:
    raw_commonvoice_paths = get_raw_commonvoice_paths()

    if len(raw_commonvoice_paths) > NUM_RAW_COMMONVOICE_MP3S:
        print('Extra Commonvoice Paths Detected')

    return len(raw_commonvoice_paths) >= NUM_RAW_COMMONVOICE_MP3S

def normalizer_exists():
    if Path(ARTIFACTS_DIR / 'norm_mean.npy').exists() and Path(ARTIFACTS_DIR / 'norm_std.npy').exists():
        return True
    try:
        import util.normalizer as n
        mean, std = n.compute_normalizer()
        n.save_normalizer(mean, std)
    except Exception as e:
        print(f'[ERROR] Could not create normalizer: {e}')
        return False
    return True



# <---------- Check Directory Creation ---------->
def make_dirs():
    for d in ALL_DIRS:
        if d.exists():
            continue
        Path.mkdir(d, parents=True, exist_ok=True)

make_dirs()

if __name__ == '__main__':
    print(len(get_raw_librispeech_paths()))
