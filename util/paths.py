from pathlib import Path
import glob

# <---------- Path variables ---------->
BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR.parent / 'datasets'

PROCESSED_DIR = DATASETS_DIR / 'processed'
RAW_COMMONVOICE_DIR = DATASETS_DIR / 'raw' / 'commonvoice'
RAW_LIBROSA_DIR = DATASETS_DIR / 'raw' / 'librosa'
ARTIFACTS_DIR = DATASETS_DIR / 'artifacts'

MODEL_WEIGHTS_PATH = ARTIFACTS_DIR / 'LSTM_weights.pt'



# <---------- Misc Variables ---------->
NUM_RAW_LIBROSA_PARQUETS = 64
NUM_RAW_COMMONVOICE_MP3S = 0 # placeholder for now


# <---------- Pathing checks ---------->
def get_raw_librosa_paths():
    return glob.glob(f'{RAW_LIBROSA_DIR}/**/*.parquet', recursive=True)

def get_raw_commonvoice_paths():
    return glob.glob(f'{RAW_COMMONVOICE_DIR}/**/*.mp3', recursive=True)

def librosa_downloaded() -> bool:
    raw_librosa_paths = get_raw_librosa_paths()

    if len(raw_librosa_paths) > NUM_RAW_LIBROSA_PARQUETS:
        print('Extra Librosa Paths Detected')

    return len(raw_librosa_paths) >= NUM_RAW_LIBROSA_PARQUETS

def commonvoice_downloaded() -> bool:
    raw_commonvoice_paths = get_raw_commonvoice_paths()

    if len(raw_commonvoice_paths) > NUM_RAW_COMMONVOICE_MP3S:
        print('Extra Commonvoice Paths Detected')

    return len(raw_commonvoice_paths) >= NUM_RAW_COMMONVOICE_MP3S



# <---------- Check Directory Creation ---------->
def make_dirs():
    all_dirs = [
        DATASETS_DIR, PROCESSED_DIR, RAW_COMMONVOICE_DIR, RAW_LIBROSA_DIR, ARTIFACTS_DIR
    ]

    for dir in all_dirs:
        if dir.exists():
            continue
        Path.mkdir(dir, parents=True, exist_ok=True)

make_dirs()

if __name__ == '__main__':
    print(len(get_raw_librosa_paths()))
