from data_processing.download_data import *
from data_processing.extract_features import process_features
import util.paths as p

def run_pipeline():
    if not p.commonvoice_downloaded():
        download_commonvoice()

    if not p.librosa_downloaded():
        download_librispeech()

    process_features(exclude_commonvoice=True)

if __name__ == '__main__':
    run_pipeline()