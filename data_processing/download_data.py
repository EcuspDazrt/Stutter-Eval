from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import os, requests, tarfile
import util.paths as p

load_dotenv()


TOKEN = os.environ.get('HUGGINGFACE_TOKEN')
API_KEY = os.environ.get('MOZILLA_API_KEY')

def download_librispeech():
    print('Downloading Librispeech...')
    snapshot_download(
        repo_id="openslr/librispeech_asr",
        repo_type="dataset",
        local_dir=p.RAW_LIBRISPEECH_DIR,
        allow_patterns="clean/*",
        token=TOKEN,
    )

    print('Done')

def download_commonvoice():
    tar_file = p.RAW_COMMONVOICE_DIR / 'common_voice_english.tar.gz'

    print('Downloading CommonVoice...')

    response = requests.post(
        "https://mozilladatacollective.com/api/datasets/cmndapwry02jnmh07dyo46mot/download",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
    )

    download_url = response.json()["downloadUrl"]

    file_response = requests.get(download_url, stream=True)
    with open(tar_file, "wb") as f:
        for chunk in file_response.iter_content(chunk_size=8192):
            f.write(chunk)

    # extracts the actual files into the folder
    with tarfile.open(tar_file, "r:gz") as tar:
        tar.extractall(p.RAW_COMMONVOICE_DIR)

    print("Done")

if __name__ == '__main__':
    download_librispeech()
    download_commonvoice()
