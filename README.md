# Stutter-Eval - Unsupervised Disfluency Detection

An unsupervised speech disfluency detection system using an autoregressive LSTM trained exclusively on fluent speech. The model learns to predict the next acoustic frame from a short context window; at inference, high reconstruction error flags regions that deviate from fluent speech patterns (stutters, repetitions, prolongations, etc.)

---

## How It Works

The system is an **anomaly detector**, not a classifier. It is trained on fluent speech only, so it builds a statistical model of normal phonetic transitions. When it encounters a disfluency, the MFCC reconstruction error spikes above a calibrated threshold, and those frames get flagged.

```
  Raw Audio (LibriSpeech clean)
        │
        ▼
  Feature Extraction
  13 MFCCs + Δ + ΔΔ → 39-dim per frame
        │
        ▼
  Global Normalisation
  Welford mean/std computed on training set
        │
        ▼
  LSTM Training
  Predict frame t from frames t-9…t-1, MSE loss
        │
        ▼
  Threshold Calibration
  95th percentile of validation reconstruction error
        │
        ▼
  Inference on new audio
  Flag frames where error > threshold
        │
        ▼
  Visualisation
  Waveform + per-component error plots with flagged regions
```

---

## Project Structure

```
root/
├── data_processing/
│   ├── download_data.py      # Downloads LibriSpeech (HuggingFace) and CommonVoice
│   ├── extract_features.py   # MFCC + delta + delta-delta extraction, saves per-clip .npy files
│   └── run.py                # End-to-end data pipeline entry point
│
├── model/
│   ├── definition.py         # StutterAnomalyLSTM architecture
│   ├── train.py              # Training loop (multi-epoch, GPU-aware)
│   ├── inference.py          # Frame-by-frame prediction and deviation scoring
│   ├── compute_anomalies.py  # Anomaly scoring, temporal smoothing, flagging
│   └── thresholds.py         # Percentile threshold computation and persistence
│
├── evaluation/
│   ├── test_model.py         # Runs the full inference pipeline on a test audio file
│   └── visualize.py          # Waveform + MFCC/delta/delta² error plots with flagged regions
│
├── util/
│   ├── dataset.py            # SpeechDataset: memory-mapped sliding-window PyTorch Dataset
│   ├── normalizer.py         # Welford normaliser: compute, save, load, apply
│   └── paths.py              # Centralised path constants and directory setup
│
└── datasets/                 # Not committed, created automatically at runtime
    ├── raw/
    │   ├── librispeech/
    │   └── commonvoice/
    ├── processed/
    │   ├── train.npy
    │   ├── validation.npy
    │   └── test.npy
    └── artifacts/
        ├── LSTM_weights.pt
        ├── thresholds.json
        ├── norm_mean.npy
        └── norm_std.npy
```

---

## Setup

### Python Version

Python 3.13. Several stdlib modules removed in 3.13 (`audioop`, `aifc`, `sunau`) are restored via backport packages already included in `requirements.txt`.

### Install Dependencies

```bash
pip install -r requirements.txt
```

### GPU Support

The training and inference code automatically uses CUDA if available. Verify your installation:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If this prints `False`, you have the CPU-only build of PyTorch. Reinstall with the appropriate CUDA version:

```bash
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu124  # CUDA 12.4
```

Match the `cu124` suffix to your driver's CUDA version, visible in the top-right of `nvidia-smi` output.

### Environment Variables

Create a `.env` file in the project root:

```env
HUGGINGFACE_TOKEN=your_token_here
MOZILLA_API_KEY=your_key_here      # only needed for CommonVoice (see Roadmap)
```

---

## Usage 
*(1-4 are only for reproducibility. The artifacts that come from them are already committed.)*

### 1. Download and process data

```bash
python data_processing/run.py
```

Downloads LibriSpeech (clean subset, ~460 hours) via HuggingFace and extracts 39-dimensional features into per-clip `.npy` files under `datasets/processed/`.

### 2. Compute the normaliser

```bash
python util/normalizer.py
```

Computes per-feature mean and standard deviation across all training clips using Welford's online algorithm (constant memory), and saves them to `datasets/artifacts/`. **Must be run before training.**

### 3. Train the model

```bash
python model/train.py
```

Trains for 100 epochs by default. Saves weights to `datasets/artifacts/LSTM_weights.pt`. Automatically uses GPU if available. Recommended batch size: 1024.

### 4. Calibrate thresholds

```bash
python model/thresholds.py
```

Runs inference over the validation set and saves the 95th-percentile reconstruction error per feature group to `datasets/artifacts/thresholds.json`. **Must be re-run any time the model weights or normaliser change.**

### 5. Test on an audio file

Place a `.m4a` file at `test/stutter_eval.m4a` (or update `TEST_PATH` in `paths.py`), then:

```bash
python evaluation/test_model.py
```

Saves `anomalies.png`, a four-panel plot showing the waveform and MFCC/delta/delta² reconstruction errors with flagged regions overlaid in red.

---

## Model Architecture

`StutterAnomalyLSTM` is a 2-layer LSTM with a linear projection head:

| Parameter   | Value                          |
|-------------|--------------------------------|
| Input size  | 39 (13 MFCCs + Δ + ΔΔ)        |
| Hidden size | 64                             |
| Layers      | 2                              |
| Output size | 39                             |
| Loss        | MSE (next-frame prediction)    |
| Optimiser   | Adam, lr=1e-3                  |

The model is trained to predict frame `t` given frames `t-9 … t-1`. At inference, the mean squared error between predicted and actual frames is computed separately for each feature group (MFCC, Δ, ΔΔ).

---

## Anomaly Detection

For each frame, reconstruction error is split into three components:

| Component | Coefficient indices | Captures                        |
|-----------|--------------------|---------------------------------|
| MFCC      | 0–12               | Unusual spectral content        |
| Delta     | 13–25              | Unusual rate of phonetic change |
| Delta²    | 26–38              | Unusual acceleration            |

The three error streams are independently smoothed with a 5-frame box filter, then compared against thresholds calibrated from the 95th percentile of validation error. A frame is flagged if **any** component exceeds its threshold.

---

## Data Sources

| Dataset | Description | Access                                                                                    |
|---------|-------------|-------------------------------------------------------------------------------------------|
| LibriSpeech (clean) | Read English speech, ~460 hours | [HuggingFace](https://huggingface.co/datasets/openslr/librispeech_asr) - requires account |
| Mozilla CommonVoice | Crowd-sourced English speech | Mozilla Data Collective API key - planned, see Roadmap                                    |

---

## Roadmap

- **CommonVoice integration** - download and extraction pipeline is implemented in `data_processing/download_data.py` and `extract_features.py` but currently excluded from training. Pending fix to `paths.py` download checks and validation of the CommonVoice API endpoint.
- **Labelled evaluation** - compute precision, recall, and F1 against a ground-truth stuttered speech corpus (SEP-28k, FluencyBank, or UCLASS).
- **Feature normalisation per utterance** - add CMVN (Cepstral Mean and Variance Normalisation) as a preprocessing step before global normalisation, to handle recording condition variation at inference.

---

## License

MIT - free to use, modify, and distribute.
