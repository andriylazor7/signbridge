# SignBridge

Real-time sign language word recognition, running in the browser.

SignBridge watches a webcam feed, tracks hand landmarks with MediaPipe, and classifies
the gesture into one of 100 ASL words (the **WLASL100** subset) using a PyTorch model
exported to ONNX and run client-side. Everything runs locally in the browser.

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Repository Structure](#repository-structure)
- [Results](#results)
- [Key Limitation: Data Availability](#key-limitation-data-availability)
- [Getting Started](#getting-started)
  - [Quick Start: Web Demo](#quick-start-web-demo)
  - [Full ML Pipeline](#full-ml-pipeline)
- [Command Reference](#command-reference)
- [Model Details](#model-details)
- [Known Limitations & Roadmap](#known-limitations--roadmap)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

The project has two parts — the same landmark preprocessing
(normalization, sequence length) — implemented twice, once in Python and once in
TypeScript, so the model sees identical input in training and in the browser:

| | |
|---|---|
| **`ml/`** | Data preparation, training, evaluation, ONNX export (Python / PyTorch) |
| **`web/`** | Demo: camera → hand tracking → live word prediction |

**Tech stack:** PyTorch · MediaPipe Tasks Vision · ONNX Runtime (Python + Web) · React · TypeScript · Vite

## How It Works

```
Camera video
      │
      ▼
MediaPipe Hand Landmarker     →  up to 2 hands × 21 points × (x, y, z) per frame
      │
      ▼
Per-hand normalization        →  translate to the wrist, scale by wrist→middle-MCP distance
      │
      ▼
64-frame sequence             →  longer clips are uniformly subsampled, shorter ones zero-padded
      │
      ▼
BiLSTM / Transformer          →  126 → ... → 100 classes (softmax)
      │
      ▼
Predicted word + confidence
```

The normalization and sequence-fitting logic is implemented in both
[`ml/data/normalize.py`](ml/data/normalize.py) / [`ml/data/dataset.py`](ml/data/dataset.py)
(Python, used for training) and [`web/src/App.tsx`](web/src/App.tsx) (TypeScript, used
live in the browser). Keeping these in sync is what makes the exported model produce
sane predictions on live camera input instead of just on its own training data.

## Repository Structure

```
signbridge/
├── ml/
│   ├── data/
│   │   ├── prepare_split.py      # WLASL_v0.3.json + nslt_*.json → wlasl100_split.json
│   │   ├── extract_landmarks.py  # video → landmarks/*.npy (MediaPipe Hand Landmarker)
│   │   ├── normalize.py          # per-hand normalization (wrist origin + scale)
│   │   ├── augment.py            # mirroring, temporal jitter, Gaussian noise
│   │   └── dataset.py            # PyTorch Dataset + pad_or_sample()
│   ├── models/
│   │   ├── bilstm.py             # bidirectional LSTM
│   │   └── transformer.py        # Transformer-encoder
│   ├── train.py                  # training
│   ├── evaluate.py               # Top-1/Top-5 + most-confused pairs on the test split
│   ├── predict.py                # CLI inference on a single video file
│   ├── export_onnx.py            # .pth → .onnx export, verified against PyTorch output
│   ├── diagnose.py, check_dataset.py  # dataset sanity checks
│   ├── hand_landmarker.task      # MediaPipe hand-landmark model bundle
│   ├── transformer.onnx          # exported model
│   └── requirements.txt
└── web/
    ├── src/
    │   ├── App.tsx                # camera capture, MediaPipe, ONNX inference, UI
    │   └── glosses.ts             # the 100 words, in the exact order the model outputs
    └── public/models/
        └── transformer.onnx       # model copy served to the browser
```

## Results

Evaluated on the WLASL100 test split (100 classes):

| Model | Top-1 | Top-5 |
|---|---|---|
| BiLSTM | 27.0% | 56.0% |
| **Transformer** | **36.0%** | **69.0%** |

For reference, random guessing across 100 classes would score ~1% Top-1 / ~5% Top-5.
The web demo uses the Transformer, since it's the strongest of the two.

## Key Limitation: Data Availability

WLASL100 nominally has 2,038 clips (1,442 train / 338 val / 258 test) across 100
classes. Its source videos were scraped from YouTube and various ASL dictionary sites
years ago, and a large fraction have since gone offline. Only **1,013 of 2,038 (49.7%)**
are available in this project's local copy of the dataset:

| Split | Clips available | Classes represented | Examples per class |
|---|---|---|---|
| train | 748 | 100 / 100 | 4–12 |
| val | 165 | 89 / 100 | 1–4 |
| test | 100 | 72 / 100 | 1–3 |

This is the actual ceiling on accuracy — no architecture or hyperparameter change gets
around a 100-way classifier being trained on 4–12 examples per class. The highest-value
next step for anyone continuing this project is re-fetching the missing source videos
(`WLASL_v0.3.json` lists a direct URL per `video_id`) and re-running
`extract_landmarks.py`, not further model tuning.

## Getting Started

### Quick Start: Web Demo

The trained model is already exported and committed, so you can try the live demo
without touching Python or the dataset at all:

```bash
# Download Node.js if you haven’t installed it yet
cd web
npm install
npm run dev
```

Open the printed local URL (typically `http://localhost:5173`) in a real desktop
browser — not a sandboxed agent preview pane, which cannot grant camera access — and
allow camera access when prompted. You'll see your hand tracked with a skeleton
overlay, and below it, the currently predicted word with a confidence percentage.

> **Note on live accuracy:** the model is trained on isolated single-word clips (one
> gesture from start to end), not a continuous stream. The browser demo approximates
> this with a sliding window over the last 64 frames, so it always shows *some*
> prediction — including on random hand motion — at the accuracy shown in
> [Results](#results) above.

### Full ML Pipeline

Use this if you want to retrain, evaluate, or run inference from the command line.

**Prerequisites:** Python 3.12+, and a local copy of the
[WLASL dataset](https://github.com/dxli94/WLASL) (`WLASL_v0.3.json`, the `nslt_*.json`
splits, the class list, and the source videos). The dataset is **not** bundled with
this repository — see [License](#license) for why.

```bash
cd ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

1. **Build the split file** (skip if `data/wlasl100_split.json` already exists):

   ```bash
   python3 data/prepare_split.py \
     --nslt data/raw/wlasl-processed/nslt_100.json \
     --classes data/raw/wlasl-processed/wlasl_class_list.txt \
     --videos-dir data/raw/wlasl-processed/videos \
     --out data/wlasl100_split.json
   ```

2. **Extract hand landmarks** from video (skips files already processed):

   ```bash
   python3 data/extract_landmarks.py
   ```

3. **Sanity-check the data:**

   ```bash
   python3 check_dataset.py
   python3 diagnose.py
   ```

4. **Train:**

   ```bash
   python3 train.py --model transformer --epochs 40
   python3 train.py --model bilstm --epochs 40
   ```

   Saves the best (by validation accuracy) checkpoint to `best_<model>.pth` and logs
   to `runs/<model>/` (view with `tensorboard --logdir runs`). Checkpoints are not
   committed to git — they're regenerated by training.

5. **Evaluate:**

   ```bash
   python3 evaluate.py --model transformer
   ```

   Prints Top-1/Top-5 accuracy and the most common misclassifications (`true → predicted`).

6. **Run inference on a single video, from the CLI:**

   ```bash
   python3 predict.py path/to/video.mp4 --model transformer --top-k 5
   ```

7. **Export to ONNX** for the web demo:

   ```bash
   python3 export_onnx.py --model transformer
   ```

   Writes `transformer.onnx` and verifies its output matches the PyTorch model to
   within ~1e-6. Copy the result to `web/public/models/transformer.onnx` if you
   retrained and want the browser demo to use your new weights.

## Command Reference

| Task | Command |
|---|---|
| Build the split file | `python3 data/prepare_split.py --nslt ... --classes ... --videos-dir ... --out ...` |
| Extract landmarks | `python3 data/extract_landmarks.py` |
| Check dataset health | `python3 check_dataset.py` / `python3 diagnose.py` |
| Train | `python3 train.py --model {bilstm,transformer} [--epochs N]` |
| Evaluate | `python3 evaluate.py --model {bilstm,transformer}` |
| Predict on a video | `python3 predict.py <video> --model {bilstm,transformer} --top-k K` |
| Export to ONNX | `python3 export_onnx.py --model {bilstm,transformer}` |
| Run the web demo | `cd web && npm install && npm run dev` |
| Build the web demo | `cd web && npm run build` |

## Model Details

- **Input:** 126 features per frame (2 hands × 21 MediaPipe landmarks × `[x, y, z]`),
  sequences fit to exactly 64 frames.
- **BiLSTM:** 2-layer bidirectional LSTM, hidden size 128 (256 after concatenation),
  dropout 0.3, LayerNorm + Linear classification head.
- **Transformer:** 3-layer encoder, `d_model=128`, 4 attention heads, feedforward size
  512, dropout 0.2, sinusoidal positional encoding, mask-aware mean pooling over time.
- **Loss:** cross-entropy weighted by inverse class frequency, since the train split
  has only 4–12 examples per class.
- **Augmentation (train only):** horizontal mirroring, temporal jitter, Gaussian noise.

## Known Limitations & Roadmap

- **Data volume** is the dominant limitation — see
  [Key Limitation: Data Availability](#key-limitation-data-availability).
- **Hands only.** Landmarks don't include body pose or facial expression, both of
  which carry meaning in real ASL.
- **Isolated clips vs. continuous stream.** The model is trained on trimmed
  single-word clips; there's no learned segmentation for "where does the sign start
  and end" in a live stream. The web demo's sliding window is a practical
  approximation, not a real solution.
- No automated tests or CI pipeline yet.

## License

No license has been added to this project yet.

**Data:** the WLASL dataset is distributed under the Computational Use of Data
Agreement (C-UDA), which prohibits redistributing the source videos. Accordingly,
this repository's `.gitignore` intentionally excludes raw videos and extracted
landmark arrays — only code, the split metadata (`wlasl100_split.json`), and the
exported model (`transformer.onnx`) are version-controlled.

## Acknowledgments

- Li, Dongxu, et al. ["Word-level Deep Sign Language Recognition from Video: A New
  Large-scale Dataset and Methods Comparison."](https://github.com/dxli94/WLASL) WACV 2020. — the WLASL dataset.
- [MediaPipe Tasks Vision](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) (Google) — hand landmark detection, in both Python and the browser.
- [ONNX Runtime](https://onnxruntime.ai/) — cross-platform model inference.
- [PyTorch](https://pytorch.org/) — model training.
