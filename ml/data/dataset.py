import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import add_gaussian_noise, mirror_hand, time_jitter
from .normalize import normalize_sequence

MAX_SEQ_LEN = 64

class WLASLDataset(Dataset):
    def __init__(self, landmarks_dir: Path, split_file: Path, split: str, augment: bool = False):
        with open(split_file) as f:
            entries = json.load(f)

        all_glosses = sorted({e["gloss"] for e in entries})
        self.label_to_idx = {g: i for i, g in enumerate(all_glosses)}
        self.entries = [e for e in entries if e["split"] == split]
        self.landmarks_dir = landmarks_dir
        self.augment = augment

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        landmarks = np.load(self.landmarks_dir / f"{entry['video_id']}.npy")
        landmarks = normalize_sequence(landmarks)

        if self.augment:
            if np.random.rand() < 0.5:
                landmarks = mirror_hand(landmarks)
            if np.random.rand() < 0.5:
                landmarks = time_jitter(landmarks)
            landmarks = add_gaussian_noise(landmarks)

        landmarks = landmarks.reshape(landmarks.shape[0], -1)

        T = landmarks.shape[0]
        if T >= MAX_SEQ_LEN:
            sample_idx = np.linspace(0, T - 1, MAX_SEQ_LEN).round().astype(int)
            landmarks = landmarks[sample_idx]
            mask = np.ones(MAX_SEQ_LEN, dtype=bool)
        else:
            pad = np.zeros((MAX_SEQ_LEN - T, landmarks.shape[1]), dtype=np.float32)
            landmarks = np.concatenate([landmarks, pad], axis=0)
            mask = np.concatenate([np.ones(T, dtype=bool), np.zeros(MAX_SEQ_LEN - T, dtype=bool)])

        label = self.label_to_idx[entry["gloss"]]
        return (
            torch.tensor(landmarks, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.bool),
            torch.tensor(label, dtype=torch.long),
        )