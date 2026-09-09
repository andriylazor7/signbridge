import argparse
import json
from pathlib import Path

import torch

from data.dataset import pad_or_sample
from data.extract_landmarks import extract_landmarks_from_video
from data.normalize import normalize_sequence
from models.bilstm import BiLSTMClassifier
from models.transformer import TransformerClassifier


def load_idx_to_label(split_file: Path) -> dict[int, str]:
    entries = json.load(open(split_file, encoding="utf-8"))
    glosses = sorted({e["gloss"] for e in entries})
    return dict(enumerate(glosses))


def predict(video_path: str, model_name: str = "transformer", top_k: int = 5):
    idx_to_label = load_idx_to_label(Path("data/wlasl100_split.json"))
    num_classes = len(idx_to_label)

    landmarks = extract_landmarks_from_video(video_path)
    if landmarks.shape[0] == 0:
        raise ValueError(
            f"No frames could be read from '{video_path}', or no hand was detected in any frame."
        )

    landmarks = normalize_sequence(landmarks)
    landmarks = landmarks.reshape(landmarks.shape[0], -1)
    landmarks, mask = pad_or_sample(landmarks)

    x = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)
    m = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)

    model = (BiLSTMClassifier if model_name == "bilstm" else TransformerClassifier)(num_classes=num_classes)
    model.load_state_dict(torch.load(f"best_{model_name}.pth", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        probs = torch.softmax(model(x, m), dim=1)[0]

    top_probs, top_idx = probs.topk(min(top_k, num_classes))
    return [(idx_to_label[i], p) for p, i in zip(top_probs.tolist(), top_idx.tolist())]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the ASL gloss for a video clip.")
    parser.add_argument("video", type=str, help="Path to a video file")
    parser.add_argument("--model", choices=["bilstm", "transformer"], default="transformer")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    for gloss, prob in predict(args.video, args.model, args.top_k):
        print(f"  {gloss:<15} {prob:.1%}")
