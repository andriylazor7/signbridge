import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from data.dataset import WLASLDataset
from models.bilstm import BiLSTMClassifier
from models.transformer import TransformerClassifier

def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    topk = logits.topk(k, dim=1).indices  
    correct = (topk == targets.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


def evaluate(model_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = WLASLDataset(Path("data/landmarks"), Path("data/wlasl100_split.json"), "test", augment=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    idx_to_label = {i: g for g, i in test_ds.label_to_idx.items()}

    model = (BiLSTMClassifier if model_name == "bilstm" else TransformerClassifier)(
        num_classes=len(test_ds.label_to_idx)
    )
    model.load_state_dict(torch.load(f"best_{model_name}.pth", map_location=device))
    model.to(device).eval()

    all_logits, all_targets = [], []
    with torch.no_grad():
        for x, mask, y in test_loader:
            logits = model(x.to(device), mask.to(device)).cpu()
            all_logits.append(logits)
            all_targets.append(y)

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)

    top1 = top_k_accuracy(all_logits, all_targets, k=1)
    top5 = top_k_accuracy(all_logits, all_targets, k=5)
    print(f"[{model_name}] Top-1: {top1:.3f}   Top-5: {top5:.3f}")

    preds = all_logits.argmax(dim=1).numpy()
    cm = confusion_matrix(all_targets.numpy(), preds)

    cm_off_diag = cm.copy()
    np.fill_diagonal(cm_off_diag, 0)
    flat_top = np.argsort(-cm_off_diag.ravel())[:10]
    top_confusions = np.dstack(np.unravel_index(flat_top, cm_off_diag.shape))[0]

    for true_idx, pred_idx in top_confusions:
        count = cm[true_idx, pred_idx]
        if count > 0:
            print(f"  {idx_to_label[true_idx]:>15} → {idx_to_label[pred_idx]:<15}  ({count}×)")

    return top1, top5, cm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bilstm", "transformer"], required=True)
    args = parser.parse_args()
    evaluate(args.model)