import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import WLASLDataset
from models.bilstm import BiLSTMClassifier
from models.transformer import TransformerClassifier


def train_model(model_name: str, num_epochs: int = 40, batch_size: int = 32, lr: float = 3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_ds = WLASLDataset(Path("data/landmarks"), Path("data/wlasl100_split.json"), "train", augment=True)
    val_ds = WLASLDataset(Path("data/landmarks"), Path("data/wlasl100_split.json"), "val", augment=False)
    num_classes = len(train_ds.label_to_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = (BiLSTMClassifier if model_name == "bilstm" else TransformerClassifier)(num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    writer = SummaryWriter(log_dir=f"runs/{model_name}")
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for x, mask, y in tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch+1}/{num_epochs}"):
            x, mask, y = x.to(device), mask.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x, mask)
            loss = criterion(logits, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(dim=1) == y).sum().item()
            train_total += x.size(0)

        scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, mask, y in val_loader:
                x, mask, y = x.to(device), mask.to(device), y.to(device)
                logits = model(x, mask)
                val_correct += (logits.argmax(dim=1) == y).sum().item()
                val_total += x.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        writer.add_scalar("Loss/train", train_loss / train_total, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
        print(f"Epoch {epoch+1}: train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_{model_name}.pth")

    writer.close()
    print(f"Best val accuracy for {model_name}: {best_val_acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bilstm", "transformer"], required=True)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    train_model(args.model, num_epochs=args.epochs)