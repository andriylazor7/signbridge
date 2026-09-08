import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim=126, hidden_dim=128, num_layers=2, num_classes=100, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),  
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, x, mask):
        lengths = mask.sum(dim=1).cpu()

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)

        forward_last = h_n[-2]
        backward_last = h_n[-1]
        pooled = torch.cat([forward_last, backward_last], dim=1) 

        return self.classifier(pooled)