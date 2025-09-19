import torch as th
from torch import nn

class LSTMClassifier(nn.Module):
    def __init__(self, D, C, hidden=64, layers=1, bidir=False):
        super().__init__()
        self.lstm = nn.LSTM(D, hidden, num_layers=layers, batch_first=True, bidirectional=bidir)
        out = hidden*(2 if bidir else 1)
        self.fc = nn.Linear(out, C)
    def forward(self, X):  # [B, T, D]
        h,_ = self.lstm(X)
        g = h.mean(dim=1)
        return self.fc(g)
