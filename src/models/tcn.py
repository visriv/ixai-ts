import torch as th
from torch import nn

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksz=3, dil=1):
        super().__init__()
        pad = (ksz-1)*dil
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, ksz, padding="same", dilation=dil),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, ksz, padding="same", dilation=dil),
            nn.ReLU()
        )
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch!=out_ch else nn.Identity()
    def forward(self, x):  # x: [B, D, T]
        y = self.net(x)
        return y + self.res(x)

class TCNClassifier(nn.Module):
    def __init__(self, D, C, hidden=64, layers=2, ksz=3, dil=1):
        super().__init__()
        blocks = []
        in_ch = D
        for i in range(layers):
            blocks.append(TCNBlock(in_ch, hidden, ksz, dil*(2**i)))
            in_ch = hidden
        self.tcn = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden, C)
    def forward(self, X):  # X: [B, T, D]
        x = X.transpose(1,2)  # [B, D, T]
        h = self.tcn(x)
        g = self.pool(h).squeeze(-1)
        return self.fc(g)
