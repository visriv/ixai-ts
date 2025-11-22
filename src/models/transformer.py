import torch as th
from torch import nn

class PosEnc(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = th.zeros(max_len, d_model)
        pos = th.arange(0, max_len).unsqueeze(1)
        div = th.exp(th.arange(0, d_model, 2) * -(th.log(th.tensor(10000.0))/d_model))
        pe[:, 0::2] = th.sin(pos*div)
        pe[:, 1::2] = th.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, L, D]
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerClassifier(nn.Module):
    def __init__(self, D, C, d_model=64, nhead=4, layers=2, dim_ff=128, all_times=False):
        super().__init__()
        self.inp = nn.Linear(D, d_model)
        self.pe = PosEnc(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, layers)
        self.fc = nn.Linear(d_model, C)
        self.all_times = all_times

    def forward(self, X):  # X: [B, T, D]
        z = self.inp(X)          # [B, T, d_model]
        z = self.pe(z)           # [B, T, d_model]
        h = self.enc(z)          # [B, T, d_model]
        if not self.all_times:
            h = h.mean(dim=1)     # [B, d_model]  <-- use aggregate over timestep
        
        return self.fc(h) # [B, T or 1, C]  <-- per-timestep logits

