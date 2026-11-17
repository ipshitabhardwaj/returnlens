# PDF ki Listing C-3 (nn_quantile.py) poora yahan paste karo
import numpy as np, torch, torch.nn as nn
QS = torch.tensor([0.1, 0.5, 0.9])
class QNet(nn.Module):
    def __init__(self, n_cal, hid=64, horizon=7, n_q=3):
        super().__init__()
        self.assoc = nn.Sequential(
            nn.Linear(n_cal, hid), nn.ReLU()
        )  # festival / calendar / price features
        self.gru = nn.GRU(1 + hid, hid, batch_first=True)  # temporal layer over last 28 days
        self.head = nn.Linear(hid, horizon * n_q)
        self.h, self.q = horizon, n_q
    def forward(self, y_hist, cal_hist):  # (B,28,1), (B,28,n_cal)
        z = torch.cat([y_hist, self.assoc(cal_hist)], -1)
        out, _ = self.gru(z)
        return (
            self.head(out[:, -1]).view(-1, self.h, self.q).sort(-1).values
        )  # sorted => no quantile crossing
def pinball(pred, y):  # pred (B,H,3), y (B,H)
    d = y[..., None] - pred
    q = QS.to(pred.device)
    return torch.maximum(q * d, (q - 1) * d).mean()
def windows(y, cal, L=28, H=7):
    """y: (T,) scaled series, cal: (T,n_cal). Returns tensors for training."""
    X, C, Y = [], [], []
    for t in range(L, len(y) - H):
        X.append(y[t - L : t, None])
        C.append(cal[t - L : t])
        Y.append(y[t : t + H])
    return map(lambda a: torch.tensor(np.array(a), dtype=torch.float32), (X, C, Y))
def fit(y, cal, epochs=40, lr=2e-3, device="cpu"):
    mu, sd = y.mean(), y.std()
    X, C, Y = windows((y - mu) / sd, cal)
    m = QNet(cal.shape[1]).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 128):
            b = perm[i : i + 128]
            loss = pinball(m(X[b].to(device), C[b].to(device)), Y[b].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m, mu, sd
