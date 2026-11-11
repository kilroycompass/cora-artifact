"""Self-contained scorers and source-only fitting utilities."""
import random
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import StratifiedShuffleSplit

def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

class Scaler:
    def fit(self, x):
        self.mean = x.mean((0, 2), keepdims=True)
        self.sd = np.maximum(x.std((0, 2), keepdims=True), 1e-4)
        return self

    def transform(self, x):
        return ((x - self.mean) / self.sd).astype(np.float32)


def summary(x):
    return np.concatenate([x.mean(2), x.std(2), x.min(2), x.max(2), x[:, :, -1]], axis=1)


def predict(model, x, bs=1024):
    if isinstance(model, nn.Module):
        model.eval()
        with torch.no_grad():
            return np.concatenate([model(torch.from_numpy(x[i:i + bs])).sigmoid().numpy()
                                   for i in range(0, len(x), bs)]) if len(x) else np.zeros(0)
    return model.predict_proba(summary(x))[:, 1]


def partition_validation(yval, n_sel=2048, seed=9912):
    n_sel = min(n_sel, int(.3 * len(yval)))
    sp = StratifiedShuffleSplit(n_splits=1, train_size=n_sel, random_state=seed)
    a, b = next(sp.split(np.zeros(len(yval)), yval))
    return np.sort(a), np.sort(b)


def sample_train(y, cap, seed):
    if cap is None or cap >= len(y):
        return np.arange(len(y))
    sp = StratifiedShuffleSplit(n_splits=1, train_size=cap, random_state=seed)
    return np.sort(next(sp.split(np.zeros(len(y)), y))[0])


class CNN1D(nn.Module):
    def __init__(self, L=64, C=12, w=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(C, w, 5, padding=2), nn.GELU(),
                                 nn.Conv1d(w, w, 5, padding=4, dilation=2), nn.GELU(),
                                 nn.Conv1d(w, w, 5, padding=8, dilation=4), nn.GELU())
        self.out = nn.Linear(2 * w, 1)

    def forward(self, x):
        h = self.net(x)
        return self.out(torch.cat([h.mean(-1), h.amax(-1)], 1)).reshape(-1)


def build_cnn(channels=12):return CNN1D(L=64,C=channels)
