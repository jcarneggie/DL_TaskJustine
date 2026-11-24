"""Utilitas bersama: seed, device, path, data loader, dan metrik.

Semua model memakai split dan seed yang SAMA (80/10/10, seed 42) supaya
perbedaan performa murni berasal dari inductive bias model.
"""
import os
import json
import time
import random

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

SEED = 42
BATCH_SIZE = 128
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RES_DIR = os.path.join(ROOT, "results")
CKPT_DIR = os.path.join(RES_DIR, "ckpt")
HIST_DIR = os.path.join(RES_DIR, "history")
FIG_DIR = os.path.join(RES_DIR, "figures")
MET_DIR = os.path.join(RES_DIR, "metrics")
for d in (CKPT_DIR, HIST_DIR, FIG_DIR, MET_DIR):
    os.makedirs(d, exist_ok=True)

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------- data
def _load_all():
    """Gabungkan train (60k) + test (10k) Fashion-MNIST = 70k citra.

    Jalur utama: torchvision.datasets.FashionMNIST (download otomatis).
    Fallback: data/fmnist_all.npz (dipakai jika mesin tidak bisa mengakses
    server unduhan torchvision).
    """
    npz = os.path.join(DATA_DIR, "fmnist_all.npz")
    try:
        from torchvision.datasets import FashionMNIST
        tr = FashionMNIST(DATA_DIR, train=True, download=True)
        te = FashionMNIST(DATA_DIR, train=False, download=True)
        x = torch.cat([tr.data, te.data]).numpy()
        y = torch.cat([tr.targets, te.targets]).numpy()
    except Exception as e:  # noqa: BLE001
        if not os.path.exists(npz):
            raise RuntimeError(f"Dataset tidak bisa diunduh: {e}")
        print("[data] torchvision download gagal, pakai fallback", npz)
        d = np.load(npz)
        x, y = d["x"], d["y"]
    return x.astype(np.float32) / 255.0, y.astype(np.int64)


_SPLIT_CACHE = None


def get_splits():
    """Return dict split -> (x[N,1,28,28] in [0,1], y[N]). Split 80/10/10 seed 42."""
    global _SPLIT_CACHE
    if _SPLIT_CACHE is None:
        x, y = _load_all()
        rng = np.random.RandomState(SEED)
        idx = rng.permutation(len(x))
        n_tr, n_va = int(0.8 * len(x)), int(0.1 * len(x))
        parts = {"train": idx[:n_tr], "val": idx[n_tr:n_tr + n_va], "test": idx[n_tr + n_va:]}
        _SPLIT_CACHE = {k: (torch.from_numpy(x[v]).unsqueeze(1), torch.from_numpy(y[v]))
                        for k, v in parts.items()}
    return _SPLIT_CACHE


def get_loader(split: str, batch_size: int = BATCH_SIZE):
    x, y = get_splits()[split]
    g = torch.Generator().manual_seed(SEED)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size,
                      shuffle=(split == "train"), generator=g, drop_last=False)


def to_pm1(x):   # [0,1] -> [-1,1] (GAN, DCGAN, DDPM memakai output tanh)
    return x * 2 - 1


def to_01(x):    # [-1,1] -> [0,1]
    return ((x + 1) / 2).clamp(0, 1)


# -------------------------------------------------------------- metrics
def mse_psnr_ssim(x, x_hat):
    """x, x_hat: tensor [N,1,28,28] in [0,1]. Rata-rata per citra."""
    from skimage.metrics import structural_similarity
    x = x.detach().cpu().numpy()[:, 0]
    x_hat = x_hat.detach().cpu().numpy()[:, 0].clip(0, 1)
    mse_i = ((x - x_hat) ** 2).reshape(len(x), -1).mean(1)
    psnr_i = 10 * np.log10(1.0 / np.maximum(mse_i, 1e-10))
    ssim_i = np.array([structural_similarity(a, b, data_range=1.0) for a, b in zip(x, x_hat)])
    return float(mse_i.mean()), float(psnr_i.mean()), float(ssim_i.mean())


@torch.inference_mode()
def latency_ms(fn, repeats: int = 20, warmup: int = 3):
    """Median latency (ms) dari fn() yang menghasilkan 1 batch (64 sampel)."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)
