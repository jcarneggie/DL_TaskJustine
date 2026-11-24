"""Training pipeline untuk keenam model generatif.

Contoh:
    python src/train.py --model ae --epochs 30
    python src/train.py --model ddpm --epochs 12
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (set_seed, get_loader, to_pm1, DEVICE, CKPT_DIR, HIST_DIR, save_json)  # noqa: E402
from models import (VanillaAE, SparseAE, VAE, MLPGenerator, MLPDiscriminator,  # noqa: E402
                    DCGenerator, DCDiscriminator, dcgan_init, DDPM)

DEFAULT_EPOCHS = {"ae": 30, "sae": 30, "vae": 30, "gan": 40, "dcgan": 15, "ddpm": 6}


# ------------------------------------------------------- AE / SAE / VAE
def train_autoencoder(name, epochs):
    model = {"ae": VanillaAE, "sae": SparseAE, "vae": VAE}[name]().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tr, va = get_loader("train"), get_loader("val")
    hist = {"train_loss": [], "val_loss": [], "val_rec_mse": [], "epoch_time": []}
    for ep in range(epochs):
        model.train()
        t0, losses = time.time(), []
        for x, _ in tr:
            x = x.to(DEVICE)
            loss, _ = model.loss(x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        hist["epoch_time"].append(time.time() - t0)
        model.eval()
        vl, vm = [], []
        with torch.no_grad():
            for x, _ in va:
                l, info = model.loss(x.to(DEVICE))
                vl.append(l.item())
                vm.append(info["rec"])
        hist["train_loss"].append(float(np.mean(losses)))
        hist["val_loss"].append(float(np.mean(vl)))
        hist["val_rec_mse"].append(float(np.mean(vm)))
        print(f"[{name}] ep {ep+1:02d} train {hist['train_loss'][-1]:.4f} "
              f"val {hist['val_loss'][-1]:.4f} mse {hist['val_rec_mse'][-1]:.5f} "
              f"({hist['epoch_time'][-1]:.1f}s)", flush=True)
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"{name}.pt"))
    save_json(hist, os.path.join(HIST_DIR, f"{name}.json"))


# ---------------------------------------------------------- GAN / DCGAN
def grad_norm(module):
    g = [p.grad.norm() ** 2 for p in module.parameters() if p.grad is not None]
    return float(torch.sqrt(torch.stack(g).sum())) if g else 0.0


def train_gan(name, epochs):
    if name == "gan":
        G, D = MLPGenerator().to(DEVICE), MLPDiscriminator().to(DEVICE)
    else:
        G, D = DCGenerator().to(DEVICE), DCDiscriminator().to(DEVICE)
        G.apply(dcgan_init)
        D.apply(dcgan_init)
    optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    tr = get_loader("train")
    fixed_z = torch.randn(64, G.nz, device=DEVICE)
    hist = {k: [] for k in ["d_loss", "g_loss", "d_real", "d_fake", "g_grad_norm",
                            "epoch_time", "iter_d_loss", "iter_g_loss"]}
    snaps = []
    for ep in range(epochs):
        G.train()
        D.train()
        t0 = time.time()
        acc = {k: [] for k in ["d", "g", "dr", "df", "gn"]}
        for it, (x, _) in enumerate(tr):
            x = to_pm1(x.to(DEVICE))
            b = x.size(0)
            ones, zeros = torch.ones(b, device=DEVICE), torch.zeros(b, device=DEVICE)
            # --- update D: max log D(x) + log(1 - D(G(z)))
            z = torch.randn(b, G.nz, device=DEVICE)
            fake = G(z)
            dr, df = D(x), D(fake.detach())
            d_loss = bce(dr, ones) + bce(df, zeros)
            optD.zero_grad()
            d_loss.backward()
            optD.step()
            # --- update G: non-saturating, max log D(G(z))
            g_loss = bce(D(fake), ones)
            optG.zero_grad()
            g_loss.backward()
            acc["gn"].append(grad_norm(G))
            optG.step()
            acc["d"].append(d_loss.item())
            acc["g"].append(g_loss.item())
            acc["dr"].append(torch.sigmoid(dr).mean().item())
            acc["df"].append(torch.sigmoid(df).mean().item())
            if it % 50 == 0:
                hist["iter_d_loss"].append(d_loss.item())
                hist["iter_g_loss"].append(g_loss.item())
        hist["epoch_time"].append(time.time() - t0)
        for k_h, k_a in [("d_loss", "d"), ("g_loss", "g"), ("d_real", "dr"),
                         ("d_fake", "df"), ("g_grad_norm", "gn")]:
            hist[k_h].append(float(np.mean(acc[k_a])))
        G.eval()
        with torch.no_grad():
            snaps.append(G(fixed_z[:16]).cpu())
        print(f"[{name}] ep {ep+1:02d} D {hist['d_loss'][-1]:.3f} G {hist['g_loss'][-1]:.3f} "
              f"D(x) {hist['d_real'][-1]:.2f} D(G(z)) {hist['d_fake'][-1]:.2f} "
              f"|gradG| {hist['g_grad_norm'][-1]:.3f} ({hist['epoch_time'][-1]:.1f}s)", flush=True)
    torch.save(G.state_dict(), os.path.join(CKPT_DIR, f"{name}_G.pt"))
    torch.save(D.state_dict(), os.path.join(CKPT_DIR, f"{name}_D.pt"))
    torch.save(torch.stack(snaps), os.path.join(CKPT_DIR, f"{name}_snapshots.pt"))
    save_json(hist, os.path.join(HIST_DIR, f"{name}.json"))


# ----------------------------------------------------------------- DDPM
def train_ddpm(epochs, T=500):
    model = DDPM(T=T).to(DEVICE)
    ema = DDPM(T=T).to(DEVICE)
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    tr, va = get_loader("train"), get_loader("val")
    hist = {"train_loss": [], "val_loss": [], "epoch_time": [], "iter_loss": []}
    for ep in range(epochs):
        model.train()
        t0, losses = time.time(), []
        for it, (x, _) in enumerate(tr):
            loss, _ = model.loss(to_pm1(x.to(DEVICE)))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():   # EMA bobot (decay 0.999)
                for pe, pm in zip(ema.net.parameters(), model.net.parameters()):
                    pe.mul_(0.999).add_(pm, alpha=0.001)
            losses.append(loss.item())
            if it % 50 == 0:
                hist["iter_loss"].append(loss.item())
        hist["epoch_time"].append(time.time() - t0)
        ema.eval()
        g = torch.Generator().manual_seed(0)
        vl = []
        with torch.no_grad():
            for i, (x, _) in enumerate(va):
                if i >= 10:
                    break
                torch.manual_seed(i)
                vl.append(ema.loss(to_pm1(x.to(DEVICE)))[0].item())
        hist["train_loss"].append(float(np.mean(losses)))
        hist["val_loss"].append(float(np.mean(vl)))
        print(f"[ddpm] ep {ep+1:02d} train {hist['train_loss'][-1]:.4f} val(EMA) "
              f"{hist['val_loss'][-1]:.4f} ({hist['epoch_time'][-1]:.1f}s)", flush=True)
        torch.save(ema.state_dict(), os.path.join(CKPT_DIR, "ddpm.pt"))
        save_json(hist, os.path.join(HIST_DIR, "ddpm.json"))
    del g


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(DEFAULT_EPOCHS))
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    set_seed()
    ep = a.epochs or DEFAULT_EPOCHS[a.model]
    if a.model in ("ae", "sae", "vae"):
        train_autoencoder(a.model, ep)
    elif a.model in ("gan", "dcgan"):
        train_gan(a.model, ep)
    else:
        train_ddpm(ep)
