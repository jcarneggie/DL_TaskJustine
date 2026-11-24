"""Vanilla AE, Sparse AE, dan VAE berbasis MLP (784 -> 256 -> 64).

Encoder sengaja dibuat identik secara struktural dengan MLP Classifier
(784 -> 256 -> 64) agar eksperimen Bagian 2 sebanding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT = 64
HIDDEN = 256


class VanillaAE(nn.Module):
    """Kode laten padat (linear, tanpa aktivasi). Loss: MSE."""

    def __init__(self, latent=LATENT, hidden=HIDDEN):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(784, hidden), nn.ReLU(),
                                     nn.Linear(hidden, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(),
                                     nn.Linear(hidden, 784), nn.Sigmoid())

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    def loss(self, x):
        x_hat, z = self(x)
        rec = F.mse_loss(x_hat, x)
        return rec, {"rec": rec.item()}


class SparseAE(VanillaAE):
    """Kode laten ReLU (non-negatif) + penalti L1: L = MSE + lambda * ||h||_1."""

    def __init__(self, latent=LATENT, hidden=HIDDEN, lam=1e-4):
        super().__init__(latent, hidden)
        self.encoder.append(nn.ReLU())   # h >= 0, memungkinkan nol eksak
        self.lam = lam

    def loss(self, x):
        x_hat, h = self(x)
        rec = F.mse_loss(x_hat, x)
        l1 = h.abs().sum(1).mean()
        return rec + self.lam * l1, {"rec": rec.item(), "l1": l1.item()}


class VAE(nn.Module):
    """q(z|x)=N(mu, diag(sigma^2)), p(z)=N(0,I), p(x|z)=Bernoulli(decoder(z)).

    Loss (negatif ELBO per sampel) = BCE_sum + KL(q(z|x) || N(0,I)).
    """

    def __init__(self, latent=LATENT, hidden=HIDDEN):
        super().__init__()
        self.latent = latent
        self.body = nn.Sequential(nn.Flatten(), nn.Linear(784, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(),
                                     nn.Linear(hidden, 784), nn.Sigmoid())

    def encode(self, x):
        h = self.body(x)
        return self.mu(h), self.logvar(h)

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)   # reparameterization trick
        return self.decode(z), mu, logvar

    def loss(self, x):
        x_hat, mu, logvar = self(x)
        b = x.size(0)
        bce = F.binary_cross_entropy(x_hat, x, reduction="sum") / b
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / b
        return bce + kl, {"bce": bce.item(), "kl": kl.item(),
                          "rec": F.mse_loss(x_hat, x).item()}

    @torch.no_grad()
    def reconstruct(self, x):          # rekonstruksi deterministik via mean posterior
        mu, _ = self.encode(x)
        return self.decode(mu)

    @torch.no_grad()
    def sample(self, n, device="cpu"):
        return self.decode(torch.randn(n, self.latent, device=device))
