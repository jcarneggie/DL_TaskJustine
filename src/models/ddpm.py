"""DDPM (Ho et al., 2020) dengan U-Net kecil untuk citra 28x28.

Forward process : q(x_t | x_0) = N(sqrt(abar_t) x_0, (1 - abar_t) I)
Objektif        : L_simple = E_{t, x0, eps} || eps - eps_theta(x_t, t) ||^2
Reverse process : x_{t-1} = 1/sqrt(a_t) (x_t - b_t/sqrt(1-abar_t) eps_theta) + sigma_t z
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(temb)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, ch=32, tdim=128):
        super().__init__()
        self.tdim = tdim
        self.temb = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inp = nn.Conv2d(1, ch, 3, padding=1)
        self.d1 = ResBlock(ch, ch, tdim)                          # 28x28
        self.down1 = nn.Conv2d(ch, ch, 3, stride=2, padding=1)    # -> 14
        self.d2 = ResBlock(ch, ch * 2, tdim)
        self.down2 = nn.Conv2d(ch * 2, ch * 2, 3, stride=2, padding=1)  # -> 7
        self.d3 = ResBlock(ch * 2, ch * 4, tdim)
        self.mid = ResBlock(ch * 4, ch * 4, tdim)
        self.u3 = ResBlock(ch * 8, ch * 2, tdim)
        self.up2 = nn.Conv2d(ch * 2, ch * 2, 3, padding=1)
        self.u2 = ResBlock(ch * 4, ch, tdim)
        self.up1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.u1 = ResBlock(ch * 2, ch, tdim)
        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, 1, 3, padding=1))

    def forward(self, x, t):
        te = self.temb(timestep_embedding(t, self.tdim))
        h1 = self.d1(self.inp(x), te)
        h2 = self.d2(self.down1(h1), te)
        h3 = self.d3(self.down2(h2), te)
        m = self.mid(h3, te)
        u = self.u3(torch.cat([m, h3], 1), te)
        u = self.up2(F.interpolate(u, scale_factor=2, mode="nearest"))
        u = self.u2(torch.cat([u, h2], 1), te)
        u = self.up1(F.interpolate(u, scale_factor=2, mode="nearest"))
        u = self.u1(torch.cat([u, h1], 1), te)
        return self.out(u)


class DDPM(nn.Module):
    def __init__(self, T=500, beta1=1e-4, betaT=0.04, ch=32):
        super().__init__()
        self.T = T
        self.net = UNet(ch)
        betas = torch.linspace(beta1, betaT, T)
        alphas = 1 - betas
        abar = torch.cumprod(alphas, 0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("abar", abar)

    def loss(self, x0):
        """x0 in [-1,1]. Supervised noise prediction (regresi MSE biasa)."""
        b = x0.size(0)
        t = torch.randint(0, self.T, (b,), device=x0.device)
        eps = torch.randn_like(x0)
        ab = self.abar[t].view(-1, 1, 1, 1)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        l = F.mse_loss(self.net(xt, t), eps)
        return l, {"eps_mse": l.item()}

    @torch.no_grad()
    def sample(self, n, device="cpu", return_traj_at=None):
        x = torch.randn(n, 1, 28, 28, device=device)
        traj = {}
        for i in reversed(range(self.T)):
            if return_traj_at is not None and (i + 1) in return_traj_at:
                traj[i + 1] = x.clone()
            t = torch.full((n,), i, device=device, dtype=torch.long)
            eps = self.net(x, t)
            a, ab, b = self.alphas[i], self.abar[i], self.betas[i]
            mean = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
            x = mean + (b.sqrt() * torch.randn_like(x) if i > 0 else 0)
        x = x.clamp(-1, 1)
        if return_traj_at is not None:
            traj[0] = x.clone()
            return x, traj
        return x
