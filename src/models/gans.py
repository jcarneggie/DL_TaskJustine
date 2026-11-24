"""Vanilla GAN (MLP / fully-connected) dan DCGAN (konvolusional) untuk 28x28.

Objektif minimax: min_G max_D E[log D(x)] + E[log(1 - D(G(z)))].
Generator dilatih dengan non-saturating loss: max_G E[log D(G(z))].
"""
import torch
import torch.nn as nn

NZ = 100  # dimensi prior noise z ~ N(0, I)


# ------------------------------------------------------------ Vanilla GAN
class MLPGenerator(nn.Module):
    def __init__(self, nz=NZ):
        super().__init__()
        self.nz = nz
        self.net = nn.Sequential(
            nn.Linear(nz, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 784), nn.Tanh())

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)


class MLPDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1))           # logit; sigmoid ada di BCEWithLogits

    def forward(self, x):
        return self.net(x).view(-1)


# ------------------------------------------------------------------ DCGAN
class DCGenerator(nn.Module):
    """Prinsip DCGAN (Radford et al., 2015): transposed conv (tanpa pooling),
    BatchNorm di generator, ReLU di hidden layer, Tanh di output."""

    def __init__(self, nz=NZ, ngf=64):
        super().__init__()
        self.nz = nz
        self.fc = nn.Sequential(nn.Linear(nz, ngf * 2 * 7 * 7, bias=False),
                                nn.BatchNorm1d(ngf * 2 * 7 * 7), nn.ReLU(True))
        self.net = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),  # 7 -> 14
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1),                     # 14 -> 28
            nn.Tanh())
        self.ngf = ngf

    def forward(self, z):
        h = self.fc(z).view(-1, self.ngf * 2, 7, 7)
        return self.net(h)


class DCDiscriminator(nn.Module):
    """Strided conv (tanpa pooling), LeakyReLU(0.2), BatchNorm kecuali layer pertama."""

    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True),                         # 28 -> 14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),                                                      # 14 -> 7
            nn.Flatten(), nn.Linear(ndf * 2 * 7 * 7, 1))

    def forward(self, x):
        return self.net(x).view(-1)


def dcgan_init(m):
    """Inisialisasi bobot N(0, 0.02) sesuai paper DCGAN."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
