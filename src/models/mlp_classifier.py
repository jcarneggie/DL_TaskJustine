"""MLP Classifier 784 -> 256 -> 64 -> 10 (strukturnya mirip encoder AE)."""
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)   # W1: 256 x 784
        self.fc2 = nn.Linear(256, 64)    # W2: 64 x 256
        self.fc3 = nn.Linear(64, 10)     # W3: 10 x 64
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.2)

    def features(self, x):
        """z_MLP in R^64: representasi tepat sebelum lapisan klasifikasi akhir."""
        h1 = self.act(self.fc1(x.flatten(1)))
        return self.act(self.fc2(self.drop(h1)))

    def forward(self, x):
        return self.fc3(self.drop(self.features(x)))


class LatentDecoder(nn.Module):
    """Decoder terpisah 64 -> 256 -> 784 (arsitektur sama dengan decoder Vanilla AE)."""

    def __init__(self, latent=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent, 256), nn.ReLU(),
                                 nn.Linear(256, 784), nn.Sigmoid())

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)
