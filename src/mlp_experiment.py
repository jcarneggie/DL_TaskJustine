"""Bagian 2: Pembuktian empiris bahwa bobot & fitur laten MLP Classifier
tidak bisa langsung dipakai untuk rekonstruksi / generasi citra.

Tahap 1  Latih MLP 784-256-64-10 (Cross-Entropy) sampai akurasi test >= 88%.
Tahap 2  Proyeksi balik one-hot kelas lewat transpos bobot: W3^T -> W2^T -> W1^T.
Tahap 3  Proyeksi balik fitur laten z_MLP citra uji lewat W2^T, W1^T.
Tahap 4  Frozen latent decoding: decoder 64-256-784 (MSE) di atas z_MLP beku,
         dibandingkan dengan Vanilla AE (z_AE in R^64).
Tahap 5  Sampling z_rand ~ N(mu_z, diag(sigma_z^2)) lalu decode dengan decoder Tahap 4.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (set_seed, get_loader, get_splits, DEVICE, CKPT_DIR, MET_DIR, HIST_DIR,  # noqa: E402
                    mse_psnr_ssim, save_json)
from models import MLPClassifier, LatentDecoder, VanillaAE  # noqa: E402


def accuracy(model, split):
    x, y = get_splits()[split]
    model.eval()
    with torch.no_grad():
        return (model(x.to(DEVICE)).argmax(1).cpu() == y).float().mean().item()


def stage1_train_classifier(epochs=25):
    clf = MLPClassifier().to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    hist = {"train_loss": [], "val_acc": []}
    for ep in range(epochs):
        clf.train()
        ls = []
        for x, y in get_loader("train"):
            loss = F.cross_entropy(clf(x.to(DEVICE)), y.to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            ls.append(loss.item())
        hist["train_loss"].append(float(np.mean(ls)))
        hist["val_acc"].append(accuracy(clf, "val"))
        print(f"[mlp] ep {ep+1:02d} CE {hist['train_loss'][-1]:.4f} val_acc {hist['val_acc'][-1]:.4f}")
    test_acc = accuracy(clf, "test")
    print(f"[mlp] TEST ACC = {test_acc:.4f}")
    assert test_acc >= 0.88, "Akurasi belum mencapai 88%, tambah epoch"
    torch.save(clf.state_dict(), os.path.join(CKPT_DIR, "mlp_classifier.pt"))
    W = {"W1": clf.fc1.weight.detach().cpu(), "W2": clf.fc2.weight.detach().cpu(),
         "W3": clf.fc3.weight.detach().cpu()}
    torch.save(W, os.path.join(CKPT_DIR, "mlp_weights_W1W2W3.pt"))
    save_json(hist, os.path.join(HIST_DIR, "mlp_classifier.json"))
    return clf, W, test_acc


def minmax(img):
    """Normalisasi min-max per citra HANYA untuk visualisasi pola bobot."""
    flat = img.flatten(1)
    lo, hi = flat.min(1, keepdim=True)[0], flat.max(1, keepdim=True)[0]
    return ((flat - lo) / (hi - lo + 1e-8)).view_as(img)


def weight_algebra(W):
    """Bukti numerik W^T W != I (non-ortogonal, tidak invertibel)."""
    out = {}
    for k, w in W.items():
        wtw = w.T @ w                       # (in x in)
        eye = torch.eye(wtw.shape[0])
        s = torch.linalg.svdvals(w)
        out[k] = {"shape": list(w.shape),
                  "rank": int(torch.linalg.matrix_rank(w)),
                  "in_dim": w.shape[1],
                  "rel_frobenius_WtW_minus_I": float((wtw - eye).norm() / eye.norm()),
                  "sigma_max": float(s.max()), "sigma_min": float(s.min()),
                  "condition_number": float(s.max() / s.min())}
    return out


def main():
    set_seed()
    xt, yt = get_splits()["test"]
    res = {}

    # ---------------- Tahap 1
    t0 = time.time()
    clf, W, acc = stage1_train_classifier()
    res["stage1"] = {"test_accuracy": acc, "train_time_s": time.time() - t0}
    res["weight_algebra"] = weight_algebra(W)
    clf.eval()
    for p in clf.parameters():
        p.requires_grad_(False)     # beku untuk Tahap 2-5
    W1, W2, W3 = W["W1"], W["W2"], W["W3"]

    # ---------------- Tahap 2: one-hot kelas -> piksel
    Y = torch.eye(10)                                  # y_c, 10 x 10
    z_hat = F.relu(Y @ W3)                             # = ReLU(W3^T y_c), 10 x 64
    h1_hat = F.relu(z_hat @ W2)                        # = ReLU(W2^T z_hat), 10 x 256
    pre = h1_hat @ W1                                  # W1^T h1_hat, 10 x 784
    x_class = torch.sigmoid(pre).view(-1, 1, 28, 28)
    x_class_vis = minmax(pre.view(-1, 1, 28, 28))
    class_means = torch.stack([xt[yt == c].mean(0) for c in range(10)])
    corr = [float(np.corrcoef(x_class_vis[c].flatten(), class_means[c].flatten())[0, 1]) for c in range(10)]
    res["stage2"] = {"sigmoid_output_range": [float(x_class.min()), float(x_class.max())],
                     "sigmoid_output_std": float(x_class.std()),
                     "zeroed_frac_z_hat": float((z_hat == 0).float().mean()),
                     "zeroed_frac_h1_hat": float((h1_hat == 0).float().mean()),
                     "pearson_corr_with_class_mean_image": corr,
                     "mse_vs_class_mean": float(((x_class - class_means) ** 2).mean())}

    # ---------------- Tahap 3: z_MLP citra uji -> piksel
    with torch.no_grad():
        z_mlp = clf.features(xt.to(DEVICE)).cpu()     # N x 64 (setelah ReLU)
        h1_true = F.relu(xt.flatten(1) @ W1.T + clf.fc1.bias.cpu())
    pre3 = F.relu(z_mlp @ W2) @ W1
    x_lat = torch.sigmoid(pre3).view(-1, 1, 28, 28)
    x_lat_vis = minmax(pre3.view(-1, 1, 28, 28))
    m3 = mse_psnr_ssim(xt, x_lat)
    m3v = mse_psnr_ssim(xt, x_lat_vis)
    res["stage3"] = {"mse": m3[0], "psnr": m3[1], "ssim": m3[2],
                     "minmax_vis_mse": m3v[0], "minmax_vis_psnr": m3v[1], "minmax_vis_ssim": m3v[2],
                     "zeroed_frac_z_mlp": float((z_mlp == 0).float().mean()),
                     "zeroed_frac_h1_forward": float((h1_true == 0).float().mean())}

    # ---------------- Tahap 4: frozen latent decoder vs Vanilla AE
    dec = LatentDecoder().to(DEVICE)
    opt = torch.optim.Adam(dec.parameters(), lr=1e-3)
    hist = {"train_loss": [], "val_loss": []}
    t0 = time.time()
    for ep in range(30):
        dec.train()
        ls = []
        for x, _ in get_loader("train"):
            x = x.to(DEVICE)
            with torch.no_grad():
                z = clf.features(x)                      # MLP beku
            loss = F.mse_loss(dec(z), x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ls.append(loss.item())
        dec.eval()
        with torch.no_grad():
            xv = get_splits()["val"][0].to(DEVICE)
            vl = F.mse_loss(dec(clf.features(xv)), xv).item()
        hist["train_loss"].append(float(np.mean(ls)))
        hist["val_loss"].append(vl)
        print(f"[frozen-dec] ep {ep+1:02d} train {hist['train_loss'][-1]:.5f} val {vl:.5f}")
    torch.save(dec.state_dict(), os.path.join(CKPT_DIR, "mlp_frozen_decoder.pt"))
    save_json(hist, os.path.join(HIST_DIR, "mlp_frozen_decoder.json"))

    ae = VanillaAE().to(DEVICE)
    ae.load_state_dict(torch.load(os.path.join(CKPT_DIR, "ae.pt"), map_location=DEVICE))
    ae.eval()
    with torch.no_grad():
        x_fd = dec(z_mlp.to(DEVICE)).cpu()
        x_ae, z_ae = ae(xt.to(DEVICE))
        x_ae, z_ae = x_ae.cpu(), z_ae.cpu()

    def intra_class_var_ratio(xr):
        """Rasio varians intra-kelas rekonstruksi terhadap citra asli (1.0 = variasi utuh)."""
        r = []
        for c in range(10):
            m = yt == c
            r.append(float(xr[m].var(0).sum() / xt[m].var(0).sum()))
        return float(np.mean(r)), r

    fd, aem = mse_psnr_ssim(xt, x_fd), mse_psnr_ssim(xt, x_ae)
    res["stage4"] = {
        "frozen_mlp_decoder": {"mse": fd[0], "psnr": fd[1], "ssim": fd[2],
                               "intra_class_var_ratio": intra_class_var_ratio(x_fd)[0],
                               "train_time_s": time.time() - t0},
        "vanilla_ae": {"mse": aem[0], "psnr": aem[1], "ssim": aem[2],
                       "intra_class_var_ratio": intra_class_var_ratio(x_ae)[0]}}

    # ---------------- Tahap 5: sampling di ruang laten MLP
    mu, sd = z_mlp.mean(0), z_mlp.std(0)
    g = torch.Generator().manual_seed(42)
    z_rand = mu + sd * torch.randn(1000, 64, generator=g)
    with torch.no_grad():
        x_samp = dec(z_rand.to(DEVICE)).cpu()
        probs = F.softmax(clf(x_samp.to(DEVICE)), 1).cpu()
        probs_real = F.softmax(clf(xt.to(DEVICE)), 1).cpu()
    # pembanding: sampling Gaussian yang sama di ruang laten Vanilla AE
    mu_a, sd_a = z_ae.mean(0), z_ae.std(0)
    with torch.no_grad():
        x_samp_ae = ae.decode((mu_a + sd_a * torch.randn(1000, 64, generator=g)).to(DEVICE)).cpu()
    res["stage5"] = {
        "frac_negative_in_z_rand": float((z_rand < 0).float().mean()),
        "frac_zero_in_real_z_mlp": float((z_mlp == 0).float().mean()),
        "classifier_max_prob_mean_samples": float(probs.max(1)[0].mean()),
        "classifier_max_prob_mean_real": float(probs_real.max(1)[0].mean()),
        "frac_samples_confident_gt_0.9": float((probs.max(1)[0] > 0.9).float().mean()),
        "frac_real_confident_gt_0.9": float((probs_real.max(1)[0] > 0.9).float().mean()),
        "predicted_class_hist_samples": torch.bincount(probs.argmax(1), minlength=10).tolist()}

    torch.save({"x_class": x_class, "x_class_vis": x_class_vis, "x_lat": x_lat, "x_lat_vis": x_lat_vis,
                "x_fd": x_fd[:64], "x_ae": x_ae[:64], "x_samp": x_samp, "x_samp_ae": x_samp_ae,
                "z_mlp": z_mlp, "z_ae": z_ae, "class_means": class_means},
               os.path.join(CKPT_DIR, "mlp_artifacts.pt"))
    save_json(res, os.path.join(MET_DIR, "mlp_experiment.json"))
    import json
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
