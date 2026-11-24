"""Membuat diagram arsitektur + 5 figur wajib laporan (results/figures/)."""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (set_seed, get_splits, to_01, DEVICE, CKPT_DIR, HIST_DIR, FIG_DIR, MET_DIR,  # noqa: E402
                    CLASSES, load_json, save_json)
from models import VanillaAE, SparseAE, VAE, MLPGenerator, DCGenerator, MLPClassifier, NZ  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 100})
DPI = 220


def load(model, name):
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, name), map_location=DEVICE))
    return model.to(DEVICE).eval()


def grid(x, nrow):
    """x: [N,1,28,28] -> array 2D dengan padding 2px."""
    x = x.detach().cpu().clamp(0, 1).numpy()[:, 0]
    n = len(x)
    ncol = int(np.ceil(n / nrow))
    out = np.ones((nrow * 30 - 2, ncol * 30 - 2))
    for i in range(n):
        r, c = divmod(i, ncol)
        out[r * 30:r * 30 + 28, c * 30:c * 30 + 28] = x[i]
    return out


def show_row(ax, x, label):
    ax.imshow(grid(x, 1), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)


def save(fig, name):
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


# ---------------------------------------------------- Figur 0 arsitektur
def fig0_architecture():
    rows = [
        ("Vanilla AE", ["x 784", "FC 256\nReLU", "z 64\n(linear)", "FC 256\nReLU", "FC 784\nSigmoid"], "MSE"),
        ("Sparse AE", ["x 784", "FC 256\nReLU", "h 64\nReLU", "FC 256\nReLU", "FC 784\nSigmoid"], "MSE + λ‖h‖₁"),
        ("VAE", ["x 784", "FC 256\nReLU", "μ, logσ²\n64", "z=μ+σ⊙ε", "FC 256→784\nSigmoid"], "BCE + KL"),
        ("Vanilla GAN", ["z 100", "FC 256-512\nLReLU", "FC 1024\nLReLU", "FC 784\nTanh", "D: FC 512-256-1"], "BCE minimax"),
        ("DCGAN", ["z 100", "FC 128×7×7\nBN ReLU", "ConvT 64\n14×14 BN", "ConvT 1\n28×28 Tanh", "D: Conv 64-128\nBN LReLU"], "BCE minimax"),
        ("DDPM", ["x_t 1×28²\n+ t-emb", "U-Net down\n32→64→128", "Mid ResBlock\n7×7", "U-Net up\n+skip", "ε̂ 1×28²"], "‖ε − ε̂‖²"),
        ("MLP Classifier", ["x 784", "FC 256\nReLU (W1)", "z_MLP 64\nReLU (W2)", "FC 10\n(W3)", "softmax\nP(y|x)"], "Cross-Entropy"),
    ]
    cols = ["#dbe9f6", "#dbe9f6", "#e3f1dd", "#fbe5d6", "#fbe5d6", "#ece2f4", "#eeeeee"]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_xlim(0, 10.6); ax.set_ylim(0, len(rows)); ax.axis("off")
    for r, ((name, blocks, loss), col) in enumerate(zip(rows, cols)):
        y = len(rows) - r - 0.5
        ax.text(0.05, y, name, va="center", ha="left", fontsize=9.5, weight="bold")
        for i, b in enumerate(blocks):
            x0 = 1.55 + i * 1.6
            ax.add_patch(FancyBboxPatch((x0, y - 0.32), 1.3, 0.64, boxstyle="round,pad=0.02",
                                        fc=col, ec="#555", lw=0.8))
            ax.text(x0 + 0.65, y, b, ha="center", va="center", fontsize=7.3)
            if i < len(blocks) - 1:
                ax.annotate("", xy=(x0 + 1.6, y), xytext=(x0 + 1.32, y),
                            arrowprops=dict(arrowstyle="->", lw=0.8))
        ax.text(9.65, y, loss, ha="center", va="center", fontsize=8, style="italic",
                bbox=dict(fc="white", ec="#999", lw=0.6, boxstyle="round,pad=0.25"))
    ax.set_title("Skema arsitektur 6 model generatif + MLP Classifier pendamping (Fashion-MNIST 28×28)")
    save(fig, "fig0_architecture.png")


# ---------------------------------------------------------- Figur 1
def fig1_recon(S, xt):
    fig, axes = plt.subplots(4, 1, figsize=(8, 4.4))
    for ax, x, lab in zip(axes, [xt[:10], S["ae_recon"][:10], S["sae_recon"][:10], S["vae_recon"][:10]],
                          ["Ground Truth", "Vanilla AE", "Sparse AE", "VAE (μ)"]):
        show_row(ax, x, lab)
    axes[0].set_title("Figur 1. Rekonstruksi Test Set (10 sampel pertama)")
    save(fig, "fig1_reconstruction.png")


# ---------------------------------------------------------- Figur 2
def fig2_samples(S, M):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    for ax, k, lab in zip(axes, ["vae_gen", "gan_gen", "dcgan_gen", "ddpm_gen"],
                          ["A. VAE", "B. Vanilla GAN", "C. DCGAN", "D. DDPM"]):
        ax.imshow(grid(S[k][:16], 4), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{lab}\nFID = {M[k.split('_')[0]]['fid']:.1f}")
    fig.suptitle("Figur 2. Random generative sampling (grid 4×4) dari z ~ N(0, I) / x_T ~ N(0, I)", y=1.02)
    save(fig, "fig2_random_samples.png")


# ---------------------------------------------------------- Figur 3
def fig3_mlp(A, xt, R):
    fig = plt.figure(figsize=(11, 7.4))
    gs = fig.add_gridspec(6, 1, hspace=0.55)
    ax = fig.add_subplot(gs[0]); show_row(ax, xt[:10], "(1) Ground\nTruth")
    ax.set_title("Figur 3. Bukti empiris batasan bobot & fitur laten MLP Classifier", fontsize=10.5)
    ax = fig.add_subplot(gs[1]); show_row(ax, A["x_class_vis"], "(2a) W^T y_c\n(min-max)")
    ax.set_xticks([14 + 30 * i for i in range(10)]); ax.set_xticklabels(CLASSES, fontsize=6.5)
    ax = fig.add_subplot(gs[2]); show_row(ax, A["x_lat_vis"][:10], "(2b) W^T z_MLP\n(min-max)")
    ax = fig.add_subplot(gs[3]); show_row(ax, A["x_fd"][:10], "(3a) Frozen\nMLP decoder")
    ax.set_title(f"MSE {R['stage4']['frozen_mlp_decoder']['mse']:.4f} | SSIM {R['stage4']['frozen_mlp_decoder']['ssim']:.3f}", fontsize=8)
    ax = fig.add_subplot(gs[4]); show_row(ax, A["x_ae"][:10], "(3b) Vanilla AE\n(z_AE)")
    ax.set_title(f"MSE {R['stage4']['vanilla_ae']['mse']:.4f} | SSIM {R['stage4']['vanilla_ae']['ssim']:.3f}", fontsize=8)
    ax = fig.add_subplot(gs[5]); show_row(ax, A["x_samp"][:10], "(4) z_rand ~\nN(μ_z, σ_z²)")
    save(fig, "fig3_mlp_limitations.png")
    # panel tambahan: output sigmoid mentah (tanpa min-max) — bukti 'abu-abu kabur'
    fig, axes = plt.subplots(2, 1, figsize=(8, 1.9))
    show_row(axes[0], A["x_class"], "σ(W^T…y_c)\nmentah")
    show_row(axes[1], A["x_lat"][:10], "σ(W^T…z)\nmentah")
    save(fig, "fig3b_mlp_raw_sigmoid.png")


# ---------------------------------------------------------- Figur 4
@torch.no_grad()
def fig4_interp(xt, yt):
    set_seed()
    clf = load(MLPClassifier(), "mlp_classifier.pt")
    ca, cb = 1, 8                                # Trouser vs Bag (kontras)
    xa = xt[(yt == ca).nonzero()[0, 0]].unsqueeze(0).to(DEVICE)
    xb = xt[(yt == cb).nonzero()[0, 0]].unsqueeze(0).to(DEVICE)
    alphas = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], device=DEVICE).view(-1, 1)
    rows, stats = {}, {}
    for k, cls in [("ae", VanillaAE), ("sae", SparseAE), ("vae", VAE)]:
        m = load(cls(), f"{k}.pt")
        za = m.encode(xa)[0] if k == "vae" else m.encode(xa)
        zb = m.encode(xb)[0] if k == "vae" else m.encode(xb)
        if k == "vae":
            za, zb = za.view(1, -1), zb.view(1, -1)
        rows[k] = m.decode((1 - alphas) * za + alphas * zb)
    for k, cls in [("gan", MLPGenerator), ("dcgan", DCGenerator)]:
        G = load(cls(), f"{k}_G.pt")
        z = torch.randn(3000, NZ, device=DEVICE)
        p = F.softmax(clf(to_01(G(z))), 1)
        za, zb = z[p[:, ca].argmax()].view(1, -1), z[p[:, cb].argmax()].view(1, -1)
        rows[k] = to_01(G((1 - alphas) * za + alphas * zb))
    labels = {"ae": "Vanilla AE", "sae": "Sparse AE", "vae": "VAE", "gan": "Vanilla GAN", "dcgan": "DCGAN"}
    order = ["ae", "sae", "vae", "gan", "dcgan"]
    for k in order:
        x = rows[k]
        p = F.softmax(clf(x), 1)
        mid_conf = p[2:4].max(1)[0].mean().item()           # α = 0.4, 0.6
        jumps = (x[1:] - x[:-1]).flatten(1).norm(dim=1)
        stats[k] = {"mean_maxprob_mid_alpha": mid_conf,
                    "step_l2": jumps.tolist(),
                    "step_l2_max_over_mean": float(jumps.max() / jumps.mean()),
                    "pred_classes_along_path": [CLASSES[i] for i in p.argmax(1).tolist()]}
    fig, axes = plt.subplots(len(order), 1, figsize=(6.2, 6))
    for ax, k in zip(axes, order):
        show_row(ax, rows[k], labels[k])
        ax.text(1.01, 0.5, f"conf(α=.4,.6)={stats[k]['mean_maxprob_mid_alpha']:.2f}",
                transform=ax.transAxes, va="center", fontsize=7.5)
    axes[-1].set_xticks([14 + 30 * i for i in range(6)])
    axes[-1].set_xticklabels([f"α={a:.1f}" for a in alphas.view(-1).tolist()])
    axes[0].set_title(f"Figur 4. Interpolasi laten linear: {CLASSES[ca]} → {CLASSES[cb]}")
    save(fig, "fig4_latent_interpolation.png")
    save_json(stats, os.path.join(MET_DIR, "interpolation.json"))


# ---------------------------------------------------------- Figur 5
def fig5_curves(S):
    H = {k: load_json(os.path.join(HIST_DIR, f"{k}.json")) for k in ["ae", "sae", "vae", "gan", "dcgan", "ddpm"]}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    ax = axes[0, 0]
    for k, lab in [("ae", "Vanilla AE"), ("sae", "Sparse AE")]:
        ax.plot(range(1, len(H[k]["train_loss"]) + 1), H[k]["train_loss"], label=f"{lab} train")
        ax.plot(range(1, len(H[k]["val_rec_mse"]) + 1), H[k]["val_rec_mse"], "--", label=f"{lab} val MSE")
    ax.set_title("Vanilla AE & Sparse AE"); ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(fontsize=7)
    ax = axes[0, 1]
    ax.plot(H["vae"]["train_loss"], label="train −ELBO"); ax.plot(H["vae"]["val_loss"], "--", label="val −ELBO")
    ax.set_title("VAE (BCE + KL per sampel)"); ax.set_xlabel("epoch"); ax.legend(fontsize=7)
    ax = axes[0, 2]
    ax.plot(H["ddpm"]["iter_loss"], lw=0.8, label="train (per 50 iter)")
    ax.set_title("DDPM: loss prediksi derau ‖ε−ε̂‖²"); ax.set_xlabel("iterasi ×50"); ax.set_yscale("log")
    ax.legend(fontsize=7)
    for ax, k, lab in [(axes[1, 0], "gan", "Vanilla GAN (MLP)"), (axes[1, 1], "dcgan", "DCGAN")]:
        ax.plot(H[k]["iter_d_loss"], lw=0.7, label="D loss")
        ax.plot(H[k]["iter_g_loss"], lw=0.7, label="G loss")
        ax.set_title(f"{lab}: loss per 50 iterasi"); ax.set_xlabel("iterasi ×50"); ax.legend(fontsize=7)
    ax = axes[1, 2]
    for k, lab in [("gan", "Vanilla GAN"), ("dcgan", "DCGAN")]:
        ax.plot(range(1, len(H[k]["d_real"]) + 1), H[k]["d_real"], label=f"{lab} D(x)")
        ax.plot(range(1, len(H[k]["d_fake"]) + 1), H[k]["d_fake"], "--", label=f"{lab} D(G(z))")
    ax.set_title("Output Discriminator rata-rata / epoch"); ax.set_xlabel("epoch"); ax.legend(fontsize=7)
    fig.suptitle("Figur 5a. Kurva training loss keenam model", y=1.0)
    fig.tight_layout()
    save(fig, "fig5a_training_curves.png")

    traj = S["ddpm_traj"]
    ts = sorted(traj.keys(), reverse=True)
    fig, axes = plt.subplots(1, len(ts), figsize=(12, 3.3))
    for ax, t in zip(axes, ts):
        ax.imshow(grid(traj[t][:6], 6), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"t = {t}")
    fig.suptitle("Figur 5b. DDPM reverse diffusion trajectory x_T → x_0 (T = 500), 6 sampel", y=1.02)
    save(fig, "fig5b_ddpm_trajectory.png")


def main():
    set_seed()
    xt, yt = get_splits()["test"]
    S = torch.load(os.path.join(CKPT_DIR, "eval_samples.pt"))
    M = load_json(os.path.join(MET_DIR, "metrics.json"))
    A = torch.load(os.path.join(CKPT_DIR, "mlp_artifacts.pt"))
    R = load_json(os.path.join(MET_DIR, "mlp_experiment.json"))
    fig0_architecture()
    fig1_recon(S, xt)
    fig2_samples(S, M)
    fig3_mlp(A, xt, R)
    fig4_interp(xt, yt)
    fig5_curves(S)


if __name__ == "__main__":
    main()
