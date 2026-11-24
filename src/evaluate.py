"""Evaluasi Test Set: MSE/PSNR/SSIM (rekonstruksi), FID (sampel sintetis),
inference latency per batch 64, training time per epoch, mode coverage.

FID: InceptionV3 pool3 (pytorch-fid), citra grayscale diulang 3 kanal dan
di-resize ke 299x299. Statistik real = seluruh Test Set (7000 citra);
setiap model generatif menyumbang N_GEN = 1000 sampel (N sama untuk semua
model agar bias ukuran sampel FID setara).
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (set_seed, get_splits, to_01, DEVICE, CKPT_DIR, HIST_DIR, MET_DIR,  # noqa: E402
                    mse_psnr_ssim, latency_ms, save_json, load_json)
from models import (VanillaAE, SparseAE, VAE, MLPGenerator, DCGenerator, DDPM,  # noqa: E402
                    MLPClassifier, NZ)

N_GEN = 1000


def load(model, name):
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, name), map_location=DEVICE))
    return model.to(DEVICE).eval()


# ------------------------------------------------------------------ FID
_INC = None


@torch.no_grad()
def inception_acts(x01, bs=200):
    global _INC
    if _INC is None:
        from pytorch_fid.inception import InceptionV3
        _INC = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(DEVICE).eval()
    out = []
    for i in range(0, len(x01), bs):
        b = x01[i:i + bs].to(DEVICE).repeat(1, 3, 1, 1)
        out.append(_INC(b)[0].squeeze(-1).squeeze(-1).cpu())
    return torch.cat(out).numpy().astype(np.float64)


def fid_from_acts(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def mode_coverage(clf, x01):
    with torch.no_grad():
        pred = clf(x01.to(DEVICE)).argmax(1).cpu()
    h = torch.bincount(pred, minlength=10).float()
    p = h / h.sum()
    ent = float(-(p[p > 0] * p[p > 0].log()).sum() / np.log(10))   # 1.0 = seragam
    return {"class_hist": h.int().tolist(), "norm_entropy": ent,
            "n_modes_ge_5pct": int((p >= 0.05).sum())}


def bench_train_epoch(name, iters=20):
    """Waktu training per epoch diukur terisolasi (tanpa proses lain di CPU):
    rata-rata waktu per iterasi (batch 128) x jumlah batch per epoch (438)."""
    import time
    import torch.nn as nn
    from models import SparseAE as _S, MLPDiscriminator, DCDiscriminator  # noqa: F401
    x = torch.rand(128, 1, 28, 28, device=DEVICE)
    n_batches = int(np.ceil(len(get_splits()["train"][0]) / 128))
    if name in ("ae", "sae", "vae"):
        m = {"ae": VanillaAE, "sae": SparseAE, "vae": VAE}[name]().to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), 1e-3)

        def step():
            loss, _ = m.loss(x)
            opt.zero_grad(); loss.backward(); opt.step()
    elif name in ("gan", "dcgan"):
        G, D = ((MLPGenerator(), MLPDiscriminator()) if name == "gan" else (DCGenerator(), DCDiscriminator()))
        G, D = G.to(DEVICE), D.to(DEVICE)
        oG, oD = torch.optim.Adam(G.parameters(), 2e-4), torch.optim.Adam(D.parameters(), 2e-4)
        bce, ones, zeros = nn.BCEWithLogitsLoss(), torch.ones(128), torch.zeros(128)
        xr = x * 2 - 1

        def step():
            fake = G(torch.randn(128, NZ))
            dl = bce(D(xr), ones) + bce(D(fake.detach()), zeros)
            oD.zero_grad(); dl.backward(); oD.step()
            gl = bce(D(fake), ones)
            oG.zero_grad(); gl.backward(); oG.step()
    else:
        m = DDPM().to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), 2e-4)
        iters = 6

        def step():
            loss, _ = m.loss(x * 2 - 1)
            opt.zero_grad(); loss.backward(); opt.step()
    for _ in range(2):
        step()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    return (time.perf_counter() - t0) / iters * n_batches


def main():
    set_seed()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    xt, yt = get_splits()["test"]
    x64 = xt[:64].to(DEVICE)
    M, samples = {}, {}
    clf = load(MLPClassifier(), "mlp_classifier.pt")

    # ------------------------------------------- AE family: rekonstruksi
    aes = {"ae": load(VanillaAE(), "ae.pt"), "sae": load(SparseAE(), "sae.pt"), "vae": load(VAE(), "vae.pt")}
    for k, m in aes.items():
        with torch.no_grad():
            if k == "vae":
                xr, z = m.reconstruct(xt.to(DEVICE)).cpu(), m.encode(xt.to(DEVICE))[0].cpu()
                lat = latency_ms(lambda: m.reconstruct(x64))
            else:
                xr, z = [t.cpu() for t in m(xt.to(DEVICE))]
                lat = latency_ms(lambda: m(x64))
        mse, psnr, ssim = mse_psnr_ssim(xt, xr)
        M[k] = {"test_mse": mse, "test_psnr": psnr, "test_ssim": ssim, "latency_recon_ms_b64": lat,
                "latent_frac_near_zero": float((z.abs() < 1e-3).float().mean()),
                "latent_mean_abs": float(z.abs().mean()),
                "latent_active_units": int(((z.abs() > 1e-3).float().mean(0) > 0.01).sum())}
        samples[f"{k}_recon"] = xr[:64]
        print(k, M[k], flush=True)

    # -------------------------------------------- generator: sampling
    g = torch.Generator().manual_seed(123)
    vae = aes["vae"]
    gan = load(MLPGenerator(), "gan_G.pt")
    dcg = load(DCGenerator(), "dcgan_G.pt")
    ddpm = load(DDPM(), "ddpm.pt")
    z_v = torch.randn(N_GEN, vae.latent, generator=g).to(DEVICE)
    z_g = torch.randn(N_GEN, NZ, generator=g).to(DEVICE)
    with torch.no_grad():
        gen = {"vae": vae.decode(z_v).cpu(), "gan": to_01(gan(z_g)).cpu(), "dcgan": to_01(dcg(z_g)).cpu()}
    torch.manual_seed(7)
    ddpm_x = []
    for i in range(0, N_GEN, 250):
        ddpm_x.append(to_01(ddpm.sample(250, DEVICE)).cpu())
        print("ddpm sampled", i + 250, flush=True)
    gen["ddpm"] = torch.cat(ddpm_x)

    zb_v, zb_g = torch.randn(64, vae.latent, device=DEVICE), torch.randn(64, NZ, device=DEVICE)
    M["vae"]["latency_gen_ms_b64"] = latency_ms(lambda: vae.decode(zb_v))
    M["gan"] = {"latency_gen_ms_b64": latency_ms(lambda: gan(zb_g))}
    M["dcgan"] = {"latency_gen_ms_b64": latency_ms(lambda: dcg(zb_g))}
    M["ddpm"] = {"latency_gen_ms_b64": latency_ms(lambda: ddpm.sample(64, DEVICE), repeats=3, warmup=1),
                 "sampling_steps": ddpm.T}

    # -------------------------------------------------------------- FID
    real_acts = inception_acts(xt)
    tr_x = get_splits()["train"][0]
    ref = fid_from_acts(inception_acts(tr_x[torch.randperm(len(tr_x), generator=g)[:N_GEN]]), real_acts)
    M["reference_fid_real_train_1000_vs_test"] = ref
    print("FID floor (real vs real, N=1000):", ref, flush=True)
    for k, x in gen.items():
        M[k]["fid"] = fid_from_acts(inception_acts(x), real_acts)
        M[k]["mode_coverage"] = mode_coverage(clf, x)
        samples[f"{k}_gen"] = x[:64]
        print(k, "FID", M[k]["fid"], M[k]["mode_coverage"], flush=True)
    M["real_test_mode_coverage"] = mode_coverage(clf, xt)

    # FID tambahan: AE deterministik & ruang laten MLP saat di-sample Gaussian
    art = torch.load(os.path.join(CKPT_DIR, "mlp_artifacts.pt"))
    M["extra_fid"] = {"vanilla_ae_gaussian_latent_sampling": fid_from_acts(inception_acts(art["x_samp_ae"]), real_acts),
                      "mlp_latent_gaussian_sampling": fid_from_acts(inception_acts(art["x_samp"]), real_acts)}
    print("extra", M["extra_fid"], flush=True)

    # --------------------------------------- training time & stabilitas
    for k in ["ae", "sae", "vae", "gan", "dcgan", "ddpm"]:
        h = load_json(os.path.join(HIST_DIR, f"{k}.json"))
        M[k]["train_time_per_epoch_s_logged"] = float(np.mean(h["epoch_time"]))
        M[k]["train_time_per_epoch_s"] = bench_train_epoch(k)
        M[k]["epochs"] = len(h["epoch_time"])
        if k in ("gan", "dcgan"):
            gl = np.array(h["g_loss"])
            M[k]["g_loss_std_last10"] = float(gl[-10:].std())
            M[k]["iter_g_loss_cv"] = float(np.std(h["iter_g_loss"]) / np.mean(h["iter_g_loss"]))
            M[k]["d_real_last"], M[k]["d_fake_last"] = h["d_real"][-1], h["d_fake"][-1]
        else:
            tl = np.array(h["train_loss"])
            M[k]["final_train_loss"] = float(tl[-1])
            M[k]["loss_monotonic_frac"] = float((np.diff(tl) <= 0).mean())

    with torch.no_grad():
        _, traj = ddpm.sample(8, DEVICE, return_traj_at=[500, 400, 300, 200, 150, 100, 50, 20, 0])
    samples["ddpm_traj"] = {k: to_01(v).cpu() for k, v in traj.items()}
    torch.save(samples, os.path.join(CKPT_DIR, "eval_samples.pt"))
    save_json(M, os.path.join(MET_DIR, "metrics.json"))
    print("DONE")


if __name__ == "__main__":
    main()
