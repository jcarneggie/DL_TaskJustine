# Studi Komparatif Model Generatif pada Fashion-MNIST
**Deep Learning (D6421) – Sesi 07 Tugas Take-Home**

Implementasi PyTorch dan perbandingan empiris enam model generatif, yaitu **Vanilla AE, Sparse AE, VAE, Vanilla GAN (MLP), DCGAN, dan DDPM**, ditambah eksperimen yang membuktikan bahwa bobot dan fitur laten **MLP Classifier** tidak bisa dipakai langsung untuk menghasilkan citra.

## Struktur
```
src/
  common.py            # seed 42, split 80/10/10, DataLoader, metrik MSE/PSNR/SSIM, latency
  models/
    autoencoders.py    # VanillaAE, SparseAE (MSE + λ‖h‖₁), VAE (BCE + KL)
    gans.py            # MLPGenerator/Discriminator, DCGenerator/Discriminator
    ddpm.py            # U-Net kecil + forward/reverse diffusion (T=500)
    mlp_classifier.py  # MLP 784-256-64-10 + LatentDecoder 64-256-784
  train.py             # training per model: --model {ae,sae,vae,gan,dcgan,ddpm}
  mlp_experiment.py    # Bagian 2, Tahap 1–5
  evaluate.py          # MSE/PSNR/SSIM, FID, latency per batch 64, mode coverage
  figures.py           # diagram arsitektur + Figur 1–5
results/
  metrics/*.json       # semua angka di laporan
  history/*.json       # kurva loss per epoch / iterasi
  figures/*.png        # figur laporan
  train_log.txt, mlp_log.txt
```

## Cara menjalankan
```bash
pip install -r requirements.txt
bash run_all.sh          # train 6 model -> eksperimen MLP -> evaluasi -> figur
# atau satu per satu:
python src/train.py --model ae --epochs 30
python src/mlp_experiment.py
python src/evaluate.py
python src/figures.py
```
Dataset diunduh otomatis lewat `torchvision.datasets.FashionMNIST`, lalu train (60k) dan test (10k) digabung menjadi 70k citra dan dibagi ulang **80/10/10 (56.000/7.000/7.000)** dengan `seed=42`. Kalau server unduhan torchvision tidak bisa diakses, `common.py` akan memakai `data/fmnist_all.npz` (isinya 70k citra yang sama) sebagai cadangan.

## Konfigurasi utama
| Model | Laten | Loss | Optimizer | Epoch |
|---|---|---|---|---|
| Vanilla AE | 64 (linear) | MSE | Adam 1e-3 | 30 |
| Sparse AE | 64 (ReLU) | MSE + 1e-4·‖h‖₁ | Adam 1e-3 | 30 |
| VAE | 64 | BCE + KL | Adam 1e-3 | 30 |
| Vanilla GAN | z=100 | BCE non-saturating | Adam 2e-4, β=(0.5,0.999) | 40 |
| DCGAN | z=100 | BCE non-saturating | Adam 2e-4, β=(0.5,0.999) | 15 |
| DDPM | 1×28×28 | ‖ε−ε̂‖² | Adam 2e-4, EMA 0.999 | 6 |
| MLP Classifier | 64 | Cross-Entropy | Adam 1e-3 | 25 |

Semua eksperimen dijalankan di CPU (2 core). Seed tetap: `torch.manual_seed(42)`, `np.random.seed(42)`.

## Catatan penggunaan AI
Kode dan draf laporan disusun dengan bantuan asisten AI (Claude). Rinciannya ada di bagian "Pernyataan Penggunaan AI" pada laporan.
