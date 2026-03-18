"""
GAN-Based Simple Channel Model
================================
Self-contained example based on GAN-explain.md.

Pipeline:
  1. Simulate Rayleigh fading channel (replaces QuaDRiGa)
  2. Define generate_real_samples() exactly as in the paper
  3. Train a Conditional GAN (CGAN)
  4. Evaluate and plot results
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────
# 1.  RAYLEIGH CHANNEL DATASET  (replaces QuaDRiGa)
# ──────────────────────────────────────────────────────────────────
np.random.seed(42)
torch.manual_seed(42)

N_SAMPLES = 20_000
sigma = 1.0 / np.sqrt(2)                        # each component ~ N(0, 0.5)
h_r = np.random.normal(0, sigma, N_SAMPLES)
h_i = np.random.normal(0, sigma, N_SAMPLES)
h_dataset = (h_r + 1j * h_i).astype(np.complex64)   # shape (20000,)

# ──────────────────────────────────────────────────────────────────
# 2.  QPSK CONSTELLATION  (mean_set_QAM)
# ──────────────────────────────────────────────────────────────────
mean_set_QAM = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64) / np.sqrt(2)

# ──────────────────────────────────────────────────────────────────
# 3.  REAL-SAMPLE GENERATOR  (exactly from GAN-explain.md)
# ──────────────────────────────────────────────────────────────────
def generate_real_samples(h_dataset, number=256):
    h_complex = np.random.choice(h_dataset, number)          # random channel realizations
    h_r = np.real(h_complex)
    h_i = np.imag(h_complex)

    labels_index = np.random.choice(len(mean_set_QAM), number)
    data = mean_set_QAM[labels_index]                        # random QAM symbols

    received = h_complex * data                              # y = h · x
    received = np.hstack([
        np.real(received).reshape(number, 1),
        np.imag(received).reshape(number, 1),
    ])

    noise = np.random.multivariate_normal(
        [0, 0], [[0.01, 0], [0, 0.01]], number
    ).astype(np.float32)
    received = received + noise                              # AWGN  (σ² = 0.01)

    conditioning = np.hstack([
        np.real(data).reshape(number, 1),
        np.imag(data).reshape(number, 1),
        h_r.reshape(number, 1),
        h_i.reshape(number, 1),
    ]) / 3.0                                                 # normalize conditioning

    return received.astype(np.float32), conditioning.astype(np.float32)

# ──────────────────────────────────────────────────────────────────
# 4.  CONDITIONAL GAN ARCHITECTURE
# ──────────────────────────────────────────────────────────────────
NOISE_DIM = 8    # latent noise vector
COND_DIM  = 4    # [Re(x), Im(x), Re(h), Im(h)]
OUT_DIM   = 2    # [Re(y), Im(y)]

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NOISE_DIM + COND_DIM, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 64),                   nn.LeakyReLU(0.2),
            nn.Linear(64, OUT_DIM),
        )

    def forward(self, z, c):
        return self.net(torch.cat([z, c], dim=1))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OUT_DIM + COND_DIM, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 64),                 nn.LeakyReLU(0.2),
            nn.Linear(64, 1),                  nn.Sigmoid(),
        )

    def forward(self, y, c):
        return self.net(torch.cat([y, c], dim=1))

# ──────────────────────────────────────────────────────────────────
# 5.  TRAINING
# ──────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

G = Generator().to(device)
D = Discriminator().to(device)

g_opt = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
d_opt = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion = nn.BCELoss()

N_EPOCHS   = 3000
BATCH_SIZE = 256
g_losses, d_losses = [], []

for epoch in range(N_EPOCHS):
    real_y, cond = generate_real_samples(h_dataset, BATCH_SIZE)
    real_y = torch.FloatTensor(real_y).to(device)
    cond_t = torch.FloatTensor(cond).to(device)

    real_lbl = torch.ones (BATCH_SIZE, 1).to(device)
    fake_lbl = torch.zeros(BATCH_SIZE, 1).to(device)

    # ── Discriminator step ──────────────────
    z      = torch.randn(BATCH_SIZE, NOISE_DIM).to(device)
    fake_y = G(z, cond_t).detach()
    d_loss = criterion(D(real_y, cond_t), real_lbl) + \
             criterion(D(fake_y, cond_t), fake_lbl)
    d_opt.zero_grad(); d_loss.backward(); d_opt.step()

    # ── Generator step ──────────────────────
    z      = torch.randn(BATCH_SIZE, NOISE_DIM).to(device)
    fake_y = G(z, cond_t)
    g_loss = criterion(D(fake_y, cond_t), real_lbl)
    g_opt.zero_grad(); g_loss.backward(); g_opt.step()

    g_losses.append(g_loss.item())
    d_losses.append(d_loss.item())

    if (epoch + 1) % 600 == 0:
        print(f"  Epoch {epoch+1:4d}/{N_EPOCHS} | "
              f"D_loss={d_loss.item():.4f}  G_loss={g_loss.item():.4f}")

print("Training complete.")

# ──────────────────────────────────────────────────────────────────
# 6.  EVALUATION — generate fake samples
# ──────────────────────────────────────────────────────────────────
G.eval()
N_EVAL = 5000

real_y_eval, cond_eval = generate_real_samples(h_dataset, N_EVAL)
cond_eval_t = torch.FloatTensor(cond_eval).to(device)
z_eval      = torch.randn(N_EVAL, NOISE_DIM).to(device)

with torch.no_grad():
    fake_y_eval = G(z_eval, cond_eval_t).cpu().numpy()

# Recover h from conditioning  (c[2]*3 = Re(h), c[3]*3 = Im(h))
h_r_eval  = cond_eval[:, 2] * 3.0
h_i_eval  = cond_eval[:, 3] * 3.0

# ──────────────────────────────────────────────────────────────────
# 7.  PLOTS
# ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("GAN-Based Channel Model — Results", fontsize=14, fontweight="bold")

# ── Plot 1: Training losses ──────────────────
ax = axes[0, 0]
ax.plot(g_losses, label="Generator",     color="royalblue", alpha=0.8, lw=1)
ax.plot(d_losses, label="Discriminator", color="tomato",    alpha=0.8, lw=1)
ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss")
ax.set_title("Training Losses")
ax.legend(); ax.grid(True, alpha=0.3)

# ── Plot 2: Received signal scatter ─────────
ax = axes[0, 1]
ax.scatter(real_y_eval[:800, 0], real_y_eval[:800, 1],
           s=6, alpha=0.5, c="royalblue", label="Real (Rayleigh sim.)")
ax.scatter(fake_y_eval[:800, 0], fake_y_eval[:800, 1],
           s=6, alpha=0.5, c="tomato",    label="GAN Generated")
ax.set_xlabel("Re(y)"); ax.set_ylabel("Im(y)")
ax.set_title("Received Signal  y = h·x + n")
ax.legend(); ax.grid(True, alpha=0.3)

# ── Plot 3: Channel coefficient scatter ─────
ax = axes[0, 2]
ax.scatter(np.real(h_dataset[:2000]), np.imag(h_dataset[:2000]),
           s=4, alpha=0.3, c="royalblue", label="True h (dataset)")
ax.scatter(h_r_eval[:500], h_i_eval[:500],
           s=4, alpha=0.4, c="orange", label="Eval batch h")
ax.set_xlabel("Re(h)"); ax.set_ylabel("Im(h)")
ax.set_title("Channel Coefficients  h ~ Rayleigh")
ax.legend(); ax.grid(True, alpha=0.3)

# ── Plot 4: Channel envelope distribution ───
ax = axes[1, 0]
h_mag = np.abs(h_dataset)
bins  = np.linspace(0, 3.5, 70)
ax.hist(h_mag, bins=bins, density=True, alpha=0.6, color="royalblue", label="|h| simulated")

# Theoretical Rayleigh PDF: f(r) = (r/σ²)·exp(−r²/2σ²),  σ² = 0.5
r_vals = np.linspace(0, 3.5, 400)
rayleigh_pdf = (r_vals / 0.5) * np.exp(-r_vals**2 / (2 * 0.5))
ax.plot(r_vals, rayleigh_pdf, "r-", lw=2, label="Rayleigh PDF (theory)")
ax.set_xlabel("|h|"); ax.set_ylabel("Density")
ax.set_title("Channel Envelope Distribution")
ax.legend(); ax.grid(True, alpha=0.3)

# ── Plot 5: Re(y) histogram ─────────────────
ax = axes[1, 1]
ax.hist(real_y_eval[:, 0], bins=60, density=True,
        alpha=0.6, color="royalblue", label="Real Re(y)")
ax.hist(fake_y_eval[:, 0], bins=60, density=True,
        alpha=0.6, color="tomato",    label="GAN  Re(y)")
ax.set_xlabel("Re(y)"); ax.set_ylabel("Density")
ax.set_title("Re(y)  Histogram Comparison")
ax.legend(); ax.grid(True, alpha=0.3)

# ── Plot 6: Im(y) histogram ─────────────────
ax = axes[1, 2]
ax.hist(real_y_eval[:, 1], bins=60, density=True,
        alpha=0.6, color="royalblue", label="Real Im(y)")
ax.hist(fake_y_eval[:, 1], bins=60, density=True,
        alpha=0.6, color="tomato",    label="GAN  Im(y)")
ax.set_xlabel("Im(y)"); ax.set_ylabel("Density")
ax.set_title("Im(y)  Histogram Comparison")
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
import os, glob
existing = glob.glob("/home/iccl813/Downloads/ai-wireless/Solutions/gan_channel_results_*.png")
next_num = len(existing) + 1
out_path = f"/home/iccl813/Downloads/ai-wireless/Solutions/gan_channel_results_{next_num}.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved → {out_path}")
