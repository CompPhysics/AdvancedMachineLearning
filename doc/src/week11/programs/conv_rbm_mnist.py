#!/usr/bin/env python3
"""
Convolutional Restricted Boltzmann Machine for MNIST
=====================================================
PyTorch — no external quantum libraries.

Architecture
------------
The standard Binary-Binary RBM energy is

    E(x, h) = -a^T x  -  b^T h  -  x^T W h

where W is a flat weight matrix connecting all visible pixels to all
hidden units.  We replace the linear encoder  x --> W x  with a small
convolutional neural network (CNN), giving the energy

    E(x, h) = -a^T x  -  b^T h  -  h^T f_enc(x)

where  f_enc : R^{1×28×28} → R^{N_hidden}  is the CNN encoder.

The generative direction (hidden → visible probabilities) uses a
mirrored decoder  f_dec : R^{N_hidden} → R^{1×28×28}  built from
transposed convolutions (sometimes called "deconvolutions").

Encoder  (analysis path)
    Input  : (B, 1, 28, 28)   — binarised MNIST images
    Conv(1→32, 3×3, pad=1)    → (B, 32, 28, 28)
    ReLU
    MaxPool(2×2)               → (B, 32, 14, 14)
    Conv(32→64, 3×3, pad=1)   → (B, 64, 14, 14)
    ReLU
    MaxPool(2×2)               → (B, 64,  7,  7)
    Flatten                    → (B, 64×7×7) = (B, 3136)
    Linear(3136 → N_hidden)    → (B, N_hidden)   pre-activations

Decoder  (synthesis path)
    Linear(N_hidden → 3136)    → (B, 3136)
    Reshape                    → (B, 64, 7, 7)
    ConvTranspose(64→32, 4×4, stride=2, pad=1)  → (B, 32, 14, 14)
    ReLU
    ConvTranspose(32→1,  4×4, stride=2, pad=1)  → (B,  1, 28, 28)
    Sigmoid                    → pixel probabilities in (0,1)

Free energy (same structure as BB-RBM, cell 205 in lecture notes)
    F(x) = -a^T x  -  Σ_j log(1 + exp( [f_enc(x)]_j + b_j ))

    p(x) ∝ exp(-F(x))

Training
    CD-k contrastive loss:  L = mean[ F(x_data) - F(x_k) ]
    Optimiser: Adam with cosine-annealing LR schedule.

Generation
    Gibbs sampling from random initialisation:
        x^(0) ~ Bernoulli(0.5)   (noise)
        for t = 1 … T:
            h^(t) ~ Bernoulli( σ( f_enc(x^(t-1)) + b ) )
            x^(t) ~ Bernoulli( f_dec(h^(t)) )

Contents
--------
    §1   Imports and helpers
    §2   CNN Encoder and Decoder modules
    §3   ConvRBM class
    §4   MNIST data loading
    §5   Training
    §6   Generation and visualisation
    §7   Reconstruction quality check
    §8   Summary
"""

import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── reproducibility ────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device         : {device}")
print(f"PyTorch version: {torch.__version__}")

SEP = "=" * 68

# =============================================================================
# §1 — HELPERS
# =============================================================================

def binarise(x, threshold=0.5):
    """
    Convert a float tensor in [0,1] to a binary {0,1} tensor.
    Used to turn continuous MNIST pixel values into the binary visible
    units required by the BB-RBM formalism.
    """
    return (x > threshold).float()


def show_grid(images, nrow=10, title="", figsize=(12, None), save_path=None):
    """
    Display a grid of 28×28 images.

    Parameters
    ----------
    images    : (N, 1, 28, 28) or (N, 28, 28) tensor or numpy array
    nrow      : images per row
    title     : figure title
    figsize   : (width, height); height auto-computed if None
    save_path : if given, also save to this path
    """
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    images = images.squeeze(1) if images.ndim == 4 else images
    N = len(images)
    ncol = nrow
    nrows_fig = math.ceil(N / ncol)
    h = figsize[1] or nrows_fig * 1.5

    fig, axes = plt.subplots(nrows_fig, ncol, figsize=(figsize[0], h))
    axes = axes.flatten()
    for ax in axes:
        ax.axis("off")
    for i, img in enumerate(images):
        axes[i].imshow(img, cmap="gray_r", vmin=0, vmax=1,
                       interpolation="nearest")
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.show()

# =============================================================================
# §2 — CNN ENCODER AND DECODER
# =============================================================================

class CNNEncoder(nn.Module):
    """
    Convolutional encoder  x → pre-activations for hidden units.

    Replaces the linear map  x W  in the standard RBM with a two-layer
    convolutional network, allowing the model to exploit the spatial
    structure of the image.

    Architecture:
        (B, 1, 28, 28)
        → Conv(1→32, 3×3, pad=1) + BN + ReLU + MaxPool(2)
        → Conv(32→64, 3×3, pad=1) + BN + ReLU + MaxPool(2)
        → Flatten  (B, 64·7·7 = 3136)
        → Linear(3136 → n_hidden)          ← pre-activation logits

    Note: no sigmoid here.  The sigmoid is applied inside free_energy
    so that autograd can differentiate log(1 + exp(pre_act + b)).
    """

    FLAT_DIM = 64 * 7 * 7   # 3136 — fixed for 28×28 input

    def __init__(self, n_hidden: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                      # → (B, 32, 14, 14)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                      # → (B, 64,  7,  7)
        )
        self.fc = nn.Linear(self.FLAT_DIM, n_hidden)
        # Small initialisation keeps encoder logits near zero at the start,
        # preventing log1p(exp(x)) overflow before training stabilises.
        nn.init.uniform_(self.fc.weight, -0.01, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        """
        x : (B, 1, 28, 28)  —  binary visible units
        Returns pre-activations h_pre : (B, n_hidden)
        """
        B = x.size(0)
        z = self.conv(x.view(B, 1, 28, 28))  # (B, 64, 7, 7)
        return self.fc(z.view(B, -1))         # (B, n_hidden)


class CNNDecoder(nn.Module):
    """
    Convolutional decoder  h → pixel probabilities.

    Mirrors the encoder using transposed convolutions (deconvolutions).

    Architecture:
        (B, n_hidden)
        → Linear(n_hidden → 3136) + ReLU
        → Reshape (B, 64, 7, 7)
        → ConvTranspose(64→32, 4×4, stride=2, pad=1) + BN + ReLU  → (B, 32, 14, 14)
        → ConvTranspose(32→1,  4×4, stride=2, pad=1)              → (B,  1, 28, 28)
        → Sigmoid                                                  → pixel probs ∈ (0,1)
    """

    FLAT_DIM = 64 * 7 * 7

    def __init__(self, n_hidden: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_hidden, self.FLAT_DIM),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),               # → (B, 32, 14, 14)
            nn.ConvTranspose2d(32, 1,  kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),                         # → (B,  1, 28, 28)
        )

    def forward(self, h):
        """
        h : (B, n_hidden)  —  hidden unit activations
        Returns pixel_probs : (B, 1, 28, 28)  in (0, 1)
        """
        B  = h.size(0)
        z  = self.fc(h).view(B, 64, 7, 7)
        return self.deconv(z)                     # (B, 1, 28, 28)

# =============================================================================
# §3 — CONVOLUTIONAL RBM
# =============================================================================

class ConvRBM(nn.Module):
    """
    Convolutional Restricted Boltzmann Machine for 28×28 binary images.

    Energy model
    ------------
    The energy of a (visible, hidden) configuration is

        E(x, h) = - a^T x  -  b^T h  -  h^T f_enc(x)

    where:
        x        : (B, 1, 28, 28)  binary visible units
        h        : (B, n_hidden)   binary hidden units
        a        : (784,)          visible biases  (one per pixel)
        b        : (n_hidden,)     hidden  biases
        f_enc(x) : (B, n_hidden)   CNN encoder pre-activations

    This is a direct generalisation of the BB-RBM energy from the lecture
    notes (cell 199), replacing the linear term x^T W with the CNN:

        BB-RBM  :  Σ_{ij} x_i w_{ij} h_j  =  h^T (W^T x)
        ConvRBM :  h^T f_enc(x)

    Free energy (marginalise over h, lecture notes cell 205)
    ---------------------------------------------------------
        F(x) = -a^T x  -  Σ_j log(1 + exp( b_j + [f_enc(x)]_j ))

        p(x) ∝ exp(-F(x))

    Conditional probabilities
    -------------------------
        p(h_j = 1 | x) = σ( b_j + [f_enc(x)]_j )   [cf. lecture cell 211]
        p(x | h)        = f_dec(h)                   [pixel-wise Bernoulli]

    The decoder gives pixel probabilities directly, so
        p(x_i = 1 | h) = [f_dec(h)]_i

    Gibbs sampling
    --------------
    Alternates:
        h^(t) ~ Bernoulli( σ( b + f_enc(x^(t)) ) )
        x^(t+1) ~ Bernoulli( f_dec(h^(t)) )

    Training
    --------
    CD-k contrastive loss  L = mean[ F(x_data) - F(x_k) ]
    Minimising L is equivalent to maximising the log-likelihood under CD-k.
    """

    def __init__(self, n_hidden: int = 256, k: int = 1):
        super().__init__()
        self.n_hidden  = n_hidden
        self.k         = k
        self.n_visible = 784    # 28 × 28

        # CNN encoder and decoder
        self.encoder = CNNEncoder(n_hidden)
        self.decoder = CNNDecoder(n_hidden)

        # Biases
        self.a = nn.Parameter(torch.zeros(self.n_visible))   # visible biases
        self.b = nn.Parameter(torch.zeros(n_hidden))         # hidden  biases

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  ConvRBM: n_hidden={n_hidden}, k={k}, params={n_params:,}")

    # ── Conditional probabilities ─────────────────────────────────────────────

    def prob_h_given_x(self, x):
        """
        p(h_j = 1 | x) = σ( b_j + [f_enc(x)]_j )   [lecture cell 211]

        x : (B, 1, 28, 28)  →  (B, n_hidden)

        Output is clamped to (ε, 1-ε) so that torch.bernoulli never
        receives values outside [0, 1] due to floating-point edge cases.
        """
        pre = self.encoder(x)                   # (B, n_hidden)  pre-activations
        return torch.sigmoid(pre + self.b).clamp(1e-6, 1.0 - 1e-6)

    def prob_x_given_h(self, h):
        """
        p(x_i = 1 | h) = [f_dec(h)]_i           [lecture cell 217 analogue]

        h : (B, n_hidden)  →  (B, 1, 28, 28)
        """
        return self.decoder(h)                  # pixel probs in (0,1)

    # ── Sampling ──────────────────────────────────────────────────────────────

    def sample_h(self, x):
        """Sample h ~ Bernoulli( p(h|x) ).  Returns (prob_h, h_sample)."""
        p = self.prob_h_given_x(x)
        return p, torch.bernoulli(p)

    def sample_x(self, h):
        """Sample x ~ Bernoulli( p(x|h) ).  Returns (prob_x, x_sample)."""
        p = self.prob_x_given_h(h)
        return p, torch.bernoulli(p)

    # ── Free energy ───────────────────────────────────────────────────────────

    def free_energy(self, x):
        """
        F(x) = -a^T x  -  Σ_j log(1 + exp( b_j + [f_enc(x)]_j ))

        Derived from marginalising h out of the joint (lecture cell 205).
        p(x) ∝ exp(-F(x)) — lower F means higher probability.

        Numerical stability
        -------------------
        log(1 + exp(z)) is replaced by F.softplus(z), which uses the
        numerically stable formulation  max(z,0) + log(1 + exp(-|z|)).
        The naive torch.log1p(torch.exp(z)) overflows for z > 88 and
        produces inf, which propagates into NaN gradients via Adam.

        x     : (B, 1, 28, 28)  binary visible configuration
        Returns scalar per sample, shape (B,)
        """
        # Visible bias term:  -a^T x   (flatten x to (B, 784) first)
        x_flat   = x.view(x.size(0), -1)         # (B, 784)
        vis_term = x_flat @ self.a                # (B,)

        # Hidden term:  -Σ_j log(1 + exp(b_j + [f_enc(x)]_j))
        # Compute encoder ONCE and reuse (avoids a redundant forward pass).
        # F.softplus(z) = log(1 + exp(z)) computed in a numerically stable way.
        pre      = self.encoder(x)                          # (B, n_hidden)
        hid_term = F.softplus(pre + self.b).sum(dim=1)      # (B,)

        return -(vis_term + hid_term)             # (B,)

    # ── Gibbs chain ───────────────────────────────────────────────────────────

    def gibbs_k(self, x0):
        """
        Run k alternating Gibbs steps from visible state x0.

        x^(t)   → h^(t) ~ p(h | x^(t))
        h^(t)   → x^(t+1) ~ p(x | h^(t))

        Returns x_k (detached from the computation graph so that the
        negative phase does not contribute additional gradients beyond
        the free-energy term F(x_k)).
        """
        x = x0
        for _ in range(self.k):
            _, h = self.sample_h(x)    # sample h ~ p(h | x)
            _, x = self.sample_x(h)    # sample x ~ p(x | h)
        return x.detach()

    # ── Forward (for the training loop) ───────────────────────────────────────

    def forward(self, x):
        """Return (x0, x_k) for the contrastive loss  F(x0) - F(x_k)."""
        x_k = self.gibbs_k(x)
        return x, x_k

    # ── Generation from scratch ───────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, n_samples: int, n_gibbs: int = 1000,
                 init: str = "noise"):
        """
        Generate new samples by running a long Gibbs chain from scratch.

        Parameters
        ----------
        n_samples : number of images to generate
        n_gibbs   : number of Gibbs sweeps (burn-in)
        init      : "noise" — start from Bernoulli(0.5) random images
                    "zeros" — start from all-zero images

        Returns
        -------
        prob_x : (n_samples, 1, 28, 28)  final pixel probabilities
        x_k    : (n_samples, 1, 28, 28)  final binary samples
        """
        self.eval()
        if init == "zeros":
            x = torch.zeros(n_samples, 1, 28, 28, device=device)
        else:
            x = torch.bernoulli(
                0.5 * torch.ones(n_samples, 1, 28, 28, device=device))

        for _ in range(n_gibbs):
            _, h = self.sample_h(x)
            _, x = self.sample_x(h)

        # One final clean step: return smooth pixel probabilities
        # (not the hard binary sample) for better visualisation.
        with torch.no_grad():
            h_final  = torch.bernoulli(self.prob_h_given_x(x))
            prob_x   = self.prob_x_given_h(h_final)   # pixel probs in (0,1)
        return prob_x, x

# =============================================================================
# §4 — MNIST DATA LOADING
# =============================================================================

print(f"\n{SEP}")
print("§4 — LOADING MNIST")
print(SEP)

DATA_DIR   = "./data"
BATCH_SIZE = 128

transform = transforms.Compose([
    transforms.ToTensor(),                         # → [0,1] float
])

train_dataset = datasets.MNIST(
    DATA_DIR, train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(
    DATA_DIR, train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=True)

print(f"Train samples: {len(train_dataset):,}")
print(f"Test  samples: {len(test_dataset):,}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Image size   : 28×28 (binarised at threshold 0.5)")

# =============================================================================
# §5 — TRAINING
# =============================================================================

print(f"\n{SEP}")
print("§5 — TRAINING")
print(SEP)

N_HIDDEN = 256
N_EPOCHS = 20
K_GIBBS  = 1          # CD-k steps
LR_INIT  = 3e-4

model = ConvRBM(n_hidden=N_HIDDEN, k=K_GIBBS).to(device)

optimizer = optim.Adam(model.parameters(), lr=LR_INIT, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

print(f"\nTraining ConvRBM on MNIST:")
print(f"  n_hidden = {N_HIDDEN},  k = {K_GIBBS},  "
      f"epochs = {N_EPOCHS},  lr = {LR_INIT}")
print(f"  Loss = mean[ F(x_data) - F(x_k) ]  (CD-{K_GIBBS})\n")

train_losses = []
t_total = time.time()

for epoch in range(1, N_EPOCHS + 1):
    model.train()
    ep_loss = 0.0
    n_batches = 0
    t_ep = time.time()

    for images, _ in train_loader:
        # Binarise MNIST images to get binary visible units
        x = binarise(images).to(device)            # (B, 1, 28, 28)

        # CD-k forward pass
        x0, x_k = model(x)

        # Contrastive loss: F(x_data) - F(x_k)
        loss = (model.free_energy(x0) - model.free_energy(x_k)).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient clipping for stability with CNN gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        ep_loss  += loss.item()
        n_batches += 1

    scheduler.step()
    avg_loss = ep_loss / n_batches
    train_losses.append(avg_loss)
    elapsed = time.time() - t_ep

    print(f"  Epoch {epoch:3d}/{N_EPOCHS}  "
          f"loss = {avg_loss:+.4f}  "
          f"lr = {scheduler.get_last_lr()[0]:.2e}  "
          f"({elapsed:.1f}s)")

print(f"\nTotal training time: {time.time()-t_total:.1f}s")

# =============================================================================
# §6 — GENERATION AND VISUALISATION
# =============================================================================

print(f"\n{SEP}")
print("§6 — GENERATION")
print(SEP)

model.eval()

# ── Training loss curve ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(range(1, N_EPOCHS + 1), train_losses, 'b-o', ms=5, lw=2)
ax.set(xlabel='Epoch', ylabel='CD loss  F(x_data) − F(x_k)',
       title='ConvRBM training on MNIST')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('conv_rbm_training.png', dpi=130, bbox_inches='tight')
plt.show()

# ── Real MNIST samples (for comparison) ──────────────────────────────────────
real_images, real_labels = next(iter(test_loader))
real_bin = binarise(real_images[:40])
show_grid(real_bin, nrow=10, title="Real MNIST digits (binarised)",
          save_path="conv_rbm_real.png")

# ── Reconstructions (encode → decode without sampling) ───────────────────────
print("Computing reconstructions (one Gibbs step from test data)...")
with torch.no_grad():
    xb = binarise(real_images[:40]).to(device)
    p_h = model.prob_h_given_x(xb)
    h_sample = torch.bernoulli(p_h)
    recon = model.prob_x_given_h(h_sample).cpu()

show_grid(recon, nrow=10,
          title="Reconstructions  (x → enc → sample_h → dec)",
          save_path="conv_rbm_recon.png")

# ── Generated samples (long Gibbs chain from noise) ──────────────────────────
print("Generating 80 new digit images  (1000 Gibbs burn-in steps)...")
gen_probs, gen_samples = model.generate(n_samples=80, n_gibbs=1000,
                                         init="noise")

show_grid(gen_probs.cpu(), nrow=10,
          title="Generated MNIST digits  (ConvRBM, 1000 Gibbs steps from noise)",
          figsize=(12, None), save_path="conv_rbm_generated.png")

# ── Generated samples starting from zeros ────────────────────────────────────
print("Generating from all-zero initialisation...")
gen_probs_z, _ = model.generate(n_samples=40, n_gibbs=1000, init="zeros")

show_grid(gen_probs_z.cpu(), nrow=10,
          title="Generated from zeros  (1000 Gibbs steps)",
          save_path="conv_rbm_generated_zeros.png")

# ── Gibbs chain evolution (watch a single sample evolve) ─────────────────────
print("Tracking one Gibbs chain over 800 steps...")
snapshots = []
snap_steps = [0, 10, 50, 100, 200, 400, 600, 800]
with torch.no_grad():
    x = torch.bernoulli(0.5 * torch.ones(1, 1, 28, 28, device=device))
    step = 0
    if step in snap_steps:
        snapshots.append((step, x.cpu()))
    for t in range(1, 801):
        _, h = model.sample_h(x)
        _, x = model.sample_x(h)
        if t in snap_steps:
            snapshots.append((t, x.cpu()))

fig, axes = plt.subplots(1, len(snapshots), figsize=(len(snapshots)*1.6, 2))
for ax, (step, img) in zip(axes, snapshots):
    ax.imshow(img.squeeze().numpy(), cmap='gray_r', vmin=0, vmax=1)
    ax.set_title(f't={step}', fontsize=8)
    ax.axis('off')
fig.suptitle('Gibbs chain evolution (single sample, random start)',
             fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('conv_rbm_chain.png', dpi=130, bbox_inches='tight')
plt.show()

# =============================================================================
# §7 — RECONSTRUCTION QUALITY
# =============================================================================

print(f"\n{SEP}")
print("§7 — RECONSTRUCTION QUALITY ON TEST SET")
print(SEP)

model.eval()
total_mse = 0.0
n_test    = 0

with torch.no_grad():
    for images, _ in test_loader:
        xb    = binarise(images).to(device)     # (B, 1, 28, 28)
        p_h   = model.prob_h_given_x(xb)
        h_s   = torch.bernoulli(p_h)
        recon = model.prob_x_given_h(h_s)
        total_mse += ((xb - recon)**2).mean(dim=[1,2,3]).sum().item()
        n_test    += len(xb)

avg_mse = total_mse / n_test
print(f"\nTest-set reconstruction MSE: {avg_mse:.5f}")
print(f"(per pixel, averaged over {n_test:,} images)")

# Per-digit-class breakdown
print(f"\nPer-class reconstruction MSE:")
class_mse  = {d: 0.0 for d in range(10)}
class_cnt  = {d: 0   for d in range(10)}
with torch.no_grad():
    for images, labels in test_loader:
        xb    = binarise(images).to(device)
        p_h   = model.prob_h_given_x(xb)
        h_s   = torch.bernoulli(p_h)
        recon = model.prob_x_given_h(h_s)
        mse_per = ((xb - recon)**2).mean(dim=[1,2,3])
        for d in range(10):
            mask = (labels == d)
            if mask.any():
                class_mse[d] += mse_per[mask].sum().item()
                class_cnt[d] += mask.sum().item()

print(f"  {'Digit':>6s}  {'MSE':>8s}  {'n_test':>8s}")
print("  " + "-"*28)
for d in range(10):
    m = class_mse[d] / (class_cnt[d] + 1e-8)
    print(f"  {d:>6d}  {m:>8.5f}  {class_cnt[d]:>8d}")

# =============================================================================
# §8 — SUMMARY
# =============================================================================

print(f"\n{SEP}")
print("§8 — SUMMARY")
print(SEP)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"""
ConvRBM Architecture
--------------------
Visible units  : 784  (28×28 binarised MNIST pixels)
Hidden units   : {N_HIDDEN}
Encoder        : Conv(1→32) + BN + ReLU + Pool
                 Conv(32→64) + BN + ReLU + Pool
                 Flatten → Linear(3136 → {N_HIDDEN})
Decoder        : Linear({N_HIDDEN} → 3136) + ReLU
                 ConvTranspose(64→32) + BN + ReLU
                 ConvTranspose(32→1) + Sigmoid
Total params   : {n_params:,}

Energy model
------------
  E(x, h) = -a^T x  -  b^T h  -  h^T f_enc(x)
  F(x)    = -a^T x  -  Σ_j log(1 + exp(b_j + [f_enc(x)]_j))
  p(x)    ∝ exp(-F(x))

This directly generalises the BB-RBM (lecture notes cells 199–217)
by replacing the linear encoder  x → W x  with a CNN  x → f_enc(x).

Training
--------
  Loss     : CD-{K_GIBBS}  mean[ F(x_data) - F(x_k) ]
  Optimiser: Adam + cosine LR, {N_EPOCHS} epochs
  Final loss: {train_losses[-1]:+.4f}
  Test MSE : {avg_mse:.5f}

Generation
----------
  Start from Bernoulli(0.5) noise
  Run 1000 Gibbs steps:  h ~ p(h|x),  x ~ p(x|h)
  Output = final pixel probability map

Saved figures
-------------
  conv_rbm_training.png      — loss curve
  conv_rbm_real.png          — real MNIST digits (binarised)
  conv_rbm_recon.png         — reconstructions from test data
  conv_rbm_generated.png     — generated samples (noise start)
  conv_rbm_generated_zeros.png — generated samples (zero start)
  conv_rbm_chain.png         — Gibbs chain evolution
""")
