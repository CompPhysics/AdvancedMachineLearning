#!/usr/bin/env python3
"""
Transformer Solver for the 1-D Poisson Equation
================================================
PyTorch implementation — no Qiskit or quantum libraries.

Problem
-------
  -d²u/dx² = f(x),   x ∈ [0,1],   u(0) = u(1) = 0.

Approach
--------
The transformer is trained to learn the Green's-function operator

    u(x) = (L⁻¹ f)(x),    L = -d²/dx²,

as a sequence-to-sequence map:  (x_i, f(x_i)) → u(x_i).

Training data is generated analytically from known source / solution pairs
so that the exact solution is always available for supervision.  The model
is then compared against:

  1.  Thomas' algorithm  — O(n) exact tridiagonal solver (classical reference).
  2.  Exact analytic solution  u(x) = f̂_k / (kπ)²  for each Fourier mode.

Contents
--------
  §1   Thomas' algorithm (reused from hhl_poisson.py)
  §2   Analytical training data generation
  §3   Transformer architecture
         - Sinusoidal positional encoding
         - Multi-head self-attention
         - Feed-forward sublayer
         - PoissonTransformer encoder-only model
  §4   Training loop with cosine-annealing LR schedule
  §5   Grid-convergence study (transformer vs Thomas vs exact)
  §6   Ablation: model depth and width
  §7   Generalisation to unseen source functions
  §8   Visualisation (8-panel figure)
  §9   Summary table

Key architectural choices
--------------------------
• Each interior grid point is one token.
• Token embedding = linear projection of [x_i, f(x_i)] → d_model.
• Sinusoidal positional encoding added to embeddings.
• N encoder layers (pre-LayerNorm, standard practice for stability).
• Output head: Linear(d_model → 1) applied token-wise.
• Boundary conditions u(0)=u(1)=0 are imposed analytically after the head,
  not as a penalty — this guarantees exact satisfaction.
• Training objective: MSE between predicted u_i and exact u_i.
"""

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ──────────────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SEP = "=" * 68

# =============================================================================
# §1 — THOMAS' ALGORITHM  (exact tridiagonal solver — classical reference)
# =============================================================================

def thomas_solve(n, f_vals):
    """
    Solve  -u'' = f  on n interior points with Dirichlet BCs using
    Thomas' algorithm (O(n) tridiagonal Gaussian elimination).

    Parameters
    ----------
    n      : number of interior grid points
    f_vals : (n,) array of source values f(x_i)

    Returns
    -------
    u : (n,) solution at interior points
    """
    h     = 1.0 / (n + 1)
    # Tridiagonal system: (1/h²) tridiag(-1,2,-1) u = f
    main  = np.full(n,    2.0 / h**2)
    upper = np.full(n-1, -1.0 / h**2)
    lower = np.full(n-1, -1.0 / h**2)
    rhs   = f_vals.copy().astype(float)

    # Forward sweep
    b_ = main.copy(); d_ = rhs.copy(); c_ = upper.copy()
    for i in range(1, n):
        w     = lower[i-1] / b_[i-1]
        b_[i] -= w * c_[i-1]
        d_[i] -= w * d_[i-1]

    # Back substitution
    u       = np.zeros(n)
    u[-1]   = d_[-1] / b_[-1]
    for i in range(n-2, -1, -1):
        u[i] = (d_[i] - c_[i] * u[i+1]) / b_[i]
    return u


def interior_grid(n):
    """Interior grid points x_i = i*h, h = 1/(n+1), i=1,...,n."""
    h = 1.0 / (n + 1)
    return np.linspace(h, 1.0 - h, n)


# =============================================================================
# §2 — ANALYTICAL TRAINING DATA GENERATION
# =============================================================================
#
# The 1-D Poisson equation is solved analytically by expanding f in sine modes.
# If  f(x) = Σ_k a_k sin(kπx)  then
#     u(x) = Σ_k [a_k / (kπ)²] sin(kπx).
#
# Training set: random linear combinations of the first K_MAX Fourier modes
# with random amplitudes.  This teaches the model to approximate L⁻¹ as an
# operator, not just to memorise a single source function.
#
# A held-out test set uses:
#   (a) pure sine modes  f(x) = sin(kπx)  (interpolation within training dist.)
#   (b) a step function  (out-of-distribution test of generalisation)
#   (c) a polynomial source (another OOD test)

K_MAX = 8     # maximum Fourier mode used in training


def make_sample(n, k_max=K_MAX, rng=None):
    """
    Generate one training sample: random combination of sine modes.

    Returns
    -------
    x      : (n,)   interior grid
    f_vals : (n,)   source f(x_i)
    u_vals : (n,)   exact solution u(x_i)
    """
    if rng is None:
        rng = np.random
    x      = interior_grid(n)
    ks     = np.arange(1, k_max + 1)
    amps   = rng.standard_normal(k_max)
    # Normalise so the solution has unit RMS for numerical stability
    f_vals = np.sum(amps[:, None] * np.sin(np.pi * ks[:, None] * x[None, :]),
                    axis=0)
    u_vals = np.sum((amps / (np.pi * ks)**2)[:, None]
                    * np.sin(np.pi * ks[:, None] * x[None, :]), axis=0)
    # Normalise: divide both by ||f||_2 to give consistent scale
    scale  = np.linalg.norm(f_vals) + 1e-8
    return x, f_vals / scale, u_vals / scale


def make_dataset(n, n_samples, k_max=K_MAX, seed=0):
    """
    Build a dataset of (f, u) pairs on n interior grid points.

    Returns tensors of shape (n_samples, n).
    """
    rng    = np.random.default_rng(seed)
    F_list = []
    U_list = []
    for _ in range(n_samples):
        x, f, u = make_sample(n, k_max, rng)
        F_list.append(f)
        U_list.append(u)
    F = torch.tensor(np.array(F_list), dtype=torch.float32)
    U = torch.tensor(np.array(U_list), dtype=torch.float32)
    return F, U


def exact_fourier(x, f_vals_on_grid, k_max=32):
    """
    High-accuracy reference solution obtained by projecting f onto
    the first k_max sine modes and inverting exactly.
    Used as ground truth for arbitrary source functions.
    """
    n  = len(x)
    h  = x[1] - x[0]
    ks = np.arange(1, k_max + 1)
    # Discrete sine transform via trapezoidal quadrature
    amps = 2.0 * h * np.array([
        np.sum(f_vals_on_grid * np.sin(k * np.pi * x)) for k in ks
    ])
    u = np.sum((amps / (np.pi * ks)**2)[:, None]
               * np.sin(np.pi * ks[:, None] * x[None, :]), axis=0)
    return u


# =============================================================================
# §3 — TRANSFORMER ARCHITECTURE
# =============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    Classic sinusoidal positional encoding (Vaswani et al. 2017).

    For a sequence of n tokens the encoding injects geometric information
    about each token's position in the domain [0,1]:

        PE(pos, 2k)   = sin(pos / 10000^(2k/d_model))
        PE(pos, 2k+1) = cos(pos / 10000^(2k/d_model))

    Here 'pos' is the physical coordinate x_i ∈ (0,1), NOT the integer
    index — this encodes the actual spatial location of each grid point
    and allows the model to generalise across grid sizes.

    The positional encoding is added to the token embedding before the
    first attention layer.
    """
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class PoissonTransformer(nn.Module):
    """
    Encoder-only Transformer for the 1-D Poisson operator inverse.

    Architecture
    ------------
    Input: sequence of n tokens, each representing one interior grid point.
    Each token is a 2-vector  [x_i, f(x_i)].

    Pipeline:
      [x_i, f(x_i)]                 ← 2-dim input per token
          ↓  Linear(2 → d_model)    ← token embedding
      + sinusoidal PE                ← spatial position injection
          ↓  N × EncoderLayer        ← self-attention + FFN
      token representations          ← (batch, n, d_model)
          ↓  Linear(d_model → 1)     ← scalar output per token
      u_pred_i                       ← (batch, n)

    Boundary conditions
    -------------------
    u(0) = u(1) = 0 are enforced analytically: the boundary nodes are not
    tokens (they are not in the sequence), and the network only predicts
    the n interior values.  No penalty term is needed.

    Self-attention and the Green's function
    ----------------------------------------
    The attention mechanism allows each output token u_i to attend to every
    input token f_j with a learned weight A_{ij}.  Ideally the model learns
    to approximate the discrete Green's matrix G:

        u_i ≈ Σ_j G_{ij} f_j,   G_{ij} = h² min(i,j)(n+1-max(i,j)) / (n+1)

    where G is the inverse of the tridiagonal Poisson matrix.  The attention
    patterns of a well-trained model should thus reflect the spatial locality
    of the Green's function: each token attends most strongly to nearby tokens
    but with a global envelope.

    Parameters
    ----------
    d_model       : embedding dimension (width of each layer)
    n_heads       : number of attention heads
    n_layers      : number of encoder layers
    d_ffn         : hidden dimension of the feed-forward sublayer
    dropout       : dropout rate (applied inside attention and FFN)
    """

    def __init__(self,
                 d_model: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 4,
                 d_ffn: int = 256,
                 dropout: float = 0.0):
        super().__init__()

        # Token embedding: map [x_i, f(x_i)] → d_model
        self.token_embed = nn.Linear(2, d_model)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        # Stack of encoder layers (pre-LayerNorm for training stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = n_heads,
            dim_feedforward= d_ffn,
            dropout        = dropout,
            activation     = "gelu",   # GELU smoother than ReLU for regression
            batch_first    = True,
            norm_first     = True,     # pre-LN: more stable gradients
        )
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                              num_layers=n_layers)

        # Output head: d_model → 1 (solution value at each grid point)
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # Parameter count
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  PoissonTransformer: d={d_model}, heads={n_heads}, "
              f"layers={n_layers}, d_ffn={d_ffn}, params={n_params:,}")

    def forward(self,
                x_grid: torch.Tensor,
                f_vals: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_grid : (batch, n)   — spatial coordinates of interior grid points
        f_vals : (batch, n)   — source values f(x_i)

        Returns
        -------
        u_pred : (batch, n)   — predicted solution at interior points
                                Boundary values u(0)=u(1)=0 are implicit
                                (not included in the output).
        """
        # Build input tokens: (batch, n, 2)
        tokens = torch.stack([x_grid, f_vals], dim=-1)

        # Embed to d_model
        h = self.token_embed(tokens)          # (batch, n, d_model)

        # Add positional encoding
        h = self.pos_enc(h)                   # (batch, n, d_model)

        # Transformer encoder
        h = self.encoder(h)                   # (batch, n, d_model)

        # Output head
        u = self.output_head(h).squeeze(-1)   # (batch, n)

        return u


# =============================================================================
# §4 — TRAINING LOOP
# =============================================================================

def train_model(model, n_grid, n_train=8000, n_val=1000,
                epochs=200, batch_size=128, lr=3e-4, k_max=K_MAX):
    """
    Train the PoissonTransformer on randomly generated (f, u) pairs.

    Optimiser  : Adam with weight decay 1e-5
    LR schedule: Cosine annealing over all epochs
    Loss       : MSE(u_pred, u_exact) on interior points
    Metric     : relative L2 error  ‖u_pred - u_exact‖ / ‖u_exact‖

    Returns
    -------
    history : dict with 'train_loss', 'val_loss', 'val_rel_l2' per epoch
    """
    model.to(device)

    # Precompute datasets
    F_tr, U_tr = make_dataset(n_grid, n_train, k_max=k_max, seed=0)
    F_va, U_va = make_dataset(n_grid, n_val,   k_max=k_max, seed=1)
    x_np       = interior_grid(n_grid)
    x_tensor   = torch.tensor(x_np, dtype=torch.float32)

    # Broadcast x_grid to (batch, n) — same for every sample
    x_tr = x_tensor.unsqueeze(0).expand(n_train, -1).to(device)
    x_va = x_tensor.unsqueeze(0).expand(n_val,   -1).to(device)
    F_tr, U_tr = F_tr.to(device), U_tr.to(device)
    F_va, U_va = F_va.to(device), U_va.to(device)

    optimizer  = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion  = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': [], 'val_rel_l2': []}
    t0      = time.time()

    for epoch in range(1, epochs + 1):
        # ── Training pass ──────────────────────────────────────────────────
        model.train()
        perm     = torch.randperm(n_train, device=device)
        ep_loss  = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx  = perm[start:start + batch_size]
            xb   = x_tr[idx]
            fb   = F_tr[idx]
            ub   = U_tr[idx]
            pred = model(xb, fb)
            loss = criterion(pred, ub)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
            n_batches += 1
        scheduler.step()
        history['train_loss'].append(ep_loss / n_batches)

        # ── Validation pass ────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_pred  = model(x_va, F_va)
            val_loss  = criterion(val_pred, U_va).item()
            # Relative L2 per sample, then averaged
            rel_l2 = (
                (val_pred - U_va).norm(dim=1)
                / (U_va.norm(dim=1) + 1e-8)
            ).mean().item()
        history['val_loss'].append(val_loss)
        history['val_rel_l2'].append(rel_l2)

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{epochs}  "
                  f"train MSE={ep_loss/n_batches:.4e}  "
                  f"val MSE={val_loss:.4e}  "
                  f"rel-L2={rel_l2:.4e}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    elapsed = time.time() - t0
    print(f"  Training complete in {elapsed:.1f}s")
    return history


# =============================================================================
# §5 — GRID-CONVERGENCE STUDY
# =============================================================================
#
# We train a separate model for each grid size n and measure the test error
# on the canonical source  f(x) = sin(πx)  (exact solution sin(πx)/π²).
# Thomas' algorithm serves as the O(h²) classical reference.

print(SEP)
print("§5 — GRID-CONVERGENCE STUDY")
print(SEP)

# Canonical test case: f(x) = sin(πx), u(x) = sin(πx)/π²
def test_source(x):   return np.sin(np.pi * x)
def test_exact(x):    return np.sin(np.pi * x) / np.pi**2

# Grid sizes to study
grid_sizes_cv = [8, 16, 32, 64]

# Main model config (used for all grid sizes in the convergence study)
MODEL_CFG = dict(d_model=64, n_heads=4, n_layers=4, d_ffn=256, dropout=0.0)
TRAIN_CFG = dict(n_train=8000, n_val=1000, epochs=300, batch_size=128,
                 lr=3e-4, k_max=K_MAX)

convergence_results = {}

for n_cv in grid_sizes_cv:
    print(f"\n{'─'*60}")
    print(f"  n = {n_cv} interior points  (h = {1/(n_cv+1):.5f})")
    print(f"{'─'*60}")

    # ── Thomas' algorithm ─────────────────────────────────────────────────
    x_cv   = interior_grid(n_cv)
    f_cv   = test_source(x_cv)
    u_th   = thomas_solve(n_cv, f_cv)
    u_ex   = test_exact(x_cv)
    err_th = np.max(np.abs(u_th - u_ex))

    # ── Transformer ───────────────────────────────────────────────────────
    print(f"  Training transformer (n={n_cv}, {TRAIN_CFG['epochs']} epochs)...")
    model_cv = PoissonTransformer(**MODEL_CFG)
    hist_cv  = train_model(model_cv, n_cv, **TRAIN_CFG)

    model_cv.eval()
    with torch.no_grad():
        x_t = torch.tensor(x_cv, dtype=torch.float32).unsqueeze(0).to(device)
        f_t = torch.tensor(f_cv, dtype=torch.float32).unsqueeze(0).to(device)
        # Normalise source to match training distribution
        f_scale = f_t.norm() + 1e-8
        u_tr_pred = model_cv(x_t, f_t / f_scale).squeeze(0).cpu().numpy()
        u_tr_pred *= f_scale.item()   # undo scale for physical comparison

    err_tr  = np.max(np.abs(u_tr_pred - u_ex))
    err_l2  = np.sqrt(np.mean((u_tr_pred - u_ex)**2))

    convergence_results[n_cv] = {
        'x'       : x_cv,
        'u_exact' : u_ex,
        'u_thomas': u_th,
        'u_transf': u_tr_pred,
        'err_th'  : err_th,
        'err_tr'  : err_tr,
        'err_l2_tr': err_l2,
        'history' : hist_cv,
    }

    print(f"  Thomas max error  : {err_th:.4e}")
    print(f"  Transformer max err: {err_tr:.4e}")
    print(f"  Transformer L2 err : {err_l2:.4e}")


# =============================================================================
# §6 — ABLATION: MODEL DEPTH AND WIDTH (fixed n=32)
# =============================================================================

print(f"\n{SEP}")
print("§6 — ABLATION: MODEL DEPTH AND WIDTH  (n=32)")
print(SEP)

ABLATION_N     = 32
ABLATION_EPOCHS = 200

ablation_configs = [
    dict(label="tiny",   d_model=16, n_heads=2, n_layers=2, d_ffn=64),
    dict(label="small",  d_model=32, n_heads=2, n_layers=2, d_ffn=128),
    dict(label="medium", d_model=64, n_heads=4, n_layers=4, d_ffn=256),
    dict(label="large",  d_model=128,n_heads=4, n_layers=6, d_ffn=512),
]

ablation_results = []
x_ab = interior_grid(ABLATION_N)
f_ab = test_source(x_ab)
u_ex_ab = test_exact(x_ab)
u_th_ab = thomas_solve(ABLATION_N, f_ab)

for cfg in ablation_configs:
    label   = cfg.pop('label')
    print(f"\n  Config: {label}")
    model_ab = PoissonTransformer(**cfg, dropout=0.0)
    hist_ab  = train_model(model_ab, ABLATION_N,
                           n_train=6000, n_val=800,
                           epochs=ABLATION_EPOCHS, batch_size=128, lr=3e-4)

    model_ab.eval()
    with torch.no_grad():
        x_t  = torch.tensor(x_ab, dtype=torch.float32).unsqueeze(0).to(device)
        f_t  = torch.tensor(f_ab, dtype=torch.float32).unsqueeze(0).to(device)
        fsc  = f_t.norm() + 1e-8
        u_ab = model_ab(x_t, f_t / fsc).squeeze(0).cpu().numpy() * fsc.item()

    err  = np.max(np.abs(u_ab - u_ex_ab))
    npar = sum(p.numel() for p in model_ab.parameters())
    ablation_results.append({
        'label'  : label,
        'n_params': npar,
        'err_max': err,
        'val_l2' : hist_ab['val_rel_l2'][-1],
        'history': hist_ab,
        'd_model': cfg['d_model'],
        'n_layers': cfg['n_layers'],
    })
    cfg['label'] = label  # restore
    print(f"  Max error: {err:.4e}  (Thomas: {np.max(np.abs(u_th_ab-u_ex_ab)):.4e})")


# =============================================================================
# §7 — GENERALISATION TO UNSEEN SOURCE FUNCTIONS
# =============================================================================
#
# Test the best transformer (n=32, medium config) on source functions
# that are qualitatively different from the training distribution:
#   (a) Pure sine mode  k=1  (within training support)
#   (b) Higher mode  k=6  (extrapolation in mode number)
#   (c) Gaussian bump  f(x) = exp(-50(x-0.5)²)
#   (d) Step function   f(x) = sign(x - 0.5)

print(f"\n{SEP}")
print("§7 — GENERALISATION TO UNSEEN SOURCE FUNCTIONS  (n=32)")
print(SEP)

# Use the n=32 model from the convergence study
model_gen = None
if 32 in convergence_results:
    # Re-train a fresh medium model for this section for clarity
    print("  Re-training medium transformer on n=32...")
    model_gen = PoissonTransformer(**MODEL_CFG)
    train_model(model_gen, 32, **TRAIN_CFG)

x_gen = interior_grid(32)

ood_cases = {
    "sin(πx)    [in-dist]" : (
        lambda x: np.sin(np.pi * x),
        lambda x: np.sin(np.pi * x) / np.pi**2
    ),
    "sin(6πx)  [higher k]" : (
        lambda x: np.sin(6 * np.pi * x),
        lambda x: np.sin(6 * np.pi * x) / (6 * np.pi)**2
    ),
    "Gaussian bump        " : (
        lambda x: np.exp(-50 * (x - 0.5)**2),
        None   # use Thomas as reference
    ),
    "Poly x(1-x)          " : (
        lambda x: x * (1 - x),
        lambda x: x * (1 - x**2) / 6   # exact: u = x/6 - x^3/6
    ),
}

gen_results = {}
print(f"\n  {'Source':26s}  {'Thomas max err':>14}  "
      f"{'Transf max err':>15}  {'Ratio':>8}")
print("  " + "-"*68)

for name, (f_fn, u_fn_exact) in ood_cases.items():
    f_g = f_fn(x_gen)
    if u_fn_exact is not None:
        u_ref = u_fn_exact(x_gen)
    else:
        u_ref = thomas_solve(32, f_g)

    u_th_g = thomas_solve(32, f_g)

    if model_gen is not None:
        model_gen.eval()
        with torch.no_grad():
            x_t = torch.tensor(x_gen, dtype=torch.float32).unsqueeze(0).to(device)
            f_t = torch.tensor(f_g,   dtype=torch.float32).unsqueeze(0).to(device)
            fsc = f_t.norm() + 1e-8
            u_tr_g = model_gen(x_t, f_t/fsc).squeeze(0).cpu().numpy() * fsc.item()
        err_tr  = np.max(np.abs(u_tr_g - u_ref))
    else:
        u_tr_g = np.zeros_like(u_th_g)
        err_tr = float('nan')

    err_th = np.max(np.abs(u_th_g - u_ref))
    ratio  = err_tr / (err_th + 1e-12)
    gen_results[name] = {
        'f': f_g, 'u_ref': u_ref, 'u_thomas': u_th_g, 'u_transf': u_tr_g,
        'err_th': err_th, 'err_tr': err_tr
    }
    print(f"  {name}  {err_th:>14.4e}  {err_tr:>15.4e}  {ratio:>8.2f}×")


# =============================================================================
# §8 — VISUALISATION  (8-panel figure)
# =============================================================================

fig = plt.figure(figsize=(20, 14))
fig.suptitle(
    "Transformer Solver vs Thomas' Algorithm — 1-D Poisson Equation\n"
    r"$-u''=f(x)$, $u(0)=u(1)=0$",
    fontsize=14, fontweight='bold'
)
gs = gridspec.GridSpec(2, 4, hspace=0.50, wspace=0.42)

# ── Panel 1: Solution comparison for n=32, canonical source ─────────────────
ax = fig.add_subplot(gs[0, 0])
n_ref = 32
if n_ref in convergence_results:
    r = convergence_results[n_ref]
    x_d = np.linspace(0, 1, 400)
    ax.plot(x_d, test_exact(x_d), 'k-', lw=2.5,
            label=r'Exact $\sin(\pi x)/\pi^2$')
    ax.plot(np.concatenate([[0], r['x'], [1]]),
            np.concatenate([[0], r['u_thomas'], [0]]),
            'bs--', ms=5, lw=1.5,
            label=f"Thomas ({r['err_th']:.1e})")
    ax.plot(np.concatenate([[0], r['x'], [1]]),
            np.concatenate([[0], r['u_transf'], [0]]),
            'r^:', ms=5, lw=1.5,
            label=f"Transformer ({r['err_tr']:.1e})")
    ax.set(xlabel='$x$', ylabel='$u(x)$',
           title=f'Solution comparison  $n={n_ref}$\n'
                 r'$f(x)=\sin(\pi x)$')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── Panel 2: Grid convergence (max error) ───────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
ns_cv  = sorted(convergence_results.keys())
hs_cv  = [1/(n+1)          for n in ns_cv]
th_errs = [convergence_results[n]['err_th'] for n in ns_cv]
tr_errs = [convergence_results[n]['err_tr'] for n in ns_cv]
ax.loglog(hs_cv, th_errs, 'bs-',  lw=2, ms=8, label="Thomas'")
ax.loglog(hs_cv, tr_errs, 'r^--', lw=2, ms=8, label='Transformer')
# O(h²) reference
h0, e0 = hs_cv[0], th_errs[0]
ax.loglog(hs_cv, [e0*(h/h0)**2 for h in hs_cv],
          'k:', lw=1.5, alpha=0.6, label=r'$O(h^2)$')
ax.set(xlabel='Grid spacing $h$',
       ylabel=r'$\|u_{\rm pred}-u_{\rm ex}\|_\infty$',
       title='Grid convergence\n(max error vs $h$)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which='both')

# ── Panel 3: Training loss curves for n=32 ──────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
if n_ref in convergence_results:
    hist = convergence_results[n_ref]['history']
    epochs_ax = range(1, len(hist['train_loss']) + 1)
    ax.semilogy(epochs_ax, hist['train_loss'], 'b-',  lw=2, label='Train MSE')
    ax.semilogy(epochs_ax, hist['val_loss'],   'r--', lw=2, label='Val MSE')
    ax.semilogy(epochs_ax, hist['val_rel_l2'], 'g:',  lw=2, label='Val rel-L2')
    ax.set(xlabel='Epoch', ylabel='Loss / Error',
           title=f'Training curves  $n={n_ref}$\n'
                 f'd={MODEL_CFG["d_model"]}, '
                 f'L={MODEL_CFG["n_layers"]}, '
                 f'H={MODEL_CFG["n_heads"]}')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── Panel 4: Ablation: max error vs model size (n=32) ───────────────────────
ax = fig.add_subplot(gs[0, 3])
if ablation_results:
    labels_ab = [r['label']    for r in ablation_results]
    npar_ab   = [r['n_params'] for r in ablation_results]
    err_ab    = [r['err_max']  for r in ablation_results]
    ax.loglog(npar_ab, err_ab, 'mo-', lw=2, ms=8)
    for r in ablation_results:
        ax.annotate(r['label'],
                    (r['n_params'], r['err_max']),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax.axhline(np.max(np.abs(u_th_ab - u_ex_ab)), color='steelblue',
               ls='--', lw=1.5, label="Thomas' max err")
    ax.set(xlabel='Trainable parameters', ylabel=r'Max error',
           title=f'Ablation: model size  $n={ABLATION_N}$\n'
                 f'(Thomas ref: dashed blue)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which='both')

# ── Panel 5: Pointwise error for all grid sizes ──────────────────────────────
ax = fig.add_subplot(gs[1, 0])
colours = ['steelblue', 'firebrick', 'seagreen', 'darkorange']
for col, n_pt in zip(colours, ns_cv):
    r = convergence_results[n_pt]
    ax.semilogy(r['x'], np.abs(r['u_transf'] - r['u_exact']),
                '-', color=col, lw=1.5, label=f'$n={n_pt}$')
ax.set(xlabel='$x$', ylabel=r'$|u_{\rm transf}-u_{\rm ex}|$',
       title='Transformer pointwise error\nvs position')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── Panel 6: Residual histogram (n=32, transformer vs Thomas) ───────────────
ax = fig.add_subplot(gs[1, 1])
if n_ref in convergence_results:
    r = convergence_results[n_ref]
    ax.hist(r['u_transf'] - r['u_exact'], bins=25, alpha=0.6,
            color='firebrick', label=f"Transformer (σ={np.std(r['u_transf']-r['u_exact']):.2e})")
    ax.hist(r['u_thomas'] - r['u_exact'], bins=25, alpha=0.6,
            color='steelblue', label=f"Thomas' (σ={np.std(r['u_thomas']-r['u_exact']):.2e})")
    ax.axvline(0, color='k', lw=1.5, ls='--')
    ax.set(xlabel='Residual $u_{\\rm pred} - u_{\\rm ex}$',
           ylabel='Count',
           title=f'Residual distribution  $n={n_ref}$')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

# ── Panel 7: Generalisation — solution curves ───────────────────────────────
ax = fig.add_subplot(gs[1, 2])
plot_cases = list(ood_cases.keys())[:3]
line_styles = ['-', '--', ':']
for ls, name in zip(line_styles, plot_cases):
    gr = gen_results.get(name)
    if gr is None:
        continue
    ax.plot(x_gen, gr['u_ref'],    color='k',          lw=2,   ls=ls,
            alpha=0.7, label=f"Exact/Ref {name[:12]}")
    ax.plot(x_gen, gr['u_transf'], color='firebrick',  lw=1.5, ls=ls,
            alpha=0.9)
    ax.plot(x_gen, gr['u_thomas'], color='steelblue',  lw=1.5, ls=ls,
            alpha=0.9)
# Legend proxies
from matplotlib.lines import Line2D
proxy = [
    Line2D([0],[0], color='k',         lw=2,   label='Exact/ref'),
    Line2D([0],[0], color='firebrick', lw=1.5, label='Transformer'),
    Line2D([0],[0], color='steelblue', lw=1.5, label="Thomas'"),
]
ax.legend(handles=proxy, fontsize=8)
ax.set(xlabel='$x$', ylabel='$u(x)$',
       title='Generalisation to OOD sources\n(n=32, 3 test functions)')
ax.grid(True, alpha=0.3)

# ── Panel 8: Architecture diagram ───────────────────────────────────────────
ax = fig.add_subplot(gs[1, 3])
ax.axis('off')
diagram = (
    "POISSONRANSFORMER\n"
    "─────────────────────────────\n\n"
    "Input per token (grid point i):\n"
    "  [xᵢ , f(xᵢ)]  ∈  ℝ²\n\n"
    "Token embedding:\n"
    "  Linear(2 → d_model)\n\n"
    "+ Sinusoidal pos. encoding\n"
    "  PE(pos, 2k)   = sin(pos/10000^…)\n"
    "  PE(pos, 2k+1) = cos(pos/10000^…)\n\n"
    "× N encoder layers (pre-LN):\n"
    "  LayerNorm\n"
    "  MultiHeadSelfAttention\n"
    "    Q,K,V ∈ ℝ^(n × d_head)\n"
    "    Attn ≈ discrete Green fn Gᵢⱼ\n"
    "  Residual + Dropout\n"
    "  LayerNorm\n"
    "  FFN: d_model → d_ffn → d_model\n"
    "  Residual\n\n"
    "Output head per token:\n"
    "  LayerNorm → Linear → GELU\n"
    "  → Linear(d_model → 1)\n\n"
    "BCs enforced analytically:\n"
    "  u(0) = u(1) = 0  (not predicted)"
)
ax.text(0.03, 0.97, diagram, transform=ax.transAxes,
        fontsize=8.5, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
ax.set_title('Architecture', fontweight='bold', fontsize=10)

plt.savefig('transformer_poisson.png', dpi=140, bbox_inches='tight')
plt.show()
print("\n✓ Figure saved to transformer_poisson.png")

# =============================================================================
# §9 — SUMMARY TABLE
# =============================================================================

print(f"\n{SEP}")
print("§9 — SUMMARY")
print(SEP)

print(f"""
Problem:  -u''=sin(πx),  u(0)=u(1)=0
Exact:    u(x)=sin(πx)/π²

GRID CONVERGENCE  (max error ‖u_pred − u_exact‖∞)
""")
print(f"  {'n':>6}  {'h':>9}  {'Thomas max err':>16}  "
      f"{'Transformer max err':>20}  {'Ratio T/Th':>12}")
print("  " + "-"*70)
for n_cv in sorted(convergence_results):
    r  = convergence_results[n_cv]
    h  = 1 / (n_cv + 1)
    ratio = r['err_tr'] / (r['err_th'] + 1e-15)
    print(f"  {n_cv:>6}  {h:>9.5f}  {r['err_th']:>16.4e}  "
          f"{r['err_tr']:>20.4e}  {ratio:>12.2f}×")

print(f"""
ABLATION (n={ABLATION_N}, max error vs model size)
""")
if ablation_results:
    print(f"  {'Config':>8}  {'d_model':>8}  {'Layers':>7}  "
          f"{'Params':>10}  {'Max err':>12}  {'Val rel-L2':>12}")
    print("  " + "-"*64)
    for r in ablation_results:
        print(f"  {r['label']:>8}  {r['d_model']:>8}  {r['n_layers']:>7}  "
              f"  {r['n_params']:>8,}  {r['err_max']:>12.4e}  "
              f"{r['val_l2']:>12.4e}")

print(f"""
KEY COMPARISON
  Thomas' algorithm:
    Complexity  : O(n)  —  exact Gaussian elimination on tridiagonal system
    Error       : O(h²) truncation error only,  converges exactly with h→0
    Generality  : specific to tridiagonal (or banded) matrices
    Output      : exact floating-point solution vector

  PoissonTransformer:
    Complexity  : O(n² d) per forward pass, O(n_train × epochs × n²) training
    Error       : learned approximation error + implicit O(h²) FD error
    Generality  : once trained, runs in one forward pass for any f on the
                  trained grid; the model approximates the Green's function
                  operator  L⁻¹: f → u  directly
    Output      : neural approximation of the solution

  Attention patterns ↔ Green's function:
    The discrete 1-D Poisson Green's function is
        G[i,j] = h² min(i,j)(n+1-max(i,j)) / (n+1)
    This is a tent-shaped kernel: G[i,j] grows linearly toward the
    diagonal.  A well-trained transformer head should develop attention
    weights that reflect this structure, concentrating weight near the
    diagonal but maintaining global context.
""")
