#!/usr/bin/env python3
"""
Feed-Forward Neural Network vs Transformer
==========================================
A self-contained PyTorch script that trains both architectures on the same
synthetic regression task and highlights the key architectural differences.

Task
----
Predict sin(x1 + x2) from the pair (x1, x2).

  • For the FFNN this is a flat 2-D input — no ordering information.
  • For the Transformer the same pair is treated as a *sequence* of two tokens
    so that the model must attend across positions to produce its prediction.

This contrast exposes the central difference:

  FFNN                    Transformer
  ──────────────────────  ──────────────────────────────────────────────
  Processes all inputs    Processes inputs as an ordered sequence of tokens
  simultaneously via      and uses self-attention to let each token "look at"
  weight matrices.        every other token before producing output.

  Parameters do not       Parameters (Query/Key/Value projections) are shared
  depend on position.     across positions; order is injected via positional
                          encodings.

  Global receptive        Global receptive field by design: O(n²) attention
  field only if the       makes every output depend on every input in one pass.
  network is deep.

  No explicit notion      Attention weights are interpretable: we can see which
  of token interaction.   input positions the model focuses on.
"""

import math
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


# =============================================================================
# 1.  DATASET
# =============================================================================
def make_dataset(n_samples: int = 8000):
    """
    Generate (x1, x2) pairs uniformly in [-pi, pi]^2 and targets sin(x1+x2).

    Returns
    -------
    X : (n, 2)  float32 tensor  — raw inputs
    y : (n, 1)  float32 tensor  — regression targets in [-1, 1]
    """
    X = (torch.rand(n_samples, 2) * 2 - 1) * math.pi   # uniform in [-pi, pi]
    y = torch.sin(X[:, 0] + X[:, 1]).unsqueeze(1)
    return X, y


X_all, y_all = make_dataset(8000)

# 70 / 15 / 15 split
n_train = int(0.70 * len(X_all))
n_val   = int(0.15 * len(X_all))

X_train, y_train = X_all[:n_train].to(device),         y_all[:n_train].to(device)
X_val,   y_val   = X_all[n_train:n_train+n_val].to(device), y_all[n_train:n_train+n_val].to(device)
X_test,  y_test  = X_all[n_train+n_val:].to(device),   y_all[n_train+n_val:].to(device)

print(f"Dataset: {len(X_train)} train | {len(X_val)} val | {len(X_test)} test")
print(f"Input shape : {X_train.shape}   Target range : [{y_train.min():.2f}, {y_train.max():.2f}]\n")


# =============================================================================
# 2.  MODEL DEFINITIONS
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 2a. Feed-Forward Neural Network (FFNN)
# ─────────────────────────────────────────────────────────────────────────────
class FFNN(nn.Module):
    """
    Standard multi-layer perceptron.

    Architecture
    ────────────
    Input (2) → Linear → ReLU → Linear → ReLU → Linear → Output (1)
                 ↑                ↑                ↑
               hidden           hidden           hidden

    Key properties
    ──────────────
    • Every neuron sees the *entire* input at once — there is no notion of
      sequence order or token identity.
    • The transformation is y = f(Wx + b): a static, position-agnostic map.
    • Depth adds non-linearity but NOT the ability to route information
      between input positions in a learned, input-dependent way.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64,
                 n_layers: int = 3, output_dim: int = 1):
        super().__init__()
        layers = []
        dims   = [input_dim] + [hidden_dim] * n_layers + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:        # no activation on the output layer
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 2)
        return self.net(x)               # (batch, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 2b. Transformer (encoder-only, for regression)
# ─────────────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """
    Classic sinusoidal positional encoding (Vaswani et al., 2017).

    Each position p gets a unique vector:
        PE(p, 2i)   = sin(p / 10000^(2i/d_model))
        PE(p, 2i+1) = cos(p / 10000^(2i/d_model))

    Why is this needed?
    ──────────────────
    Self-attention is *permutation-equivariant*: without positional encodings
    swapping token order would produce identical outputs.  Adding PE breaks
    this symmetry and lets the model distinguish position 0 from position 1.

    In our task this means the model can tell x1 apart from x2 even though
    the self-attention mechanism itself has no built-in notion of order.
    """
    def __init__(self, d_model: int, max_len: int = 16):
        super().__init__()
        pe   = torch.zeros(max_len, d_model)
        pos  = torch.arange(max_len).unsqueeze(1).float()
        div  = torch.exp(torch.arange(0, d_model, 2).float()
                         * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class TransformerRegressor(nn.Module):
    """
    Encoder-only Transformer for scalar regression.

    Architecture
    ────────────
    Input (batch, 2)
        ↓   embed each scalar token to d_model dimensions
    Token embeddings (batch, 2, d_model)
        ↓   add sinusoidal positional encoding
    PE-enriched embeddings
        ↓   N × TransformerEncoderLayer (MultiHeadAttention + FFN + LayerNorm)
    Contextualised representations (batch, 2, d_model)
        ↓   mean-pool across the sequence dimension
    Pooled representation (batch, d_model)
        ↓   linear projection to output
    Prediction (batch, 1)

    Key properties
    ──────────────
    • Self-attention: each token computes Query (Q), Key (K), Value (V)
      projections and attends to all other tokens with weights
          Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
      This is an *input-dependent* routing mechanism — the FFNN has no
      equivalent.

    • Multi-head attention runs h independent attention functions in parallel,
      letting the model attend to different aspects simultaneously.

    • The feed-forward sublayer inside each encoder layer is itself a small
      FFNN applied position-wise — so a Transformer contains FFNNs as
      components, but surrounds them with attention.

    • Layer normalisation (pre-norm here) stabilises training.
    """
    def __init__(self, seq_len: int = 2, d_model: int = 64,
                 n_heads: int = 2, n_layers: int = 2,
                 dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()

        # Each input scalar becomes a d_model-dimensional token embedding.
        # We use a shared linear projection; a lookup table (nn.Embedding)
        # would be used for discrete tokens (e.g. words).
        self.token_embed = nn.Linear(1, d_model)

        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = n_heads,
            dim_feedforward= dim_feedforward,
            dropout        = dropout,
            batch_first    = True,   # (batch, seq, d_model) convention
            norm_first     = True,   # pre-norm (more stable than post-norm)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor,
                return_attn: bool = False) -> torch.Tensor:
        """
        Parameters
        ----------
        x           : (batch, 2)  — the two scalar inputs
        return_attn : if True, also return attention weights for visualisation

        Returns
        -------
        out  : (batch, 1)
        attn : list of (batch, n_heads, seq, seq) — only when return_attn=True
        """
        # Step 1: reshape scalars to tokens and embed
        #   (batch, 2) -> (batch, 2, 1) -> (batch, 2, d_model)
        tokens = self.token_embed(x.unsqueeze(-1))

        # Step 2: add positional encoding
        tokens = self.pos_enc(tokens)

        # Step 3: self-attention layers
        if return_attn:
            attn_weights = []
            h = tokens
            for layer in self.encoder.layers:
                # Run multi-head attention manually to capture weights
                h_norm = layer.norm1(h)
                attn_out, weights = layer.self_attn(
                    h_norm, h_norm, h_norm, need_weights=True,
                    average_attn_weights=False)
                attn_weights.append(weights.detach().cpu())
                h = h + layer.dropout1(attn_out)
                h = h + layer.dropout2(
                    layer.linear2(layer.dropout(
                        layer.activation(layer.linear1(layer.norm2(h))))))
            encoded = h
        else:
            encoded      = self.encoder(tokens)    # (batch, 2, d_model)
            attn_weights = None

        # Step 4: mean-pool and project to scalar
        out = self.output_head(encoded.mean(dim=1))   # (batch, 1)

        if return_attn:
            return out, attn_weights
        return out


# Parameter counts
ffnn_model        = FFNN(hidden_dim=64, n_layers=3).to(device)
transformer_model = TransformerRegressor(d_model=64, n_heads=2,
                                         n_layers=2, dim_feedforward=128).to(device)

ffnn_params  = sum(p.numel() for p in ffnn_model.parameters() if p.requires_grad)
trans_params = sum(p.numel() for p in transformer_model.parameters() if p.requires_grad)

print("=" * 60)
print("MODEL ARCHITECTURES")
print("=" * 60)
print(f"\nFFNN:\n{ffnn_model}\n  Trainable parameters: {ffnn_params:,}")
print(f"\nTransformer:\n{transformer_model}\n  Trainable parameters: {trans_params:,}\n")


# =============================================================================
# 3.  TRAINING
# =============================================================================
def train(model, X_tr, y_tr, X_val, y_val,
          epochs: int = 300, lr: float = 1e-3,
          batch_size: int = 256, label: str = "Model"):
    """
    Minibatch Adam training with MSE loss.

    Returns
    -------
    train_losses, val_losses : lists of per-epoch mean MSE
    elapsed                  : wall-clock training time in seconds
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    n = X_tr.shape[0]
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(n, device=device)
        ep_loss = 0.0
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            xb, yb = X_tr[batch_idx], y_tr[batch_idx]
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(xb)
        scheduler.step()
        train_losses.append(ep_loss / n)

        model.eval()
        with torch.no_grad():
            val_losses.append(criterion(model(X_val), y_val).item())

        if epoch % 50 == 0:
            print(f"  [{label}] ep {epoch:3d}/{epochs}  "
                  f"train MSE={train_losses[-1]:.5f}  "
                  f"val MSE={val_losses[-1]:.5f}")

    return train_losses, val_losses, time.time() - t0


print("=" * 60)
print("TRAINING: FFNN")
print("=" * 60)
ffnn_tr_losses, ffnn_val_losses, ffnn_time = train(
    ffnn_model, X_train, y_train, X_val, y_val,
    epochs=300, lr=1e-3, label="FFNN")

print()
print("=" * 60)
print("TRAINING: TRANSFORMER")
print("=" * 60)
trans_tr_losses, trans_val_losses, trans_time = train(
    transformer_model, X_train, y_train, X_val, y_val,
    epochs=300, lr=1e-3, label="Transformer")


# =============================================================================
# 4.  TEST EVALUATION
# =============================================================================
criterion = nn.MSELoss()

ffnn_model.eval()
transformer_model.eval()
with torch.no_grad():
    ffnn_test_mse  = criterion(ffnn_model(X_test),        y_test).item()
    trans_test_mse = criterion(transformer_model(X_test),  y_test).item()
    ffnn_test_mae  = (ffnn_model(X_test) - y_test).abs().mean().item()
    trans_test_mae = (transformer_model(X_test) - y_test).abs().mean().item()

print()
print("=" * 60)
print("TEST SET RESULTS")
print("=" * 60)
print(f"  {'Model':<20s} {'MSE':>10s} {'MAE':>10s} {'Params':>10s} {'Time (s)':>10s}")
print("  " + "-" * 64)
print(f"  {'FFNN':<20s} {ffnn_test_mse:>10.5f} {ffnn_test_mae:>10.5f} "
      f"{ffnn_params:>10,} {ffnn_time:>10.1f}")
print(f"  {'Transformer':<20s} {trans_test_mse:>10.5f} {trans_test_mae:>10.5f} "
      f"{trans_params:>10,} {trans_time:>10.1f}")


# =============================================================================
# 5.  ATTENTION WEIGHT INSPECTION
# =============================================================================
# Run a few examples through the Transformer and capture the attention weights.
# These weights show which input token each output position attends to most.
# The FFNN has no equivalent — there is simply no attention mechanism.

transformer_model.eval()
with torch.no_grad():
    sample_x = X_test[:8]
    _, attn_all_layers = transformer_model(sample_x, return_attn=True)

# attn_all_layers[layer]: (batch, n_heads, seq=2, seq=2)
# Average over heads and examples for a cleaner summary
print()
print("=" * 60)
print("ATTENTION WEIGHTS (Transformer, layer 1, averaged over heads & examples)")
print("=" * 60)
print("Rows = attending token  |  Cols = attended-to token")
print("  (token 0 = x1, token 1 = x2)")
print()
layer0_attn = attn_all_layers[0].mean(dim=(0, 1))   # (2, 2)
print(f"                token_0   token_1")
for r, row in enumerate(layer0_attn):
    print(f"  token_{r} →    {row[0]:.3f}     {row[1]:.3f}")
print()
print("  Interpretation: each row shows how much a given output token")
print("  attends to each input token when building its representation.")


# =============================================================================
# 6.  KEY DIFFERENCES — PRINTED SUMMARY
# =============================================================================
print()
print("=" * 60)
print("KEY ARCHITECTURAL DIFFERENCES")
print("=" * 60)
diffs = [
    ("Input processing",
     "All inputs combined at once\nvia a single weight matrix.",
     "Each input is a separate token;\ninformation flows via attention."),
    ("Attention mechanism",
     "None — no explicit routing\nbetween input positions.",
     "Multi-head self-attention:\neach token attends to all others\nwith learned, input-dependent weights."),
    ("Position awareness",
     "No notion of input order;\nswapping (x1,x2)→(x2,x1) changes\nthe output only because W is asymmetric.",
     "Permutation-equivariant by design;\norder injected via positional encodings."),
    ("Interpretability",
     "Black box — no natural way to\nsee which input drives the output.",
     "Attention weights are inspectable\nand show token-to-token relevance."),
    ("Scaling behaviour",
     "Parameters grow as O(width × depth);\ngood for fixed-size tabular data.",
     "Attention cost is O(seq_len²) but\nparameters scale with d_model, making\ntransformers powerful for long sequences."),
    ("Inductive bias",
     "Strong: assumes flat, fixed-size\ninput with no sequential structure.",
     "Weak: minimal assumptions about\ntoken relationships — learned from data."),
]

for name, ffnn_desc, trans_desc in diffs:
    print(f"\n  {name}")
    print(f"    FFNN        : {ffnn_desc.replace(chr(10), chr(10)+'                ')}")
    print(f"    Transformer : {trans_desc.replace(chr(10), chr(10)+'                ')}")


# =============================================================================
# 7.  VISUALISATION
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Feed-Forward Neural Network vs Transformer\n"
             "Task: predict sin(x₁ + x₂)", fontsize=14, fontweight="bold")

# ── Panel 1: Training loss curves ──────────────────────────────────────────
ax = axes[0, 0]
epochs_range = range(1, len(ffnn_tr_losses) + 1)
ax.semilogy(epochs_range, ffnn_tr_losses,  "b-",  lw=2, label="FFNN train")
ax.semilogy(epochs_range, ffnn_val_losses, "b--", lw=1.5, label="FFNN val", alpha=0.8)
ax.semilogy(epochs_range, trans_tr_losses, "r-",  lw=2, label="Transformer train")
ax.semilogy(epochs_range, trans_val_losses,"r--", lw=1.5, label="Transformer val", alpha=0.8)
ax.set(xlabel="Epoch", ylabel="MSE Loss (log scale)",
       title="Training & Validation Loss")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# ── Panel 2: Predictions vs ground truth ───────────────────────────────────
ax = axes[0, 1]
n_plot = 200
x_cpu  = X_test[:n_plot].cpu().numpy()
y_cpu  = y_test[:n_plot].cpu().numpy().flatten()
ffnn_model.eval(); transformer_model.eval()
with torch.no_grad():
    ffnn_pred  = ffnn_model(X_test[:n_plot]).cpu().numpy().flatten()
    trans_pred = transformer_model(X_test[:n_plot]).cpu().numpy().flatten()

sort_idx = np.argsort(y_cpu)
ax.plot(y_cpu[sort_idx], "k-",  lw=1.5, label="Ground truth", alpha=0.6)
ax.plot(ffnn_pred[sort_idx],  "b--", lw=1.5, label=f"FFNN  (MAE={ffnn_test_mae:.4f})")
ax.plot(trans_pred[sort_idx], "r--", lw=1.5, label=f"Transf. (MAE={trans_test_mae:.4f})")
ax.set(xlabel="Sample (sorted by target)", ylabel="sin(x₁ + x₂)",
       title="Predictions vs Ground Truth (test set)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# ── Panel 3: Residuals ─────────────────────────────────────────────────────
ax = axes[0, 2]
ax.hist(y_cpu - ffnn_pred,  bins=40, alpha=0.6, color="blue",
        label=f"FFNN  MSE={ffnn_test_mse:.4f}")
ax.hist(y_cpu - trans_pred, bins=40, alpha=0.6, color="red",
        label=f"Transf. MSE={trans_test_mse:.4f}")
ax.axvline(0, color="k", lw=1.5, ls="--")
ax.set(xlabel="Residual (true − predicted)", ylabel="Count",
       title="Prediction Error Distribution (test set)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

# ── Panel 4: Attention heatmap ─────────────────────────────────────────────
ax = axes[1, 0]
# Show attention weights for each layer averaged over heads and examples
n_layers = len(attn_all_layers)
all_attn  = np.stack([a.mean(dim=(0, 1)).numpy()
                       for a in attn_all_layers])   # (n_layers, 2, 2)
# Tile into a (2*n_layers, 2) display matrix
display = np.vstack(all_attn)
im = ax.imshow(display, cmap="Blues", vmin=0, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, label="Attention weight")
# Tick labels
ax.set_xticks([0, 1]); ax.set_xticklabels(["Token 0\n(x₁)", "Token 1\n(x₂)"])
yticks = []; ylabels = []
for layer_i in range(n_layers):
    for tok in range(2):
        yticks.append(layer_i * 2 + tok)
        ylabels.append(f"L{layer_i+1} tok{tok}")
ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=8)
# Draw horizontal lines between layers
for layer_i in range(1, n_layers):
    ax.axhline(layer_i * 2 - 0.5, color="white", lw=2)
for i in range(display.shape[0]):
    for j in range(display.shape[1]):
        ax.text(j, i, f"{display[i, j]:.2f}", ha="center", va="center",
                fontsize=9, color="black" if display[i, j] < 0.6 else "white")
ax.set(xlabel="Attended-to token", title="Self-Attention Weights\n"
       "(Transformer — averaged over heads & examples)")

# ── Panel 5: Architecture diagram comparison ───────────────────────────────
ax = axes[1, 1]
ax.axis("off")
diagram_text = (
    "FEED-FORWARD NETWORK\n"
    "─────────────────────\n"
    "  x₁ ──┐\n"
    "        ├─► [Linear→ReLU]×3 ─► ŷ\n"
    "  x₂ ──┘\n\n"
    "  • All inputs concatenated\n"
    "  • One-directional flow\n"
    "  • Fixed, position-agnostic weights\n"
    "  • No interaction between inputs\n"
    "    except through shared weights\n\n\n"
    "TRANSFORMER (encoder-only)\n"
    "──────────────────────────\n"
    "  x₁ ─► [embed] ─► tok₀\n"
    "  x₂ ─► [embed] ─► tok₁\n\n"
    "  tok₀ ←──── Attention ────► tok₀'\n"
    "  tok₁ ←──── Attention ────► tok₁'\n\n"
    "  [mean pool] ─► [Linear] ─► ŷ\n\n"
    "  • Inputs are separate tokens\n"
    "  • Attention = input-dependent\n"
    "    token interaction\n"
    "  • Positional encodings add order"
)
ax.text(0.05, 0.97, diagram_text, transform=ax.transAxes,
        fontsize=9.5, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
ax.set_title("Architecture Comparison", fontweight="bold")

# ── Panel 6: Prediction surface (2-D input space) ──────────────────────────
ax = axes[1, 2]
grid_pts = 60
g = np.linspace(-math.pi, math.pi, grid_pts)
gx1, gx2 = np.meshgrid(g, g)
grid_in = torch.tensor(
    np.stack([gx1.ravel(), gx2.ravel()], axis=1), dtype=torch.float32
).to(device)

with torch.no_grad():
    ffnn_surf  = ffnn_model(grid_in).cpu().numpy().reshape(grid_pts, grid_pts)
    trans_surf = transformer_model(grid_in).cpu().numpy().reshape(grid_pts, grid_pts)
    true_surf  = np.sin(gx1 + gx2)

# Show the residual surface: Transformer error minus FFNN error
diff = np.abs(trans_surf - true_surf) - np.abs(ffnn_surf - true_surf)
im2  = ax.contourf(gx1, gx2, diff, levels=30, cmap="RdBu_r")
plt.colorbar(im2, ax=ax, label="|Transf. err| − |FFNN err|  (blue = Transf. better)")
ax.set(xlabel="x₁", ylabel="x₂",
       title="|Transformer error| − |FFNN error|\n"
             "Blue = Transformer better, Red = FFNN better")
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("ffnn_vs_transformer.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n✓ Figure saved to ffnn_vs_transformer.png")
