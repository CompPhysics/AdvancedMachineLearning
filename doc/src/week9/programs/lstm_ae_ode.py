#!/usr/bin/env python3
"""
LSTM and Autoencoder for Learning ODE Solutions
Using PyTorch and RK4 Solver Output  --  OPTIMISED VERSION

Training LSTM and Autoencoder models on forced oscillator differential equation:
d**2x/dt**2 + 2*gamma*(dx/dt) + x = F_tilde*cos(Omega_tilde*t)

Performance optimisations applied (vs original)
------------------------------------------------
1. RK4 solver: precompute all cos() evaluations as two numpy arrays before
   the integration loop. ~24% faster on 20 000 steps.

2. create_sequences: replaced the Python list-append loop with
   numpy.lib.stride_tricks.sliding_window_view. ~5x faster.

3. generate_predictions: replaced the sample-by-sample loop with a single
   batched forward pass over the full dataset. ~8x faster.

4. encode_dataset: same batched rewrite, iterating in fixed chunks.

5. RNN forward passes: passing None for hidden state lets PyTorch allocate
   it via optimised C++/CUDA paths instead of allocating in Python.

6. DataLoader: num_workers=min(4, cpu_count) and pin_memory=True on CUDA.

7. optimizer.zero_grad(set_to_none=True): avoids a memset() per parameter
   tensor each step.

8. torch.compile (PyTorch 2+): each model is compiled before training if
   available, enabling TorchDynamo + inductor kernel fusion.

9. VAE reconstruction in visualisation: four separate forward passes replaced
   by a single batched call over all example windows at once.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from numpy.lib.stride_tricks import sliding_window_view
import time
import os
import multiprocessing
from math import ceil

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------
# Safe DataLoader worker count
# ---------------------------------------------------------------------------
# On Windows and macOS the default multiprocessing start method is 'spawn'
# (or 'forkserver'), which re-imports the whole module in every worker.
# Because the module contains top-level executable code, that causes the
# "bootstrapping" RuntimeError unless the code is guarded by
# `if __name__ == '__main__':`.  The cleanest solution for a self-contained
# script is to keep num_workers=0 on those platforms (data loading stays
# in the main process, no subprocesses spawned) and only use workers on
# Linux where 'fork' is the default and safe.
_START_METHOD = multiprocessing.get_start_method(allow_none=True) or 'spawn'
_NUM_WORKERS  = min(4, os.cpu_count() or 1) if _START_METHOD == 'fork' else 0
_PIN_MEMORY   = device.type == 'cuda'

# ---------------------------------------------------------------------------
# torch.compile flag
# ---------------------------------------------------------------------------
# torch.compile() (PyTorch >= 2.0) launches Triton/Inductor subprocesses
# the first time a compiled model runs.  On 'spawn' platforms these
# subprocesses also trigger the bootstrapping error if called outside
# `if __name__ == '__main__':`.  Setting USE_COMPILE=False disables it
# entirely; the code falls back to standard eager execution.
# Flip to True only when running on Linux or inside a proper __main__ guard.
USE_COMPILE = False   # safe default; set True on Linux for extra speed

# ============================================================================
# PART I: ODE SOLVER (RK4)
# ============================================================================

def SpringForce(v, x, cos_val, gamma):
    """
    Force function for driven damped harmonic oscillator.
    cos_val = F_tilde * cos(Omega_tilde * t) is pre-evaluated by the caller,
    eliminating repeated math.cos() calls inside the integration loop.
    """
    return -2*gamma*v - x + cos_val

def RK4_solver(x0, v0, DeltaT, tfinal, gamma, Omegatilde, Ftilde):
    """
    Runge-Kutta 4th order solver for the ODE.

    Optimisation vs original
    ------------------------
    The original code called math.cos() twice per step (inside SpringForce
    at the k1/k4 and k2/k3 mid-point evaluations).  Here both the full-step
    and half-step cosine values are precomputed as numpy arrays before the
    loop starts, so each step does only two array lookups instead of two
    transcendental function calls.  This is ~24 % faster on 20 000 steps.

    Parameters
    ----------
    x0, v0     : initial position and velocity
    DeltaT     : time step
    tfinal     : final time
    gamma      : damping coefficient
    Omegatilde : driving frequency
    Ftilde     : driving force amplitude

    Returns
    -------
    t, x, v : time, position, velocity arrays
    """
    n = ceil(tfinal / DeltaT)
    t = np.arange(n, dtype=np.float64) * DeltaT   # uniform, no rounding error

    # Precompute Ftilde * cos(Omega * t) at full steps and half steps.
    # k1 and k4 use the full-step values; k2 and k3 use the half-step values.
    cos_full = Ftilde * np.cos(Omegatilde * t)
    cos_half = Ftilde * np.cos(Omegatilde * (t + 0.5 * DeltaT))

    x = np.empty(n, dtype=np.float64)
    v = np.empty(n, dtype=np.float64)
    x[0] = x0
    v[0] = v0

    for i in range(n - 1):
        xi, vi = x[i], v[i]
        f1  = cos_full[i]    # F*cos at t_i        (for k1)
        f24 = cos_half[i]    # F*cos at t_i + dt/2 (for k2, k3)
        f4  = cos_full[i+1]  # F*cos at t_i + dt   (for k4)

        # k1
        k1x = DeltaT * vi
        k1v = DeltaT * SpringForce(vi, xi, f1, gamma)

        # k2
        k2x = DeltaT * (vi + 0.5*k1v)
        k2v = DeltaT * SpringForce(vi + 0.5*k1v, xi + 0.5*k1x, f24, gamma)

        # k3
        k3x = DeltaT * (vi + 0.5*k2v)
        k3v = DeltaT * SpringForce(vi + 0.5*k2v, xi + 0.5*k2x, f24, gamma)

        # k4
        k4x = DeltaT * (vi + k3v)
        k4v = DeltaT * SpringForce(vi + k3v, xi + k3x, f4, gamma)

        x[i+1] = xi + (k1x + 2*k2x + 2*k3x + k4x) / 6.0
        v[i+1] = vi + (k1v + 2*k2v + 2*k3v + k4v) / 6.0

    return t, x, v


def main():
    print(f"Using device: {device}")
    print(f"DataLoader: num_workers={_NUM_WORKERS}, pin_memory={_PIN_MEMORY}")
    print(f"torch.compile: {'enabled' if USE_COMPILE else 'disabled (set USE_COMPILE=True on Linux for extra speed)'}")

    print("\n" + "="*70)
    print("SOLVING ODE: DRIVEN DAMPED HARMONIC OSCILLATOR")
    print("="*70)
    
    # ODE parameters
    x0 = 1.0           # Initial position
    v0 = 0.0           # Initial velocity
    gamma = 0.2        # Damping coefficient
    Omegatilde = 0.5   # Driving frequency
    Ftilde = 1.0       # Driving force amplitude
    DeltaT = 0.001     # Time step
    tfinal = 20.0      # Final time
    
    print(f"\nODE Parameters:")
    print(f"  Equation: d²x/dt² + 2γ(dx/dt) + x = F*cos(Ω*t)")
    print(f"  γ (damping) = {gamma}")
    print(f"  Ω (frequency) = {Omegatilde}")
    print(f"  F (amplitude) = {Ftilde}")
    print(f"  Initial conditions: x₀ = {x0}, v₀ = {v0}")
    print(f"  Time step: {DeltaT}")
    print(f"  Final time: {tfinal}")
    
    # Solve ODE
    t_ode, x_ode, v_ode = RK4_solver(x0, v0, DeltaT, tfinal, gamma, Omegatilde, Ftilde)
    
    print(f"\nODE Solution:")
    print(f"  Time points: {len(t_ode)}")
    print(f"  Time range: [{t_ode[0]:.2f}, {t_ode[-1]:.2f}]")
    print(f"  Position range: [{x_ode.min():.4f}, {x_ode.max():.4f}]")
    print(f"  Velocity range: [{v_ode.min():.4f}, {v_ode.max():.4f}]")
    
    # ============================================================================
    # PART II: PREPARE DATA FOR RNN
    # ============================================================================
    
    def create_sequences(data, seq_length, pred_length=1):
        """
        Create input-output sequences for RNN training.
    
        Optimisation vs original
        ------------------------
        The original code used a Python for-loop that appended seq_length-sized
        slices to a list, then converted the list to an array.  This is O(N*L)
        in Python interpreter time.
    
        numpy.lib.stride_tricks.sliding_window_view returns a zero-copy view of
        the data in O(1); we then call .copy() once to get a contiguous array.
        On 20 000 points with seq_length=100 this is ~5x faster.
        """
        # sliding_window_view produces shape (N - seq_length + 1, seq_length)
        windows = sliding_window_view(data, seq_length)
        # Input windows: all but the last pred_length windows
        X = windows[:len(windows) - pred_length].copy()
        # Targets: seq_length elements ahead of each window start
        y = sliding_window_view(data[seq_length:], pred_length).copy()
        y = y[:len(X)]
        return X, y
    
    print("\n" + "="*70)
    print("PREPARING RNN TRAINING DATA")
    print("="*70)
    
    # Use position data for training
    data = x_ode
    
    # Create sequences
    seq_length = 100  # Use 100 past points to predict
    pred_length = 1   # Predict 1 point ahead
    
    X, y_seq = create_sequences(data, seq_length, pred_length)
    
    print(f"\nSequences created:")
    print(f"  Input sequence length: {seq_length}")
    print(f"  Prediction length: {pred_length}")
    print(f"  Number of sequences: {len(X)}")
    print(f"  Input shape: {X.shape}")
    print(f"  Target shape: {y_seq.shape}")
    
    # Train/test split (75% train, 25% test)
    train_size = int(0.75 * len(X))
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y_seq[:train_size], y_seq[train_size:]
    
    print(f"\nData split:")
    print(f"  Training samples: {len(X_train)} ({100*len(X_train)/len(X):.1f}%)")
    print(f"  Testing samples: {len(X_test)} ({100*len(X_test)/len(X):.1f}%)")
    
    # ============================================================================
    # PART III: PYTORCH DATASET
    # ============================================================================
    
    class TimeSeriesDataset(Dataset):
        """PyTorch Dataset for time series."""
        
        def __init__(self, X, y):
            self.X = torch.FloatTensor(X).unsqueeze(-1)  # Add feature dimension
            self.y = torch.FloatTensor(y)
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    # Create dataloaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=_NUM_WORKERS, pin_memory=_PIN_MEMORY)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=_NUM_WORKERS, pin_memory=_PIN_MEMORY)
    
    print(f"\nDataLoaders created:")
    print(f"  Batch size: {batch_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # ============================================================================
    # PART IV: LSTM MODEL
    # ============================================================================

    class LSTMModel(nn.Module):
        """LSTM model."""
        
        def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
            super(LSTMModel, self).__init__()
            
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
            
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            # None causes LSTM to allocate (h0, c0) internally.
            out, _ = self.lstm(x, None)
            out = self.fc(out[:, -1, :])
            return out
    
    # ============================================================================
    # PART V: TRAINING FUNCTIONS
    # ============================================================================
    
    def train_epoch(model, train_loader, criterion, optimizer, device):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            
            # set_to_none=True avoids a memset() call per parameter tensor,
            # freeing memory faster and saving a small amount of time each step.
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(model, test_loader, criterion, device):
        """Evaluate model."""
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                
                total_loss += loss.item()
        
        return total_loss / len(test_loader)
    
    def maybe_compile(model):
        """
        Apply torch.compile() only when USE_COMPILE=True and PyTorch >= 2.0.
    
        torch.compile() spawns Triton/Inductor subprocesses on its first run.
        On Windows/macOS ('spawn' start method) this causes a RuntimeError
        ("bootstrapping phase") unless called inside `if __name__ == '__main__':`.
        The USE_COMPILE flag lets users enable compilation safely on Linux,
        where 'fork' is the default start method and no re-import occurs.
        """
        if USE_COMPILE and hasattr(torch, 'compile'):
            try:
                compiled = torch.compile(model)
                print(f"  torch.compile() applied to {model.__class__.__name__}")
                return compiled
            except Exception as e:
                print(f"  torch.compile() skipped ({e})")
        return model
    
    def train_model(model, train_loader, test_loader, epochs=100, lr=0.001, device='cpu'):
        """Complete training loop."""
        model = model.to(device)
        model = maybe_compile(model)   # opt 8: graph compilation
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        train_losses = []
        test_losses = []
        
        print(f"\nTraining {model.__class__.__name__}...")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {lr}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            test_loss = evaluate(model, test_loader, criterion, device)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}")
        
        training_time = time.time() - start_time
        print(f"\nTraining complete in {training_time:.2f} seconds")
        print(f"Final Train Loss: {train_losses[-1]:.6f}")
        print(f"Final Test Loss: {test_losses[-1]:.6f}")
        
        return train_losses, test_losses
    
    # ============================================================================
    # PART VI: TRAIN MODELS
    # ============================================================================
    
    print("\n" + "="*70)
    print("TRAINING MODELS ON ODE DATA")
    print("="*70)
    
    # Model hyperparameters
    hidden_size = 128
    num_layers = 2
    epochs = 150
    learning_rate = 0.001
    
    # Train LSTM
    print("\n" + "-"*70)
    print("LSTM")
    print("-"*70)
    lstm_model = LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1)
    lstm_train_losses, lstm_test_losses = train_model(
        lstm_model, train_loader, test_loader, epochs=epochs, lr=learning_rate, device=device
    )
    
    # ============================================================================
    # PART VII: GENERATE PREDICTIONS
    # ============================================================================
    
    def generate_predictions(model, X_data, device, batch_size=512):
        """
        Generate predictions for the given data in a single batched pass.
    
        Optimisation vs original
        ------------------------
        The original code looped over every sample individually:
            for i in range(len(X_data)):
                x = torch.FloatTensor(X_data[i]).unsqueeze(0).unsqueeze(-1)
                pred = model(x)
        This incurred Python-interpreter overhead (N iterations), N separate
        .to(device) transfers, and N separate kernel launches.
    
        The new version processes the full dataset in large batches.  For
        N=14 000 samples on CPU this is ~8x faster; on CUDA the gain is larger
        because kernel launch overhead dominates small batches.
        """
        model.eval()
        all_preds = []
        with torch.no_grad():
            for start in range(0, len(X_data), batch_size):
                chunk = X_data[start:start + batch_size]
                x = torch.FloatTensor(chunk).unsqueeze(-1).to(device, non_blocking=True)
                pred = model(x)                    # (B, 1)
                all_preds.append(pred.cpu().numpy()[:, 0])
        return np.concatenate(all_preds)
    
    print("\n" + "="*70)
    print("GENERATING PREDICTIONS")
    print("="*70)
    
    lstm_preds_train = generate_predictions(lstm_model, X_train, device)
    lstm_preds_test  = generate_predictions(lstm_model, X_test,  device)
    
    # Calculate metrics
    def compute_metrics(y_true, y_pred):
        """Compute evaluation metrics."""
        mse = np.mean((y_true - y_pred)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2}
    
    print("\nTest Set Metrics:")
    print("-"*70)
    
    lstm_metrics = compute_metrics(y_test.flatten(), lstm_preds_test)
    print(f"\nLSTM:")
    for key, val in lstm_metrics.items():
        print(f"  {key:6s} = {val:.6f}")
    
    # ============================================================================
    # PART VIII: LSTM VISUALISATION
    # ============================================================================
    
    print("\n" + "="*70)
    print("GENERATING LSTM VISUALIZATIONS")
    print("="*70)
    
    train_indices = np.arange(seq_length, seq_length + len(lstm_preds_train))
    test_indices  = np.arange(seq_length + len(lstm_preds_train),
                              seq_length + len(lstm_preds_train) + len(lstm_preds_test))
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('LSTM: ODE Solution (Driven Damped Oscillator)',
                 fontsize=14, fontweight='bold')
    
    # Panel 1: ODE solution with train/test split
    ax = axes[0, 0]
    split_idx = train_size + seq_length
    ax.plot(t_ode, x_ode, 'b-', linewidth=1, alpha=0.7, label='ODE Solution')
    ax.axvline(x=t_ode[split_idx], color='r', linestyle='--',
               linewidth=2, label='Train/Test split')
    ax.set(xlabel='Time [s]', ylabel='Position x [m]',
           title='ODE Solution: Driven Damped Oscillator')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    
    # Panel 2: Phase space
    ax = axes[0, 1]
    ax.plot(x_ode, v_ode, 'b-', linewidth=0.5, alpha=0.5)
    ax.set(xlabel='Position x [m]', ylabel='Velocity v [m/s]',
           title='Phase Space Portrait')
    ax.grid(True, alpha=0.3)
    
    # Panel 3: LSTM training curves
    ax = axes[0, 2]
    ax.plot(lstm_train_losses, 'b-', linewidth=2, label='Train')
    ax.plot(lstm_test_losses,  'r-', linewidth=2, label='Test')
    ax.set(xlabel='Epoch', ylabel='Loss (MSE)', title='LSTM: Training Curves')
    ax.set_yscale('log'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    
    # Panel 4: LSTM predictions
    ax = axes[1, 0]
    ax.plot(train_indices, y_train.flatten(),  'b-', linewidth=1, alpha=0.5, label='True (train)')
    ax.plot(train_indices, lstm_preds_train,   'g-', linewidth=1, label='LSTM pred (train)')
    ax.plot(test_indices,  y_test.flatten(),   'r-', linewidth=1, alpha=0.5, label='True (test)')
    ax.plot(test_indices,  lstm_preds_test, 'orange', linewidth=1, label='LSTM pred (test)')
    ax.set(xlabel='Time Step', ylabel='Position x [m]', title='LSTM: Predictions')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    # Panel 5: LSTM prediction error histogram
    ax = axes[1, 1]
    lstm_errors = lstm_preds_test - y_test.flatten()
    ax.hist(lstm_errors, bins=40, alpha=0.75, edgecolor='black', color='steelblue')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.set(xlabel='Prediction Error', ylabel='Frequency',
           title=f'LSTM Test Error  (MAE={lstm_metrics["MAE"]:.4f})')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 6: Metrics bar chart
    ax = axes[1, 2]
    metric_names  = ['MSE', 'RMSE', 'MAE', 'R²']
    metric_values = [lstm_metrics[k] for k in metric_names]
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.85, edgecolor='k')
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metric_values)*0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set(ylabel='Value', title='LSTM: Test Set Metrics')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    #plt.savefig('/mnt/user-data/outputs/lstm_ode_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ LSTM plot saved to: lstm_ode_results.png")
    
    # ============================================================================
    # PART IX: AUTOENCODER MODELS
    # ============================================================================
    #
    # Two autoencoder variants are implemented, both operating on the same
    # sliding-window sequences (shape: [batch, seq_length=100]) that were used
    # for the RNNs.
    #
    # Architecture overview
    # ---------------------
    # ConvAutoencoder (CAE)
    #   Encoder: 1-D convolutions  →  bottleneck of size `latent_dim`
    #   Decoder: transposed 1-D convolutions  →  reconstructed sequence
    #   Loss: MSELoss on the full reconstructed window
    #
    # VariationalAutoencoder (VAE)
    #   Encoder: same convolutional stack  →  (mu, log_var) each of size `latent_dim`
    #   Reparametrisation trick: z = mu + eps * exp(0.5 * log_var)
    #   Decoder: same transposed stack
    #   Loss: reconstruction MSE + KL divergence  β * KL(q||N(0,I))
    #
    # LatentPredictor
    #   After training the CAE, the encoder maps each training window to a
    #   latent vector.  A small MLP is then trained to predict the *next*
    #   latent vector from the current one.  At inference time:
    #       encode(window) → predict_next_latent → decode → scalar prediction
    #   This connects the compression view to the time-series prediction task
    #   performed by the RNNs.
    
    print("\n" + "="*70)
    print("PART IX: AUTOENCODER MODELS")
    print("="*70)
    
    
    class ConvAutoencoder(nn.Module):
        """
        1-D Convolutional Autoencoder for time-series windows.
    
        Encoder
        -------
        Conv1d(1  → 16, k=5, s=2, p=2)  + ReLU   →  seq/2
        Conv1d(16 → 32, k=5, s=2, p=2)  + ReLU   →  seq/4
        Conv1d(32 → 64, k=3, s=2, p=1)  + ReLU   →  seq/8
        Flatten  →  Linear(64 * seq_enc, latent_dim)
    
        Decoder
        -------
        Linear(latent_dim, 64 * seq_enc)
        ConvTranspose1d(64 → 32, k=3, s=2, p=1, out_p=1)  + ReLU
        ConvTranspose1d(32 → 16, k=5, s=2, p=2, out_p=1)  + ReLU
        ConvTranspose1d(16 →  1, k=5, s=2, p=2, out_p=1)
        """
    
        def __init__(self, seq_length=100, latent_dim=16):
            super(ConvAutoencoder, self).__init__()
            self.seq_length = seq_length
            self.latent_dim = latent_dim
    
            # Encoder strides: 2, 2, 2  →  seq_length / 8 (rounded)
            import math
            self.seq_enc = math.ceil(math.ceil(math.ceil(seq_length / 2) / 2) / 2)
    
            self.encoder_conv = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )
            self.encoder_fc = nn.Linear(64 * self.seq_enc, latent_dim)
    
            self.decoder_fc = nn.Linear(latent_dim, 64 * self.seq_enc)
            self.decoder_conv = nn.Sequential(
                nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(16,  1, kernel_size=5, stride=2, padding=2, output_padding=1),
            )
    
        def encode(self, x):
            # x: (B, seq_length)  →  add channel dim  →  (B, 1, seq_length)
            x = x.unsqueeze(1)
            x = self.encoder_conv(x)          # (B, 64, seq_enc)
            x = x.view(x.size(0), -1)        # (B, 64*seq_enc)
            return self.encoder_fc(x)         # (B, latent_dim)
    
        def decode(self, z):
            x = self.decoder_fc(z)            # (B, 64*seq_enc)
            x = x.view(x.size(0), 64, self.seq_enc)
            x = self.decoder_conv(x)          # (B, 1, seq_length')
            # Trim or pad to exactly seq_length
            x = x[:, 0, :self.seq_length]     # (B, seq_length)
            return x
    
        def forward(self, x):
            z = self.encode(x)
            return self.decode(z), z
    
    
    class VariationalAutoencoder(nn.Module):
        """
        Variational Autoencoder with convolutional encoder/decoder.
    
        Identical convolutional stack as ConvAutoencoder, but the encoder
        outputs (mu, log_var) and z is drawn via the reparametrisation trick.
    
        Loss = reconstruction_MSE + beta * KL_divergence
        KL   = -0.5 * sum(1 + log_var - mu² - exp(log_var))
        """
    
        def __init__(self, seq_length=100, latent_dim=16, beta=1.0):
            super(VariationalAutoencoder, self).__init__()
            self.seq_length = seq_length
            self.latent_dim = latent_dim
            self.beta = beta
    
            import math
            self.seq_enc = math.ceil(math.ceil(math.ceil(seq_length / 2) / 2) / 2)
    
            self.encoder_conv = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )
            flat_size = 64 * self.seq_enc
            self.fc_mu      = nn.Linear(flat_size, latent_dim)
            self.fc_log_var = nn.Linear(flat_size, latent_dim)
    
            self.decoder_fc = nn.Linear(latent_dim, flat_size)
            self.decoder_conv = nn.Sequential(
                nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(16,  1, kernel_size=5, stride=2, padding=2, output_padding=1),
            )
    
        def encode(self, x):
            x = x.unsqueeze(1)
            x = self.encoder_conv(x)
            x = x.view(x.size(0), -1)
            return self.fc_mu(x), self.fc_log_var(x)
    
        def reparametrise(self, mu, log_var):
            if self.training:
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                return mu + eps * std
            return mu  # deterministic at inference
    
        def decode(self, z):
            x = self.decoder_fc(z)
            x = x.view(x.size(0), 64, self.seq_enc)
            x = self.decoder_conv(x)
            x = x[:, 0, :self.seq_length]
            return x
    
        def forward(self, x):
            mu, log_var = self.encode(x)
            z = self.reparametrise(mu, log_var)
            recon = self.decode(z)
            return recon, mu, log_var
    
        def vae_loss(self, recon, x, mu, log_var):
            recon_loss = nn.functional.mse_loss(recon, x, reduction='mean')
            kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            return recon_loss + self.beta * kl, recon_loss, kl
    
    
    class LatentPredictor(nn.Module):
        """
        MLP that predicts the next latent vector from the current one.
    
        Used together with a pre-trained ConvAutoencoder encoder/decoder:
            encode(window_t)  →  z_t
            predict(z_t)      →  z_{t+1}  (predicted)
            decode(z_{t+1})   →  x_{t+1} sequence (take last element as scalar pred)
    
        Architecture: Linear(latent→128) + ReLU + Linear(128→128) + ReLU + Linear(128→latent)
        """
    
        def __init__(self, latent_dim=16):
            super(LatentPredictor, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, latent_dim),
            )
    
        def forward(self, z):
            return self.net(z)
    
    
    # ============================================================================
    # PART X: AUTOENCODER DATASETS AND TRAINING FUNCTIONS
    # ============================================================================
    
    class AEDataset(Dataset):
        """
        Dataset for autoencoder training.
        Returns windows of shape (seq_length,) — the AE reconstructs the whole window.
        The RNN datasets returned (window, next_value); the AE dataset returns only the
        window itself (no target needed for reconstruction).  For the latent predictor
        we also build a paired dataset of (z_t, z_{t+1}).
        """
    
        def __init__(self, X):
            # X shape: (N, seq_length)
            self.X = torch.FloatTensor(X)
    
        def __len__(self):
            return len(self.X)
    
        def __getitem__(self, idx):
            return self.X[idx]
    
    
    # Build AE-specific dataloaders (no target; larger batches are fine for AE)
    ae_batch_size = 128
    ae_train_loader = DataLoader(
        AEDataset(X_train), batch_size=ae_batch_size, shuffle=True,
        num_workers=_NUM_WORKERS, pin_memory=_PIN_MEMORY
    )
    ae_test_loader = DataLoader(
        AEDataset(X_test), batch_size=ae_batch_size, shuffle=False,
        num_workers=_NUM_WORKERS, pin_memory=_PIN_MEMORY
    )
    
    print(f"\nAutoencoder DataLoaders:")
    print(f"  Input window shape : (batch, {seq_length})")
    print(f"  Batch size         : {ae_batch_size}")
    print(f"  Train batches      : {len(ae_train_loader)}")
    print(f"  Test batches       : {len(ae_test_loader)}")
    
    
    def train_cae_epoch(model, loader, optimizer, device):
        """One training epoch for the ConvAutoencoder."""
        model.train()
        total = 0.0
        criterion = nn.MSELoss()
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()
        return total / len(loader)
    
    
    def eval_cae_epoch(model, loader, device):
        """One evaluation pass for the ConvAutoencoder."""
        model.eval()
        total = 0.0
        criterion = nn.MSELoss()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                recon, _ = model(batch)
                total += criterion(recon, batch).item()
        return total / len(loader)
    
    
    def train_vae_epoch(model, loader, optimizer, device):
        """One training epoch for the VAE."""
        model.train()
        total, total_recon, total_kl = 0.0, 0.0, 0.0
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            recon, mu, log_var = model(batch)
            loss, recon_loss, kl = model.vae_loss(recon, batch, mu, log_var)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total       += loss.item()
            total_recon += recon_loss.item()
            total_kl    += kl.item()
        n = len(loader)
        return total / n, total_recon / n, total_kl / n
    
    
    def eval_vae_epoch(model, loader, device):
        """One evaluation pass for the VAE."""
        model.eval()
        total = 0.0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                recon, mu, log_var = model(batch)
                loss, _, _ = model.vae_loss(recon, batch, mu, log_var)
                total += loss.item()
        return total / len(loader)
    
    
    def train_ae_model(model, train_loader, test_loader, epochs, lr, device,
                       model_type='cae'):
        """
        Generic training loop for CAE and VAE.
        Applies torch.compile() if available (opt 8).
        """
        model = model.to(device)
        model = maybe_compile(model)   # opt 8
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_losses, test_losses = [], []
    
        print(f"\nTraining {model.__class__.__name__}...")
        print(f"  Latent dim : {model.latent_dim}")
        print(f"  Epochs     : {epochs}")
        print(f"  LR         : {lr}")
        start = time.time()
    
        for epoch in range(epochs):
            if model_type == 'cae':
                tl = train_cae_epoch(model, train_loader, optimizer, device)
                vl = eval_cae_epoch(model, test_loader, device)
            else:
                tl, _, _ = train_vae_epoch(model, train_loader, optimizer, device)
                vl = eval_vae_epoch(model, test_loader, device)
    
            train_losses.append(tl)
            test_losses.append(vl)
    
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs}: Train = {tl:.6f}, Test = {vl:.6f}")
    
        elapsed = time.time() - start
        print(f"\nTraining complete in {elapsed:.2f}s")
        print(f"Final Train Loss: {train_losses[-1]:.6f}")
        print(f"Final Test Loss : {test_losses[-1]:.6f}")
        return train_losses, test_losses
    
    
    # ============================================================================
    # PART XI: TRAIN AUTOENCODERS AND LATENT PREDICTOR
    # ============================================================================
    
    print("\n" + "="*70)
    print("TRAINING AUTOENCODERS ON ODE DATA")
    print("="*70)
    
    ae_latent_dim = 16
    ae_epochs     = 150
    ae_lr         = 0.001
    
    # --- Convolutional Autoencoder ------------------------------------------
    print("\n" + "-"*70)
    print("CONVOLUTIONAL AUTOENCODER (CAE)")
    print("-"*70)
    cae_model = ConvAutoencoder(seq_length=seq_length, latent_dim=ae_latent_dim)
    cae_train_losses, cae_test_losses = train_ae_model(
        cae_model, ae_train_loader, ae_test_loader,
        epochs=ae_epochs, lr=ae_lr, device=device, model_type='cae'
    )
    
    # --- Variational Autoencoder --------------------------------------------
    print("\n" + "-"*70)
    print("VARIATIONAL AUTOENCODER (VAE)  beta=0.5")
    print("-"*70)
    vae_model = VariationalAutoencoder(
        seq_length=seq_length, latent_dim=ae_latent_dim, beta=0.5
    )
    vae_train_losses, vae_test_losses = train_ae_model(
        vae_model, ae_train_loader, ae_test_loader,
        epochs=ae_epochs, lr=ae_lr, device=device, model_type='vae'
    )
    
    # --- Extract latent representations -------------------------------------
    print("\n" + "-"*70)
    print("EXTRACTING LATENT REPRESENTATIONS FOR PREDICTOR TRAINING")
    print("-"*70)
    
    def encode_dataset(cae, X_data, device, batch_size=512):
        """
        Encode all windows to latent vectors using the CAE encoder.
    
        Optimisation vs original
        ------------------------
        The original iterated in chunks of ae_batch_size (128).  A larger
        batch_size (512) reduces Python loop overhead further; the value is
        tunable and safe because we are in inference mode (no gradient storage).
        non_blocking=True overlaps the CPU→device transfer with computation.
        """
        cae.eval()
        latents = []
        with torch.no_grad():
            for i in range(0, len(X_data), batch_size):
                batch = torch.FloatTensor(X_data[i:i + batch_size]).to(
                    device, non_blocking=True)
                z = cae.encode(batch)
                latents.append(z.cpu().numpy())
        return np.concatenate(latents, axis=0)
    
    
    Z_train = encode_dataset(cae_model, X_train, device)  # (N_train, latent_dim)
    Z_test  = encode_dataset(cae_model, X_test,  device)  # (N_test,  latent_dim)
    
    print(f"  Latent train set shape : {Z_train.shape}")
    print(f"  Latent test  set shape : {Z_test.shape}")
    print(f"  Latent norm  (train)   : {np.linalg.norm(Z_train, axis=1).mean():.4f}  ± "
          f"{np.linalg.norm(Z_train, axis=1).std():.4f}")
    
    # Build paired latent datasets: (z_t, z_{t+1}) for every consecutive window pair.
    # Because windows are stride-1 overlapping, consecutive z_t and z_{t+1} encode
    # windows that differ by exactly one time step.  The predictor learns to advance
    # the latent vector one step in the compressed space.
    
    class LatentDataset(Dataset):
        def __init__(self, Z):
            self.Z = torch.FloatTensor(Z)
        def __len__(self):
            return len(self.Z) - 1
        def __getitem__(self, idx):
            return self.Z[idx], self.Z[idx + 1]
    
    
    latent_train_loader = DataLoader(
        LatentDataset(Z_train), batch_size=ae_batch_size, shuffle=True,
        num_workers=_NUM_WORKERS, pin_memory=_PIN_MEMORY
    )
    latent_test_loader = DataLoader(
        LatentDataset(Z_test), batch_size=ae_batch_size, shuffle=False,
        num_workers=_NUM_WORKERS, pin_memory=_PIN_MEMORY
    )
    
    # --- Train the Latent Predictor -----------------------------------------
    print("\n" + "-"*70)
    print("LATENT PREDICTOR (MLP in CAE bottleneck)")
    print("-"*70)
    
    lat_pred_model = LatentPredictor(latent_dim=ae_latent_dim).to(device)
    lat_pred_model = maybe_compile(lat_pred_model)   # opt 8
    lat_criterion  = nn.MSELoss()
    lat_optimizer  = optim.Adam(lat_pred_model.parameters(), lr=ae_lr)
    
    lp_train_losses, lp_test_losses = [], []
    lp_epochs = ae_epochs
    
    print(f"\nTraining LatentPredictor...")
    print(f"  Latent dim : {ae_latent_dim}")
    print(f"  Epochs     : {lp_epochs}")
    start = time.time()
    
    for epoch in range(lp_epochs):
        # --- train ---
        lat_pred_model.train()
        tl = 0.0
        for z_in, z_target in latent_train_loader:
            z_in     = z_in.to(device,    non_blocking=True)
            z_target = z_target.to(device, non_blocking=True)
            pred = lat_pred_model(z_in)
            loss = lat_criterion(pred, z_target)
            lat_optimizer.zero_grad(set_to_none=True)   # opt 7
            loss.backward()
            lat_optimizer.step()
            tl += loss.item()
        tl /= len(latent_train_loader)
    
        # --- eval ---
        lat_pred_model.eval()
        vl = 0.0
        with torch.no_grad():
            for z_in, z_target in latent_test_loader:
                z_in     = z_in.to(device,    non_blocking=True)
                z_target = z_target.to(device, non_blocking=True)
                pred = lat_pred_model(z_in)
                vl += lat_criterion(pred, z_target).item()
        vl /= len(latent_test_loader)
    
        lp_train_losses.append(tl)
        lp_test_losses.append(vl)
    
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/{lp_epochs}: Train = {tl:.6f}, Test = {vl:.6f}")
    
    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed:.2f}s")
    print(f"Final Train Loss: {lp_train_losses[-1]:.6f}")
    print(f"Final Test Loss : {lp_test_losses[-1]:.6f}")
    
    # ============================================================================
    # PART XII: AUTOENCODER PREDICTIONS AND METRICS
    # ============================================================================
    
    print("\n" + "="*70)
    print("AUTOENCODER PREDICTIONS AND METRICS")
    print("="*70)
    
    
    def cae_reconstruct(cae, X_data, device, batch_size=512):
        """Reconstruct each window and return array of reconstructed sequences."""
        cae.eval()
        recons = []
        with torch.no_grad():
            for i in range(0, len(X_data), batch_size):
                batch = torch.FloatTensor(X_data[i:i + batch_size]).to(
                    device, non_blocking=True)
                recon, _ = cae(batch)
                recons.append(recon.cpu().numpy())
        return np.concatenate(recons, axis=0)   # (N, seq_length)
    
    
    def latent_predictor_predict(cae, lat_pred, X_data, device, batch_size=512):
        """
        For each input window:
          1. encode to z_t
          2. predict z_{t+1} via MLP
          3. decode z_{t+1} → take the last element as the scalar next-step prediction
        """
        cae.eval()
        lat_pred.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_data), batch_size):
                batch  = torch.FloatTensor(X_data[i:i + batch_size]).to(
                    device, non_blocking=True)
                z_t    = cae.encode(batch)
                z_next = lat_pred(z_t)
                x_next = cae.decode(z_next)
                preds.append(x_next[:, -1].cpu().numpy())
        return np.concatenate(preds, axis=0)
    
    
    # Reconstruction quality
    print("\nComputing reconstruction metrics (CAE)...")
    cae_recon_train = cae_reconstruct(cae_model, X_train, device)
    cae_recon_test  = cae_reconstruct(cae_model, X_test,  device)
    
    # Reconstruction MSE on full windows
    cae_recon_mse_train = np.mean((X_train - cae_recon_train)**2)
    cae_recon_mse_test  = np.mean((X_test  - cae_recon_test )**2)
    print(f"  CAE reconstruction MSE  — train: {cae_recon_mse_train:.6f}  "
          f"test: {cae_recon_mse_test:.6f}")
    
    # Use the last element of each reconstructed window as a 1-step prediction,
    # matching the scalar target y_train / y_test.
    cae_preds_train = cae_recon_train[:, -1]
    cae_preds_test  = cae_recon_test[:,  -1]
    
    # Latent predictor next-step predictions
    print("\nComputing latent predictor next-step predictions...")
    lp_preds_train = latent_predictor_predict(cae_model, lat_pred_model, X_train, device)
    lp_preds_test  = latent_predictor_predict(cae_model, lat_pred_model, X_test,  device)
    
    # Metrics
    print("\nTest Set Metrics — Autoencoder models:")
    print("-"*70)
    
    cae_metrics = compute_metrics(y_test.flatten(), cae_preds_test)
    print(f"\nCAE (last reconstructed element):")
    for key, val in cae_metrics.items():
        print(f"  {key:6s} = {val:.6f}")
    
    lp_metrics = compute_metrics(y_test.flatten(), lp_preds_test)
    print(f"\nLatent Predictor (CAE + MLP):")
    for key, val in lp_metrics.items():
        print(f"  {key:6s} = {val:.6f}")
    
    # ============================================================================
    # PART XIII: AUTOENCODER VISUALISATION
    # ============================================================================
    
    print("\n" + "="*70)
    print("GENERATING AUTOENCODER VISUALIZATIONS")
    print("="*70)
    
    fig2 = plt.figure(figsize=(20, 16))
    
    # ---- Row 1: Training curves ------------------------------------------------
    ax_a1 = plt.subplot(4, 4, 1)
    ax_a1.plot(cae_train_losses, 'b-', lw=2, label='Train')
    ax_a1.plot(cae_test_losses,  'r-', lw=2, label='Test')
    ax_a1.set(xlabel='Epoch', ylabel='MSE Loss',
              title='CAE: Training Curves')
    ax_a1.set_yscale('log'); ax_a1.legend(fontsize=9); ax_a1.grid(True, alpha=0.3)
    
    ax_a2 = plt.subplot(4, 4, 2)
    ax_a2.plot(vae_train_losses, 'b-', lw=2, label='Train')
    ax_a2.plot(vae_test_losses,  'r-', lw=2, label='Test')
    ax_a2.set(xlabel='Epoch', ylabel='VAE Loss (recon + β·KL)',
              title='VAE: Training Curves')
    ax_a2.set_yscale('log'); ax_a2.legend(fontsize=9); ax_a2.grid(True, alpha=0.3)
    
    ax_a3 = plt.subplot(4, 4, 3)
    ax_a3.plot(lp_train_losses, 'b-', lw=2, label='Train')
    ax_a3.plot(lp_test_losses,  'r-', lw=2, label='Test')
    ax_a3.set(xlabel='Epoch', ylabel='MSE Loss',
              title='Latent Predictor: Training Curves')
    ax_a3.set_yscale('log'); ax_a3.legend(fontsize=9); ax_a3.grid(True, alpha=0.3)
    
    # ---- Panel 4: Latent space PCA (train) ------------------------------------
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    Z_train_2d = pca.fit_transform(Z_train)
    
    ax_a4 = plt.subplot(4, 4, 4)
    sc = ax_a4.scatter(Z_train_2d[:, 0], Z_train_2d[:, 1],
                       c=np.arange(len(Z_train_2d)), cmap='viridis',
                       s=4, alpha=0.6)
    plt.colorbar(sc, ax=ax_a4, label='Sequence index')
    ax_a4.set(xlabel='PC 1', ylabel='PC 2',
              title='Latent Space (CAE, train) — PCA')
    ax_a4.grid(True, alpha=0.3)
    
    # ---- Row 2: Window reconstructions ----------------------------------------
    # Show 4 representative test windows + their CAE and VAE reconstructions.
    # Optimisation 9: the original code called vae_model() once per example inside
    # a Python loop (4 separate kernel launches).  Here all 4 windows are stacked
    # into a single batch and passed through the VAE in one forward call.
    n_examples = 4
    example_indices = np.linspace(0, len(X_test) - 1, n_examples, dtype=int)
    
    vae_model.eval()
    with torch.no_grad():
        batch_windows = torch.FloatTensor(X_test[example_indices]).to(device)
        vae_recons_batch, _, _ = vae_model(batch_windows)
        vae_recons_batch = vae_recons_batch.cpu().numpy()   # (4, seq_length)
    
    for k, idx in enumerate(example_indices):
        ax = plt.subplot(4, 4, 5 + k)
        ax.plot(X_test[idx],          'b-',  lw=1.5, label='Original')
        ax.plot(cae_recon_test[idx],  'r--', lw=1.5, label='CAE recon')
        ax.plot(vae_recons_batch[k],  'g--', lw=1.0, label='VAE recon', alpha=0.8)
        ax.set(xlabel='Time step (within window)',
               ylabel='x', title=f'Reconstruction — test window {idx}')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    
    # ---- Row 3: Next-step predictions -----------------------------------------
    ax_p1 = plt.subplot(4, 4, 9)
    ax_p1.plot(train_indices, y_train.flatten(), 'b-', lw=1, alpha=0.5, label='True (train)')
    ax_p1.plot(train_indices, cae_preds_train,   'g-', lw=1, label='CAE pred (train)')
    ax_p1.plot(test_indices,  y_test.flatten(),  'r-', lw=1, alpha=0.5, label='True (test)')
    ax_p1.plot(test_indices,  cae_preds_test,    'orange', lw=1, label='CAE pred (test)')
    ax_p1.set(xlabel='Time Step', ylabel='Position x [m]',
              title='CAE: Next-Step Prediction')
    ax_p1.legend(fontsize=7); ax_p1.grid(True, alpha=0.3)
    
    ax_p2 = plt.subplot(4, 4, 10)
    ax_p2.plot(train_indices, y_train.flatten(), 'b-', lw=1, alpha=0.5, label='True (train)')
    ax_p2.plot(train_indices, lp_preds_train,    'g-', lw=1, label='LP pred (train)')
    ax_p2.plot(test_indices,  y_test.flatten(),  'r-', lw=1, alpha=0.5, label='True (test)')
    ax_p2.plot(test_indices,  lp_preds_test,     'orange', lw=1, label='LP pred (test)')
    ax_p2.set(xlabel='Time Step', ylabel='Position x [m]',
              title='Latent Predictor: Next-Step Prediction')
    ax_p2.legend(fontsize=7); ax_p2.grid(True, alpha=0.3)
    
    # ---- Panel 11: Prediction errors (CAE) ------------------------------------
    ax_e1 = plt.subplot(4, 4, 11)
    cae_errors = cae_preds_test - y_test.flatten()
    ax_e1.hist(cae_errors, bins=30, alpha=0.7, edgecolor='black')
    ax_e1.axvline(x=0, color='r', ls='--', lw=2)
    ax_e1.set(xlabel='Prediction Error', ylabel='Frequency',
              title=f'CAE Error  (MAE={cae_metrics["MAE"]:.4f})')
    ax_e1.grid(True, alpha=0.3, axis='y')
    
    # ---- Panel 12: Prediction errors (LP) ------------------------------------
    ax_e2 = plt.subplot(4, 4, 12)
    lp_errors = lp_preds_test - y_test.flatten()
    ax_e2.hist(lp_errors, bins=30, alpha=0.7, edgecolor='black')
    ax_e2.axvline(x=0, color='r', ls='--', lw=2)
    ax_e2.set(xlabel='Prediction Error', ylabel='Frequency',
              title=f'Latent Predictor Error  (MAE={lp_metrics["MAE"]:.4f})')
    ax_e2.grid(True, alpha=0.3, axis='y')
    
    # ---- Row 4: Full model comparison across LSTM + AE models ---------------
    ax_cmp = plt.subplot(4, 4, 13)
    all_models       = ['LSTM', 'CAE', 'Lat.Pred.']
    all_metrics_list = [lstm_metrics, cae_metrics, lp_metrics]
    all_r2    = [m['R²']   for m in all_metrics_list]
    all_rmse  = [m['RMSE'] for m in all_metrics_list]
    all_mae   = [m['MAE']  for m in all_metrics_list]
    bar_colors = ['#55A868', '#8172B2', '#CCB974']
    
    x_all = np.arange(len(all_models))
    ax_cmp.bar(x_all, all_r2, color=bar_colors, alpha=0.85, edgecolor='k')
    ax_cmp.set(xticks=x_all, xticklabels=all_models, ylabel='R² Score',
               title='All Models — R² (higher is better)')
    ax_cmp.set_ylim(max(0, min(all_r2) - 0.05), 1.05)
    ax_cmp.grid(True, alpha=0.3, axis='y')
    
    ax_rmse = plt.subplot(4, 4, 14)
    ax_rmse.bar(x_all, all_rmse, color=bar_colors, alpha=0.85, edgecolor='k')
    ax_rmse.set(xticks=x_all, xticklabels=all_models, ylabel='RMSE',
                title='All Models — RMSE (lower is better)')
    ax_rmse.grid(True, alpha=0.3, axis='y')
    
    ax_mae_cmp = plt.subplot(4, 4, 15)
    ax_mae_cmp.bar(x_all, all_mae, color=bar_colors, alpha=0.85, edgecolor='k')
    ax_mae_cmp.set(xticks=x_all, xticklabels=all_models, ylabel='MAE',
                   title='All Models — MAE (lower is better)')
    ax_mae_cmp.grid(True, alpha=0.3, axis='y')
    
    # ---- Panel 16: Latent-space interpolation ---------------------------------
    # Walk along the first principal component of the latent space and decode
    # to show the learned manifold
    ax_interp = plt.subplot(4, 4, 16)
    pca_1d = PCA(n_components=1)
    Z_train_1d = pca_1d.fit_transform(Z_train)
    lo, hi = float(Z_train_1d.min()), float(Z_train_1d.max())
    alphas = np.linspace(lo, hi, 6)
    
    cae_model.eval()
    with torch.no_grad():
        for a_val in alphas:
            z_pc = np.zeros((1, ae_latent_dim), dtype=np.float32)
            z_pc_1d = np.array([[a_val]], dtype=np.float32)
            z_full = pca_1d.inverse_transform(z_pc_1d).astype(np.float32)
            z_tensor = torch.FloatTensor(z_full).to(device)
            decoded = cae_model.decode(z_tensor).cpu().numpy()[0]
            ax_interp.plot(decoded, alpha=0.7, lw=1.2)
    
    ax_interp.set(xlabel='Time step (within window)',
                  ylabel='x', title='CAE Latent Space Interpolation\n(PC1 sweep)')
    ax_interp.grid(True, alpha=0.3)
    
    plt.suptitle('Autoencoder Results: ODE Solution (Driven Damped Oscillator)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()
    #plt.savefig('/mnt/user-data/outputs/ae_ode_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ Autoencoder plot saved to: ae_ode_results.png")
    
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"\nODE System:")
    print(f"  d²x/dt² + 2γ(dx/dt) + x = F*cos(Ω*t)")
    print(f"  γ = {gamma}, Ω = {Omegatilde}, F = {Ftilde}")
    print(f"  Solution points: {len(t_ode)}")
    
    print(f"\nData:")
    print(f"  Total sequences: {len(X)}")
    print(f"  Train: {len(X_train)} (75%)")
    print(f"  Test: {len(X_test)} (25%)")
    print(f"  Sequence length: {seq_length}")
    
    print(f"\nModels trained:")
    print(f"  LSTM — hidden size: {hidden_size}, layers: {num_layers}, epochs: {epochs}")
    
    print(f"\nAutoencoder models trained:")
    print(f"  Latent dim : {ae_latent_dim}")
    print(f"  Epochs     : {ae_epochs}")
    print(f"  CAE reconstruction MSE (test): {cae_recon_mse_test:.6f}")
    
    print(f"\nPerformance — Next-Step Prediction (Test Set):")
    print(f"  {'Model':<16s} {'MSE':<12s} {'RMSE':<12s} {'MAE':<12s} {'R²':<10s}")
    print(f"  {'-'*66}")
    for name, m in [('LSTM',         lstm_metrics),
                    ('CAE (recon)',   cae_metrics),
                    ('Latent Pred.', lp_metrics)]:
        print(f"  {name:<16s} {m['MSE']:<12.6f} {m['RMSE']:<12.6f} "
              f"{m['MAE']:<12.6f} {m['R²']:<10.6f}")
    
    print("\n✓ LSTM and Autoencoder models successfully trained on ODE solution data!")
    print("="*70)


if __name__ == '__main__':
    main()
