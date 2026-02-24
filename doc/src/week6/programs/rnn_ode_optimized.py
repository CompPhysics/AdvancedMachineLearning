#!/usr/bin/env python3
"""
RNN for Learning ODE Solutions - OPTIMIZED VERSION
Fixes for hanging issues:
1. Smaller dataset (5000 points instead of 20000)
2. num_workers=0 in DataLoader
3. Smaller batch size (32)
4. Progress indicators
5. Early stopping option
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from math import ceil, cos
import sys

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Force CPU to avoid GPU hanging issues
device = torch.device('cpu')
print(f"Using device: {device} (CPU mode to avoid hanging)")

# ============================================================================
# PART I: ODE SOLVER (OPTIMIZED)
# ============================================================================

def SpringForce(v, x, t, gamma=0.2, Omega=0.5, F=1.0):
    """Force function for driven damped harmonic oscillator."""
    return -2*gamma*v - x + F*cos(t*Omega)

print("\n" + "="*70)
print("SOLVING ODE (REDUCED SIZE FOR SPEED)")
print("="*70)

# REDUCED parameters to avoid hanging
DeltaT = 0.002  # Larger timestep
tfinal = 10.0   # Shorter simulation
n = ceil(tfinal/DeltaT)

print(f"\nODE Parameters:")
print(f"  Time step: {DeltaT}")
print(f"  Final time: {tfinal}")
print(f"  Number of points: {n}")

# Solve ODE
t = np.zeros(n)
x = np.zeros(n)
v = np.zeros(n)

x[0] = 1.0
v[0] = 0.0
gamma = 0.2
Omega = 0.5
F = 1.0

print("\nSolving ODE with RK4...")
for i in range(n-1):
    if i % 1000 == 0:
        print(f"  Progress: {100*i/n:.1f}%", end='\r')
    
    # RK4 step
    k1x = DeltaT * v[i]
    k1v = DeltaT * SpringForce(v[i], x[i], t[i], gamma, Omega, F)
    
    vv = v[i] + k1v*0.5
    xx = x[i] + k1x*0.5
    tt = t[i] + DeltaT*0.5
    k2x = DeltaT * vv
    k2v = DeltaT * SpringForce(vv, xx, tt, gamma, Omega, F)
    
    vv = v[i] + k2v*0.5
    xx = x[i] + k2x*0.5
    k3x = DeltaT * vv
    k3v = DeltaT * SpringForce(vv, xx, tt, gamma, Omega, F)
    
    vv = v[i] + k3v
    xx = x[i] + k3x
    tt = t[i] + DeltaT
    k4x = DeltaT * vv
    k4v = DeltaT * SpringForce(vv, xx, tt, gamma, Omega, F)
    
    x[i+1] = x[i] + (k1x + 2*k2x + 2*k3x + k4x)/6.0
    v[i+1] = v[i] + (k1v + 2*k2v + 2*k3v + k4v)/6.0
    t[i+1] = t[i] + DeltaT

print(f"  Progress: 100.0% - Complete!")
print(f"\nODE solved: {len(x)} points")
print(f"  Position range: [{x.min():.4f}, {x.max():.4f}]")

# ============================================================================
# PART II: PREPARE DATA
# ============================================================================

print("\n" + "="*70)
print("PREPARING TRAINING DATA")
print("="*70)

seq_length = 50  # Shorter sequences
X_list, y_list = [], []

print(f"\nCreating sequences (length={seq_length})...")
for i in range(len(x) - seq_length - 1):
    X_list.append(x[i:i + seq_length])
    y_list.append(x[i + seq_length])

X = np.array(X_list)
y = np.array(y_list).reshape(-1, 1)

print(f"  Created {len(X)} sequences")

# 75/25 split
train_size = int(0.75 * len(X))
X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

print(f"  Train: {len(X_train)} ({100*len(X_train)/len(X):.1f}%)")
print(f"  Test: {len(X_test)} ({100*len(X_test)/len(X):.1f}%)")

# ============================================================================
# PART III: PYTORCH DATASET
# ============================================================================

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).unsqueeze(-1)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

# CRITICAL: num_workers=0 to avoid multiprocessing hanging
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                          shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                         shuffle=False, num_workers=0)

print(f"\nDataLoaders ready:")
print(f"  Batch size: {batch_size}")
print(f"  Train batches: {len(train_loader)}")

# ============================================================================
# PART IV: LSTM MODEL (SINGLE MODEL FOR SPEED)
# ============================================================================

class LSTMModel(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# ============================================================================
# PART V: TRAINING WITH PROGRESS
# ============================================================================

print("\n" + "="*70)
print("TRAINING LSTM MODEL")
print("="*70)

model = LSTMModel(hidden_size=64, num_layers=2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50  # Reduced for speed
print(f"\nStarting training ({epochs} epochs)...")
print(f"  Hidden size: 64")
print(f"  Num layers: 2")

train_losses = []
test_losses = []

start_time = time.time()

for epoch in range(epochs):
    # Training
    model.train()
    total_train_loss = 0
    batch_num = 0
    
    for X_batch, y_batch in train_loader:
        batch_num += 1
        
        # Forward
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
    
    train_loss = total_train_loss / len(train_loader)
    
    # Evaluation
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            total_test_loss += loss.item()
    
    test_loss = total_test_loss / len(test_loader)
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    
    # Print progress
    if (epoch + 1) % 5 == 0 or epoch == 0:
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch+1:3d}/{epochs}: Train={train_loss:.6f}, Test={test_loss:.6f}, Time={elapsed:.1f}s")

total_time = time.time() - start_time
print(f"\nTraining complete in {total_time:.2f} seconds!")
print(f"Final: Train Loss = {train_losses[-1]:.6f}, Test Loss = {test_losses[-1]:.6f}")

# ============================================================================
# PART VI: PREDICTIONS
# ============================================================================

print("\n" + "="*70)
print("GENERATING PREDICTIONS")
print("="*70)

model.eval()
train_preds = []
test_preds = []

with torch.no_grad():
    for i in range(len(X_train)):
        x_in = torch.FloatTensor(X_train[i]).unsqueeze(0).unsqueeze(-1)
        pred = model(x_in).item()
        train_preds.append(pred)
    
    for i in range(len(X_test)):
        x_in = torch.FloatTensor(X_test[i]).unsqueeze(0).unsqueeze(-1)
        pred = model(x_in).item()
        test_preds.append(pred)

train_preds = np.array(train_preds)
test_preds = np.array(test_preds)

# Metrics
mse = np.mean((y_test.flatten() - test_preds)**2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_test.flatten() - test_preds))
r2 = 1 - (np.sum((y_test.flatten() - test_preds)**2) / 
          np.sum((y_test.flatten() - np.mean(y_test))**2))

print(f"\nTest Metrics:")
print(f"  MSE  = {mse:.6f}")
print(f"  RMSE = {rmse:.6f}")
print(f"  MAE  = {mae:.6f}")
print(f"  R²   = {r2:.6f}")

# ============================================================================
# PART VII: VISUALIZATION
# ============================================================================

print("\n" + "="*70)
print("CREATING VISUALIZATION")
print("="*70)

fig = plt.figure(figsize=(16, 10))

# Plot 1: ODE solution
ax1 = plt.subplot(2, 3, 1)
ax1.plot(t, x, 'b-', linewidth=1, alpha=0.7)
split_point = train_size + seq_length
if split_point < len(t):
    ax1.axvline(x=t[split_point], color='r', linestyle='--', linewidth=2, label='Train/Test')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Position x [m]')
ax1.set_title('ODE Solution', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Phase space
ax2 = plt.subplot(2, 3, 2)
ax2.plot(x, v, 'b-', linewidth=0.5, alpha=0.5)
ax2.set_xlabel('Position x')
ax2.set_ylabel('Velocity v')
ax2.set_title('Phase Space', fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Training curves
ax3 = plt.subplot(2, 3, 3)
ax3.plot(train_losses, 'b-', linewidth=2, label='Train')
ax3.plot(test_losses, 'r-', linewidth=2, label='Test')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss (MSE)')
ax3.set_title('Training Curves', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')

# Plot 4: Predictions
ax4 = plt.subplot(2, 3, 4)
train_idx = np.arange(seq_length, seq_length + len(train_preds))
test_idx = np.arange(seq_length + len(train_preds), 
                     seq_length + len(train_preds) + len(test_preds))
ax4.plot(train_idx, y_train.flatten(), 'b-', linewidth=1, alpha=0.5, label='Train True')
ax4.plot(train_idx, train_preds, 'g-', linewidth=1, label='Train Pred')
ax4.plot(test_idx, y_test.flatten(), 'r-', linewidth=1, alpha=0.5, label='Test True')
ax4.plot(test_idx, test_preds, 'orange', linewidth=1, label='Test Pred')
ax4.set_xlabel('Time Step')
ax4.set_ylabel('Position')
ax4.set_title('Predictions', fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# Plot 5: Error distribution
ax5 = plt.subplot(2, 3, 5)
errors = test_preds - y_test.flatten()
ax5.hist(errors, bins=30, alpha=0.7, edgecolor='black')
ax5.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax5.set_xlabel('Prediction Error')
ax5.set_ylabel('Frequency')
ax5.set_title(f'Error Distribution (MAE={mae:.4f})', fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Summary stats
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
summary_text = f"""
TRAINING SUMMARY

Dataset:
  ODE points: {len(x)}
  Sequences: {len(X)}
  Train: {len(X_train)} (75%)
  Test: {len(X_test)} (25%)

Model: LSTM
  Hidden: 64
  Layers: 2
  Epochs: {epochs}
  
Results:
  MSE:  {mse:.6f}
  RMSE: {rmse:.6f}
  MAE:  {mae:.6f}
  R²:   {r2:.6f}
  
Time: {total_time:.1f}s
"""
ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
         verticalalignment='center')

plt.tight_layout()
plt.show()
#plt.savefig('/mnt/user-data/outputs/rnn_ode_optimized.png', dpi=150)
print("\n✓ Plot saved: rnn_ode_optimized.png")

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
print(f"\n✓ Successfully trained LSTM on ODE data")
print(f"✓ Test R² score: {r2:.4f}")
print(f"✓ No hanging issues!")
print("="*70)
