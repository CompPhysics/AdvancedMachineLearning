#!/usr/bin/env python3
"""
Recurrent Neural Network for Learning ODE Solutions
Using PyTorch and RK4 Solver Output

Training RNN on forced oscillator differential equation:
d²x/dt² + 2γ(dx/dt) + x = F_tilde*cos(Ω_tilde*t)

Features:
- RK4 ODE solver for data generation
- Simple RNN, LSTM, and GRU implementations
- 70-80% train/test split
- Comprehensive visualization
- Quality metrics and analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from math import ceil, cos

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# PART I: ODE SOLVER (RK4)
# ============================================================================

def SpringForce(v, x, t, gamma, Omegatilde, Ftilde):
    """
    Force function for driven damped harmonic oscillator.
    Returns acceleration: d²x/dt² = -2γ(dx/dt) - x + F_tilde*cos(Ω_tilde*t)
    """
    return -2*gamma*v - x + Ftilde*cos(t*Omegatilde)

def RK4_solver(x0, v0, DeltaT, tfinal, gamma, Omegatilde, Ftilde):
    """
    Runge-Kutta 4th order solver for the ODE.
    
    Parameters:
    -----------
    x0, v0 : float
        Initial position and velocity
    DeltaT : float
        Time step
    tfinal : float
        Final time
    gamma : float
        Damping coefficient
    Omegatilde : float
        Driving frequency
    Ftilde : float
        Driving force amplitude
    
    Returns:
    --------
    t, x, v : arrays
        Time, position, and velocity arrays
    """
    n = ceil(tfinal/DeltaT)
    t = np.zeros(n)
    v = np.zeros(n)
    x = np.zeros(n)
    
    x[0] = x0
    v[0] = v0
    t[0] = 0.0
    
    for i in range(n-1):
        # k1
        k1x = DeltaT * v[i]
        k1v = DeltaT * SpringForce(v[i], x[i], t[i], gamma, Omegatilde, Ftilde)
        
        # k2
        vv = v[i] + k1v*0.5
        xx = x[i] + k1x*0.5
        tt = t[i] + DeltaT*0.5
        k2x = DeltaT * vv
        k2v = DeltaT * SpringForce(vv, xx, tt, gamma, Omegatilde, Ftilde)
        
        # k3
        vv = v[i] + k2v*0.5
        xx = x[i] + k2x*0.5
        k3x = DeltaT * vv
        k3v = DeltaT * SpringForce(vv, xx, tt, gamma, Omegatilde, Ftilde)
        
        # k4
        vv = v[i] + k3v
        xx = x[i] + k3x
        tt = t[i] + DeltaT
        k4x = DeltaT * vv
        k4v = DeltaT * SpringForce(vv, xx, tt, gamma, Omegatilde, Ftilde)
        
        # Update
        x[i+1] = x[i] + (k1x + 2*k2x + 2*k3x + k4x)/6.0
        v[i+1] = v[i] + (k1v + 2*k2v + 2*k3v + k4v)/6.0
        t[i+1] = t[i] + DeltaT
    
    return t, x, v

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
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length - pred_length + 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + pred_length])
    
    return np.array(X), np.array(y)

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
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"\nDataLoaders created:")
print(f"  Batch size: {batch_size}")
print(f"  Train batches: {len(train_loader)}")
print(f"  Test batches: {len(test_loader)}")

# ============================================================================
# PART IV: RNN MODELS
# ============================================================================

class SimpleRNN(nn.Module):
    """Simple RNN model."""
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(SimpleRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

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
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        
        return out

class GRUModel(nn.Module):
    """GRU model."""
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(GRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        
        return out

# ============================================================================
# PART V: TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        
        optimizer.zero_grad()
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
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            
            total_loss += loss.item()
    
    return total_loss / len(test_loader)

def train_model(model, train_loader, test_loader, epochs=100, lr=0.001, device='cpu'):
    """Complete training loop."""
    model = model.to(device)
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

# Train Simple RNN
print("\n" + "-"*70)
print("SIMPLE RNN")
print("-"*70)
rnn_model = SimpleRNN(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1)
rnn_train_losses, rnn_test_losses = train_model(
    rnn_model, train_loader, test_loader, epochs=epochs, lr=learning_rate, device=device
)

# Train LSTM
print("\n" + "-"*70)
print("LSTM")
print("-"*70)
lstm_model = LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1)
lstm_train_losses, lstm_test_losses = train_model(
    lstm_model, train_loader, test_loader, epochs=epochs, lr=learning_rate, device=device
)

# Train GRU
print("\n" + "-"*70)
print("GRU")
print("-"*70)
gru_model = GRUModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1)
gru_train_losses, gru_test_losses = train_model(
    gru_model, train_loader, test_loader, epochs=epochs, lr=learning_rate, device=device
)

# ============================================================================
# PART VII: GENERATE PREDICTIONS
# ============================================================================

def generate_predictions(model, X_data, device):
    """Generate predictions for given data."""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(len(X_data)):
            x = torch.FloatTensor(X_data[i]).unsqueeze(0).unsqueeze(-1).to(device)
            pred = model(x)
            predictions.append(pred.cpu().numpy()[0, 0])
    
    return np.array(predictions)

print("\n" + "="*70)
print("GENERATING PREDICTIONS")
print("="*70)

rnn_preds_train = generate_predictions(rnn_model, X_train, device)
rnn_preds_test = generate_predictions(rnn_model, X_test, device)

lstm_preds_train = generate_predictions(lstm_model, X_train, device)
lstm_preds_test = generate_predictions(lstm_model, X_test, device)

gru_preds_train = generate_predictions(gru_model, X_train, device)
gru_preds_test = generate_predictions(gru_model, X_test, device)

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

rnn_metrics = compute_metrics(y_test.flatten(), rnn_preds_test)
print(f"\nSimple RNN:")
for key, val in rnn_metrics.items():
    print(f"  {key:6s} = {val:.6f}")

lstm_metrics = compute_metrics(y_test.flatten(), lstm_preds_test)
print(f"\nLSTM:")
for key, val in lstm_metrics.items():
    print(f"  {key:6s} = {val:.6f}")

gru_metrics = compute_metrics(y_test.flatten(), gru_preds_test)
print(f"\nGRU:")
for key, val in gru_metrics.items():
    print(f"  {key:6s} = {val:.6f}")

# ============================================================================
# PART VIII: VISUALIZATION
# ============================================================================

print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

fig = plt.figure(figsize=(18, 14))

# Plot 1: Original ODE solution
ax1 = plt.subplot(4, 3, 1)
split_idx = train_size + seq_length
ax1.plot(t_ode, x_ode, 'b-', linewidth=1, alpha=0.7, label='ODE Solution')
ax1.axvline(x=t_ode[split_idx], color='r', linestyle='--', 
            linewidth=2, label='Train/Test split')
ax1.set_xlabel('Time [s]', fontsize=10)
ax1.set_ylabel('Position x [m]', fontsize=10)
ax1.set_title('ODE Solution: Driven Damped Oscillator', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: Phase space
ax2 = plt.subplot(4, 3, 2)
ax2.plot(x_ode, v_ode, 'b-', linewidth=0.5, alpha=0.5)
ax2.set_xlabel('Position x [m]', fontsize=10)
ax2.set_ylabel('Velocity v [m/s]', fontsize=10)
ax2.set_title('Phase Space Portrait', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Data distribution
ax3 = plt.subplot(4, 3, 3)
ax3.hist(x_ode, bins=50, alpha=0.7, edgecolor='black')
ax3.set_xlabel('Position x [m]', fontsize=10)
ax3.set_ylabel('Frequency', fontsize=10)
ax3.set_title('Position Distribution', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Training curves - RNN
ax4 = plt.subplot(4, 3, 4)
ax4.plot(rnn_train_losses, 'b-', linewidth=2, label='Train')
ax4.plot(rnn_test_losses, 'r-', linewidth=2, label='Test')
ax4.set_xlabel('Epoch', fontsize=10)
ax4.set_ylabel('Loss (MSE)', fontsize=10)
ax4.set_title('Simple RNN: Training Curves', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_yscale('log')

# Plot 5: Training curves - LSTM
ax5 = plt.subplot(4, 3, 5)
ax5.plot(lstm_train_losses, 'b-', linewidth=2, label='Train')
ax5.plot(lstm_test_losses, 'r-', linewidth=2, label='Test')
ax5.set_xlabel('Epoch', fontsize=10)
ax5.set_ylabel('Loss (MSE)', fontsize=10)
ax5.set_title('LSTM: Training Curves', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)
ax5.set_yscale('log')

# Plot 6: Training curves - GRU
ax6 = plt.subplot(4, 3, 6)
ax6.plot(gru_train_losses, 'b-', linewidth=2, label='Train')
ax6.plot(gru_test_losses, 'r-', linewidth=2, label='Test')
ax6.set_xlabel('Epoch', fontsize=10)
ax6.set_ylabel('Loss (MSE)', fontsize=10)
ax6.set_title('GRU: Training Curves', fontsize=12, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)
ax6.set_yscale('log')

# Plot 7: Predictions - Simple RNN
ax7 = plt.subplot(4, 3, 7)
train_indices = np.arange(seq_length, seq_length + len(rnn_preds_train))
test_indices = np.arange(seq_length + len(rnn_preds_train), 
                         seq_length + len(rnn_preds_train) + len(rnn_preds_test))
ax7.plot(train_indices, y_train.flatten(), 'b-', linewidth=1, alpha=0.5, label='Train True')
ax7.plot(train_indices, rnn_preds_train, 'g-', linewidth=1, label='Train Pred')
ax7.plot(test_indices, y_test.flatten(), 'r-', linewidth=1, alpha=0.5, label='Test True')
ax7.plot(test_indices, rnn_preds_test, 'orange', linewidth=1, label='Test Pred')
ax7.set_xlabel('Time Step', fontsize=10)
ax7.set_ylabel('Position x [m]', fontsize=10)
ax7.set_title('Simple RNN: Predictions', fontsize=12, fontweight='bold')
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)

# Plot 8: Predictions - LSTM
ax8 = plt.subplot(4, 3, 8)
ax8.plot(train_indices, y_train.flatten(), 'b-', linewidth=1, alpha=0.5, label='Train True')
ax8.plot(train_indices, lstm_preds_train, 'g-', linewidth=1, label='Train Pred')
ax8.plot(test_indices, y_test.flatten(), 'r-', linewidth=1, alpha=0.5, label='Test True')
ax8.plot(test_indices, lstm_preds_test, 'orange', linewidth=1, label='Test Pred')
ax8.set_xlabel('Time Step', fontsize=10)
ax8.set_ylabel('Position x [m]', fontsize=10)
ax8.set_title('LSTM: Predictions', fontsize=12, fontweight='bold')
ax8.legend(fontsize=8)
ax8.grid(True, alpha=0.3)

# Plot 9: Predictions - GRU
ax9 = plt.subplot(4, 3, 9)
ax9.plot(train_indices, y_train.flatten(), 'b-', linewidth=1, alpha=0.5, label='Train True')
ax9.plot(train_indices, gru_preds_train, 'g-', linewidth=1, label='Train Pred')
ax9.plot(test_indices, y_test.flatten(), 'r-', linewidth=1, alpha=0.5, label='Test True')
ax9.plot(test_indices, gru_preds_test, 'orange', linewidth=1, label='Test Pred')
ax9.set_xlabel('Time Step', fontsize=10)
ax9.set_ylabel('Position x [m]', fontsize=10)
ax9.set_title('GRU: Predictions', fontsize=12, fontweight='bold')
ax9.legend(fontsize=8)
ax9.grid(True, alpha=0.3)

# Plot 10: Test error - RNN
ax10 = plt.subplot(4, 3, 10)
rnn_errors = rnn_preds_test - y_test.flatten()
ax10.hist(rnn_errors, bins=30, alpha=0.7, edgecolor='black')
ax10.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax10.set_xlabel('Prediction Error', fontsize=10)
ax10.set_ylabel('Frequency', fontsize=10)
ax10.set_title(f'RNN Error (MAE={rnn_metrics["MAE"]:.4f})', fontsize=12, fontweight='bold')
ax10.grid(True, alpha=0.3, axis='y')

# Plot 11: Test error - LSTM
ax11 = plt.subplot(4, 3, 11)
lstm_errors = lstm_preds_test - y_test.flatten()
ax11.hist(lstm_errors, bins=30, alpha=0.7, edgecolor='black')
ax11.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax11.set_xlabel('Prediction Error', fontsize=10)
ax11.set_ylabel('Frequency', fontsize=10)
ax11.set_title(f'LSTM Error (MAE={lstm_metrics["MAE"]:.4f})', fontsize=12, fontweight='bold')
ax11.grid(True, alpha=0.3, axis='y')

# Plot 12: Model comparison
ax12 = plt.subplot(4, 3, 12)
models = ['RNN', 'LSTM', 'GRU']
test_losses_final = [rnn_test_losses[-1], lstm_test_losses[-1], gru_test_losses[-1]]
r2_scores = [rnn_metrics['R²'], lstm_metrics['R²'], gru_metrics['R²']]

x_pos = np.arange(len(models))
width = 0.35

bars1 = ax12.bar(x_pos - width/2, test_losses_final, width, label='Test Loss', alpha=0.7)
ax12_twin = ax12.twinx()
bars2 = ax12_twin.bar(x_pos + width/2, r2_scores, width, label='R² Score', 
                      alpha=0.7, color='orange')

ax12.set_xlabel('Model', fontsize=10)
ax12.set_ylabel('Test Loss (MSE)', fontsize=10, color='blue')
ax12_twin.set_ylabel('R² Score', fontsize=10, color='orange')
ax12.set_title('Model Comparison', fontsize=12, fontweight='bold')
ax12.set_xticks(x_pos)
ax12.set_xticklabels(models)
ax12.tick_params(axis='y', labelcolor='blue')
ax12_twin.tick_params(axis='y', labelcolor='orange')
ax12.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
#plt.savefig('/mnt/user-data/outputs/rnn_ode_results.png', dpi=150, bbox_inches='tight')
print("\n✓ Plot saved to: rnn_ode_results.png")

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
print(f"  Hidden size: {hidden_size}")
print(f"  Num layers: {num_layers}")
print(f"  Epochs: {epochs}")

print(f"\nPerformance (Test Set):")
print(f"  {'Model':<10s} {'MSE':<12s} {'RMSE':<12s} {'MAE':<12s} {'R²':<10s}")
print(f"  {'-'*58}")
print(f"  {'RNN':<10s} {rnn_metrics['MSE']:<12.6f} {rnn_metrics['RMSE']:<12.6f} {rnn_metrics['MAE']:<12.6f} {rnn_metrics['R²']:<10.6f}")
print(f"  {'LSTM':<10s} {lstm_metrics['MSE']:<12.6f} {lstm_metrics['RMSE']:<12.6f} {lstm_metrics['MAE']:<12.6f} {lstm_metrics['R²']:<10.6f}")
print(f"  {'GRU':<10s} {gru_metrics['MSE']:<12.6f} {gru_metrics['RMSE']:<12.6f} {gru_metrics['MAE']:<12.6f} {gru_metrics['R²']:<10.6f}")

print("\n✓ All models successfully trained on ODE solution data!")
print("="*70)
