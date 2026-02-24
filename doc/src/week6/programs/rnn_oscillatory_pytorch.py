#!/usr/bin/env python3
"""
Recurrent Neural Network for Learning Oscillatory Functions
Using PyTorch

Features:
- Simple RNN, LSTM, and GRU implementations
- Training on sine/cosine functions
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

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# PART I: DATA GENERATION
# ============================================================================

def generate_oscillatory_data(func_type='sine', n_points=1000, noise_level=0.0):
    """
    Generate oscillatory time series data.
    
    Parameters:
    -----------
    func_type : str
        'sine', 'cosine', 'combined', or 'damped'
    n_points : int
        Number of time points
    noise_level : float
        Standard deviation of Gaussian noise
    
    Returns:
    --------
    t : ndarray
        Time values
    y : ndarray
        Function values
    """
    t = np.linspace(0, 10 * np.pi, n_points)
    
    if func_type == 'sine':
        y = np.sin(t)
    elif func_type == 'cosine':
        y = np.cos(t)
    elif func_type == 'combined':
        # Sum of two frequencies
        y = np.sin(t) + 0.5 * np.sin(2 * t)
    elif func_type == 'damped':
        # Damped oscillation
        y = np.exp(-t / 10) * np.sin(t)
    else:
        raise ValueError(f"Unknown function type: {func_type}")
    
    # Add noise
    if noise_level > 0:
        y += np.random.normal(0, noise_level, size=y.shape)
    
    return t, y

def create_sequences(data, seq_length, pred_length=1):
    """
    Create input-output sequences for RNN training.
    
    Parameters:
    -----------
    data : ndarray
        Time series data
    seq_length : int
        Length of input sequence
    pred_length : int
        Length of prediction (future steps)
    
    Returns:
    --------
    X : ndarray (n_sequences, seq_length)
        Input sequences
    y : ndarray (n_sequences, pred_length)
        Target sequences
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length - pred_length + 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + pred_length])
    
    return np.array(X), np.array(y)

# Generate data
print("\n" + "="*70)
print("GENERATING OSCILLATORY DATA")
print("="*70)

t, y = generate_oscillatory_data(func_type='sine', n_points=1000, noise_level=0.05)

print(f"\nData generated:")
print(f"  Function: sine wave")
print(f"  Points: {len(t)}")
print(f"  Time range: [{t[0]:.2f}, {t[-1]:.2f}]")
print(f"  Value range: [{y.min():.2f}, {y.max():.2f}]")

# Create sequences
seq_length = 50  # Use 50 past points to predict
pred_length = 1  # Predict 1 point ahead

X, y_seq = create_sequences(y, seq_length, pred_length)

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
# PART II: PYTORCH DATASET
# ============================================================================

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series."""
    
    def __init__(self, X, y):
        # Convert to PyTorch tensors
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
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"\nDataLoaders created:")
print(f"  Batch size: {batch_size}")
print(f"  Train batches: {len(train_loader)}")
print(f"  Test batches: {len(test_loader)}")

# ============================================================================
# PART III: RNN MODELS
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
        # x shape: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.rnn(x, h0)
        
        # Take the last time step
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
# PART IV: TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward pass
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        
        # Backward pass
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
# PART V: TRAIN MODELS
# ============================================================================

print("\n" + "="*70)
print("TRAINING MODELS")
print("="*70)

# Model hyperparameters
hidden_size = 64
num_layers = 2
epochs = 100
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
# PART VI: GENERATE PREDICTIONS
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

# Generate predictions for all models
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
    
    # R² score
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)
    
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
# PART VII: VISUALIZATION
# ============================================================================

print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

fig = plt.figure(figsize=(18, 12))

# Plot 1: Original data
ax1 = plt.subplot(3, 3, 1)
ax1.plot(t, y, 'b-', linewidth=1, alpha=0.7, label='Sine wave')
ax1.axvline(x=t[train_size + seq_length], color='r', linestyle='--', 
            linewidth=2, label='Train/Test split')
ax1.set_xlabel('Time', fontsize=10)
ax1.set_ylabel('Value', fontsize=10)
ax1.set_title('Original Time Series', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: Training curves - RNN
ax2 = plt.subplot(3, 3, 2)
ax2.plot(rnn_train_losses, 'b-', linewidth=2, label='Train')
ax2.plot(rnn_test_losses, 'r-', linewidth=2, label='Test')
ax2.set_xlabel('Epoch', fontsize=10)
ax2.set_ylabel('Loss (MSE)', fontsize=10)
ax2.set_title('Simple RNN: Training Curves', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

# Plot 3: Training curves - LSTM
ax3 = plt.subplot(3, 3, 3)
ax3.plot(lstm_train_losses, 'b-', linewidth=2, label='Train')
ax3.plot(lstm_test_losses, 'r-', linewidth=2, label='Test')
ax3.set_xlabel('Epoch', fontsize=10)
ax3.set_ylabel('Loss (MSE)', fontsize=10)
ax3.set_title('LSTM: Training Curves', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')

# Plot 4: Predictions - Simple RNN
ax4 = plt.subplot(3, 3, 4)
train_indices = np.arange(seq_length, seq_length + len(rnn_preds_train))
test_indices = np.arange(seq_length + len(rnn_preds_train), 
                         seq_length + len(rnn_preds_train) + len(rnn_preds_test))
ax4.plot(train_indices, y_train.flatten(), 'b-', linewidth=1, alpha=0.5, label='Train True')
ax4.plot(train_indices, rnn_preds_train, 'g-', linewidth=1, label='Train Pred')
ax4.plot(test_indices, y_test.flatten(), 'r-', linewidth=1, alpha=0.5, label='Test True')
ax4.plot(test_indices, rnn_preds_test, 'orange', linewidth=1, label='Test Pred')
ax4.set_xlabel('Time Step', fontsize=10)
ax4.set_ylabel('Value', fontsize=10)
ax4.set_title('Simple RNN: Predictions', fontsize=12, fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# Plot 5: Predictions - LSTM
ax5 = plt.subplot(3, 3, 5)
ax5.plot(train_indices, y_train.flatten(), 'b-', linewidth=1, alpha=0.5, label='Train True')
ax5.plot(train_indices, lstm_preds_train, 'g-', linewidth=1, label='Train Pred')
ax5.plot(test_indices, y_test.flatten(), 'r-', linewidth=1, alpha=0.5, label='Test True')
ax5.plot(test_indices, lstm_preds_test, 'orange', linewidth=1, label='Test Pred')
ax5.set_xlabel('Time Step', fontsize=10)
ax5.set_ylabel('Value', fontsize=10)
ax5.set_title('LSTM: Predictions', fontsize=12, fontweight='bold')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# Plot 6: Predictions - GRU
ax6 = plt.subplot(3, 3, 6)
ax6.plot(train_indices, y_train.flatten(), 'b-', linewidth=1, alpha=0.5, label='Train True')
ax6.plot(train_indices, gru_preds_train, 'g-', linewidth=1, label='Train Pred')
ax6.plot(test_indices, y_test.flatten(), 'r-', linewidth=1, alpha=0.5, label='Test True')
ax6.plot(test_indices, gru_preds_test, 'orange', linewidth=1, label='Test Pred')
ax6.set_xlabel('Time Step', fontsize=10)
ax6.set_ylabel('Value', fontsize=10)
ax6.set_title('GRU: Predictions', fontsize=12, fontweight='bold')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

# Plot 7: Test error distribution - RNN
ax7 = plt.subplot(3, 3, 7)
rnn_errors = rnn_preds_test - y_test.flatten()
ax7.hist(rnn_errors, bins=30, alpha=0.7, edgecolor='black')
ax7.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax7.set_xlabel('Prediction Error', fontsize=10)
ax7.set_ylabel('Frequency', fontsize=10)
ax7.set_title(f'RNN Error (MAE={rnn_metrics["MAE"]:.4f})', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')

# Plot 8: Test error distribution - LSTM
ax8 = plt.subplot(3, 3, 8)
lstm_errors = lstm_preds_test - y_test.flatten()
ax8.hist(lstm_errors, bins=30, alpha=0.7, edgecolor='black')
ax8.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax8.set_xlabel('Prediction Error', fontsize=10)
ax8.set_ylabel('Frequency', fontsize=10)
ax8.set_title(f'LSTM Error (MAE={lstm_metrics["MAE"]:.4f})', fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3, axis='y')

# Plot 9: Model comparison
ax9 = plt.subplot(3, 3, 9)
models = ['RNN', 'LSTM', 'GRU']
test_losses_final = [rnn_test_losses[-1], lstm_test_losses[-1], gru_test_losses[-1]]
r2_scores = [rnn_metrics['R²'], lstm_metrics['R²'], gru_metrics['R²']]

x_pos = np.arange(len(models))
width = 0.35

bars1 = ax9.bar(x_pos - width/2, test_losses_final, width, label='Test Loss', alpha=0.7)
ax9_twin = ax9.twinx()
bars2 = ax9_twin.bar(x_pos + width/2, r2_scores, width, label='R² Score', 
                     alpha=0.7, color='orange')

ax9.set_xlabel('Model', fontsize=10)
ax9.set_ylabel('Test Loss (MSE)', fontsize=10, color='blue')
ax9_twin.set_ylabel('R² Score', fontsize=10, color='orange')
ax9.set_title('Model Comparison', fontsize=12, fontweight='bold')
ax9.set_xticks(x_pos)
ax9.set_xticklabels(models)
ax9.tick_params(axis='y', labelcolor='blue')
ax9_twin.tick_params(axis='y', labelcolor='orange')
ax9.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
plt.savefig('rnn_oscillatory_results.png', dpi=150, bbox_inches='tight')
print("\n✓ Plot saved to: rnn_oscillatory_results.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"\nData:")
print(f"  Total samples: {len(X)}")
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

print("\n✓ All models successfully trained on oscillatory functions!")
print("="*70)
