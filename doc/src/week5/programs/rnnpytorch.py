import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic sine wave data
t = torch.linspace(0, 4*np.pi, 1000)
data = torch.sin(t)

# Split data into training and validation
train_data = data[:800]
val_data = data[800:]

# Hyperparameters
seq_len = 20
batch_size = 32
hidden_size = 64
num_epochs = 100
learning_rate = 0.001

# Create dataset and dataloaders
class SineDataset(torch.utils.data.Dataset):
   def __init__(self, data, seq_len):
       self.data = data
       self.seq_len = seq_len

   def __len__(self):
       return len(self.data) - self.seq_len

   def __getitem__(self, idx):
       x = self.data[idx:idx+self.seq_len]
       y = self.data[idx+self.seq_len]
       return x.unsqueeze(-1), y  # Add feature dimension

train_dataset = SineDataset(train_data, seq_len)
val_dataset = SineDataset(val_data, seq_len)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define RNN model
class RNNModel(nn.Module):
   def __init__(self, input_size, hidden_size, output_size):
       super(RNNModel, self).__init__()
       self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
       self.fc = nn.Linear(hidden_size, output_size)

   def forward(self, x):
       out, _ = self.rnn(x)  # out: (batch_size, seq_len, hidden_size)
       out = out[:, -1, :]   # Take last time step output
       out = self.fc(out)
       return out

model = RNNModel(input_size=1, hidden_size=hidden_size, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
train_losses = []
val_losses = []

for epoch in range(num_epochs):
   model.train()
   epoch_train_loss = 0
   for x_batch, y_batch in train_loader:
       optimizer.zero_grad()
       y_pred = model(x_batch)
       loss = criterion(y_pred, y_batch.unsqueeze(-1))
       loss.backward()
       optimizer.step()
       epoch_train_loss += loss.item()

   # Validation
   model.eval()
   epoch_val_loss = 0
   with torch.no_grad():
       for x_val, y_val in val_loader:
           y_pred_val = model(x_val)
           val_loss = criterion(y_pred_val, y_val.unsqueeze(-1))
           epoch_val_loss += val_loss.item()

   # Calculate average losses
   train_loss = epoch_train_loss / len(train_loader)
   val_loss = epoch_val_loss / len(val_loader)
   train_losses.append(train_loss)
   val_losses.append(val_loss)

   print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Generate predictions
model.eval()
initial_sequence = train_data[-seq_len:].reshape(1, seq_len, 1)
predictions = []
current_sequence = initial_sequence.clone()

with torch.no_grad():
   for _ in range(len(val_data)):
       pred = model(current_sequence)
       predictions.append(pred.item())
       # Update sequence by removing first element and adding new prediction
       current_sequence = torch.cat([current_sequence[:, 1:, :], pred.unsqueeze(1)], dim=1)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(t[800:].numpy(), val_data.numpy(), label='True values')
plt.plot(t[800:].numpy(), predictions, label='Predictions')
plt.title('Sine Wave Prediction')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

"""
This code:

1. Generates a sine wave and splits it into training and validation sets
2. Creates a custom Dataset for sequence generation
3. Defines an RNN model with one RNN layer and a final linear layer
4. Trains the model using MSE loss and Adam optimizer
5. Evaluates on validation data during training
6. Generates predictions using recursive forecasting
7. Plots the results and training/validation loss curves

The model takes sequences of 20 previous values to predict the next value in the sine wave. The recursive prediction uses the model's own predictions to generate future values, showing how well it maintains the sine wave pattern over time.

You can adjust hyperparameters like:
- `seq_len`: Length of input sequences
- `hidden_size`: Number of hidden units in the RNN
- `num_epochs`: Number of training epochs
- `learning_rate`: Learning rate for optimizer
- `batch_size`: Number of samples per training batch

The final plots show:
1. The predicted vs actual sine wave for the validation period
2. The training and validation loss curves to monitor for overfitting
"""
