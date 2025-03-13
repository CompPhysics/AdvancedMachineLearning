"""
Key components:
1. **Data Handling**: Uses PyTorch DataLoader with MNIST dataset
2. **LSTM Architecture**:
  - Input sequence of 28 timesteps (image rows)
  - 128 hidden units in LSTM layer
  - Fully connected layer for classification
3. **Training**:
  - Cross-entropy loss
  - Adam optimizer
  - Automatic GPU utilization if available

This implementation typically achieves **97-98% accuracy** after 10 epochs. The main differences from the TensorFlow/Keras version:
- Explicit device management (CPU/GPU)
- Manual training loop
- Different data loading pipeline
- More explicit tensor reshaping

To improve performance, you could:
1. Add dropout regularization
2. Use bidirectional LSTM
3. Implement learning rate scheduling
4. Add batch normalization
5. Increase model capacity (more layers/units)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Hyperparameters
input_size = 28     # Number of features (pixels per row)
hidden_size = 128   # LSTM hidden state size
num_classes = 10    # Digits 0-9
num_epochs = 10     # Training iterations
batch_size = 64     # Batch size
learning_rate = 0.001

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST dataset
transform = transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

train_dataset = datasets.MNIST(root='./data',
                              train=True,
                              transform=transform,
                              download=True)

test_dataset = datasets.MNIST(root='./data',
                             train=False,
                             transform=transform)

train_loader = DataLoader(dataset=train_dataset,
                         batch_size=batch_size,
                         shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                        batch_size=batch_size,
                        shuffle=False)

# LSTM model
class LSTMModel(nn.Module):
   def __init__(self, input_size, hidden_size, num_classes):
       super(LSTMModel, self).__init__()
       self.hidden_size = hidden_size
       self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
       self.fc = nn.Linear(hidden_size, num_classes)

   def forward(self, x):
       # Reshape input to (batch_size, sequence_length, input_size)
       x = x.reshape(-1, 28, 28)

       # Forward propagate LSTM
       out, _ = self.lstm(x)  # out: (batch_size, seq_length, hidden_size)

       # Decode the hidden state of the last time step
       out = out[:, -1, :]
       out = self.fc(out)
       return out

# Initialize model
model = LSTMModel(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
total_step = len(train_loader)
for epoch in range(num_epochs):
   model.train()
   for i, (images, labels) in enumerate(train_loader):
       images = images.to(device)
       labels = labels.to(device)

       # Forward pass
       outputs = model(images)
       loss = criterion(outputs, labels)

       # Backward and optimize
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

       if (i+1) % 100 == 0:
           print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

   # Test the model
   model.eval()
   with torch.no_grad():
       correct = 0
       total = 0
       for images, labels in test_loader:
           images = images.to(device)
           labels = labels.to(device)
           outputs = model(images)
           _, predicted = torch.max(outputs.data, 1)
           total += labels.size(0)
           correct += (predicted == labels).sum().item()

       print(f'Test Accuracy: {100 * correct / total:.2f}%')

print('Training finished.')


