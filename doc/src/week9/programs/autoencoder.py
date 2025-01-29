import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Define the Autoencoder model
class Autoencoder(nn.Module):
   def __init__(self, input_dim, encoding_dim):
       super(Autoencoder, self).__init__()

       # Encoder
       self.encoder = nn.Sequential(
           nn.Linear(input_dim, 128),
           nn.ReLU(),
           nn.Linear(128, 64),
           nn.ReLU(),
           nn.Linear(64, encoding_dim),
           nn.ReLU()
       )

       # Decoder
       self.decoder = nn.Sequential(
           nn.Linear(encoding_dim, 64),
           nn.ReLU(),
           nn.Linear(64, 128),
           nn.ReLU(),
           nn.Linear(128, input_dim),
           nn.Sigmoid()  # Use sigmoid to ensure output is between 0 and 1
       )

   def forward(self, x):
       encoded = self.encoder(x)
       decoded = self.decoder(encoded)
       return decoded

# Hyperparameters
input_dim = 784  # Example for MNIST dataset (28x28 images)
encoding_dim = 32  # Size of the encoded representation
learning_rate = 0.001
num_epochs = 20
batch_size = 64

# Example data (replace with your dataset)
# For demonstration, we'll use random data
data = np.random.rand(1000, input_dim)  # 1000 samples, each with 784 features
dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
model = Autoencoder(input_dim, encoding_dim)
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
   for batch in dataloader:
       # Get the input data
       inputs = batch[0]

       # Zero the gradients
       optimizer.zero_grad()

       # Forward pass
       outputs = model(inputs)

       # Compute the loss
       loss = criterion(outputs, inputs)

       # Backward pass and optimize
       loss.backward()
       optimizer.step()

   print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete!")
