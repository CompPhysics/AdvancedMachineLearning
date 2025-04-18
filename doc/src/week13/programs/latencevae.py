import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Define the VAE
class VAE(nn.Module):
   def __init__(self, input_dim=784, hidden_dim=400, latent_dim=2):
       super(VAE, self).__init__()
       self.fc1 = nn.Linear(input_dim, hidden_dim)
       self.fc_mu = nn.Linear(hidden_dim, latent_dim)
       self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
       self.fc3 = nn.Linear(latent_dim, hidden_dim)
       self.fc4 = nn.Linear(hidden_dim, input_dim)

   def encode(self, x):
       h1 = F.relu(self.fc1(x))
       return self.fc_mu(h1), self.fc_logvar(h1)

   def reparameterize(self, mu, logvar):
       std = torch.exp(0.5 * logvar)
       eps = torch.randn_like(std)
       return mu + eps * std

   def decode(self, z):
       h3 = F.relu(self.fc3(z))
       return torch.sigmoid(self.fc4(h3))

   def forward(self, x):
       mu, logvar = self.encode(x.view(-1, 784))
       z = self.reparameterize(mu, logvar)
       return self.decode(z), mu, logvar

# Loss function
def loss_function(recon_x, x, mu, logvar):
   BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
   KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
   return BCE + KLD

# Prepare data
transform = transforms.ToTensor()
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training function
def train(epoch):
   model.train()
   train_loss = 0
   for batch_idx, (data, _) in enumerate(train_loader):
       data = data.to(device)
       optimizer.zero_grad()
       recon_batch, mu, logvar = model(data)
       loss = loss_function(recon_batch, data, mu, logvar)
       loss.backward()
       train_loss += loss.item()
       optimizer.step()
   print(f"Epoch {epoch}: Avg Loss: {train_loss / len(train_loader.dataset):.4f}")

# Run training
for epoch in range(1, 11):
   train(epoch)

# Reconstruction Visualization
model.eval()
with torch.no_grad():
   data, _ = next(iter(test_loader))
   data = data.to(device)
   recon_batch, _, _ = model(data)

   n = 8
   comparison = torch.cat([data[:n], recon_batch.view(-1, 1, 28, 28)[:n]])
   comparison = comparison.cpu()

   plt.figure(figsize=(12, 3))
   for i in range(n):
       plt.subplot(2, n, i + 1)
       plt.imshow(comparison[i][0], cmap='gray')
       plt.axis('off')
       plt.subplot(2, n, i + 1 + n)
       plt.imshow(comparison[i + n][0], cmap='gray')
       plt.axis('off')
   plt.suptitle("Top: Original | Bottom: Reconstructed")
   plt.show()

# Latent Space Visualization
model.eval()
z_list = []
label_list = []

with torch.no_grad():
   for data, labels in test_loader:
       data = data.to(device)
       mu, _ = model.encode(data.view(-1, 784))
       z_list.append(mu.cpu())
       label_list.append(labels)

z = torch.cat(z_list).numpy()
labels = torch.cat(label_list).numpy()

plt.figure(figsize=(8, 6))
scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10', alpha=0.7, s=10)
plt.colorbar(scatter, ticks=range(10))
plt.title("2D Latent Space of MNIST")
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.grid(True)
plt.show()
