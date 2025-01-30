import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Hyperparameters
batch_size = 128
epochs = 20
latent_dim = 20
learning_rate = 1e-3

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Define the VAE model
class VAE(nn.Module):
   def __init__(self, latent_dim):
       super(VAE, self).__init__()
       self.latent_dim = latent_dim

       # Encoder
       self.encoder = nn.Sequential(
           nn.Linear(784, 512),
           nn.ReLU(),
           nn.Linear(512, 256),
           nn.ReLU(),
           nn.Linear(256, 2 * latent_dim)  # Mean and log variance
       )

       # Decoder
       self.decoder = nn.Sequential(
           nn.Linear(latent_dim, 256),
           nn.ReLU(),
           nn.Linear(256, 512),
           nn.ReLU(),
           nn.Linear(512, 784),
           nn.Sigmoid()  # Output is between 0 and 1
       )

   def reparameterize(self, mu, logvar):
       std = torch.exp(0.5 * logvar)
       eps = torch.randn_like(std)
       return mu + eps * std

   def forward(self, x):
       # Flatten the input
       x = x.view(-1, 784)

       # Encode
       h = self.encoder(x)
       mu, logvar = h[:, :self.latent_dim], h[:, self.latent_dim:]

       # Reparameterization trick
       z = self.reparameterize(mu, logvar)

       # Decode
       x_recon = self.decoder(z)
       return x_recon, mu, logvar

# Loss function
def loss_function(recon_x, x, mu, logvar):
   BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
   KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
   return BCE + KLD
# Initialize the VAE
model = VAE(latent_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
   model.train()
   train_loss = 0
   for batch_idx, (data, _) in enumerate(train_loader):
       optimizer.zero_grad()
       recon_batch, mu, logvar = model(data)
       loss = loss_function(recon_batch, data, mu, logvar)
       loss.backward()
       train_loss += loss.item()
       optimizer.step()

   print(f'Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset):.4f}')

   # Save generated images
   with torch.no_grad():
       sample = torch.randn(64, latent_dim)
       sample = model.decoder(sample).cpu()
       save_image(sample.view(64, 1, 28, 28), f'sample_epoch_{epoch + 1}.png')

print("Training complete!")
