"""
Data Loading: Uses torchvision’s MNIST loader with basic normalization.
Noise Scheduler: Linearly increases noise over time steps.
Model: A small convolutional network inspired by a U-Net (without skip connections).
Forward Diffusion: Adds noise based on a given timestep.
Training Loop: Learns to predict the noise added at each step.
Sampling: Generates new images by reversing the diffusion process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Configurations
image_size = 28
batch_size = 64
num_steps = 1000
device = "cpu"
epochs = 1  # For demonstration

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)  # Scale to [-1, 1]
])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Linear noise schedule
beta_start, beta_end = 1e-4, 0.02
betas = torch.linspace(beta_start, beta_end, num_steps).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)

# Simple convolutional model (mini U-Net)
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(1, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, padding=1)
        self.dec1 = nn.ConvTranspose2d(64, 32, 3, padding=1)
        self.out = nn.ConvTranspose2d(32, 1, 3, padding=1)

    def forward(self, x, t):
        t_embed = t[:, None, None, None].float() / num_steps
        t_embed = t_embed.expand(x.shape)
        x = torch.cat([x, t_embed], dim=1)
        x1 = F.relu(self.enc1(x[:, :1]))  # Only image through conv
        x2 = F.relu(self.enc2(x1))
        x3 = F.relu(self.dec1(x2))
        return self.out(x3)

model = SimpleUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Forward diffusion process
def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_alpha_bar = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
    sqrt_one_minus = torch.sqrt(1 - alphas_cumprod[t])[:, None, None, None]
    return sqrt_alpha_bar * x0 + sqrt_one_minus * noise

# Training
def train():
    model.train()
    for epoch in range(epochs):
        for batch, (x, _) in enumerate(train_loader):
            x = x.to(device)
            t = torch.randint(0, num_steps, (x.shape[0],), device=device)
            noise = torch.randn_like(x)
            x_noisy = q_sample(x, t, noise)
            predicted_noise = model(x_noisy, t)
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch}, Loss: {loss.item():.4f}")

# Reverse sampling
@torch.no_grad()
def sample(model, n=8):
    model.eval()
    img = torch.randn(n, 1, image_size, image_size).to(device)
    for i in reversed(range(num_steps)):
        t = torch.full((n,), i, device=device, dtype=torch.long)
        predicted_noise = model(img, t)
        beta = betas[i]
        alpha = alphas[i]
        alpha_bar = alphas_cumprod[i]
        if i > 0:
            noise = torch.randn_like(img)
        else:
            noise = 0
        img = (1 / torch.sqrt(alpha)) * (img - beta / torch.sqrt(1 - alpha_bar) * predicted_noise) + torch.sqrt(beta) * noise
    return img

# Run
if __name__ == "__main__":
    train()
    samples = sample(model, n=8).cpu()
    samples = (samples + 1) / 2  # Convert back to [0,1]
    grid = torch.cat([s.squeeze(0) for s in samples], dim=1)
    plt.imshow(grid, cmap='gray')
    plt.axis('off')
    plt.show()
