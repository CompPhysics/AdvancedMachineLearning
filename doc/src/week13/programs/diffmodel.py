import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# === Hyperparameters ===
T = 300  # Number of diffusion steps
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Beta schedule (linear) ===
betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

# === Forward diffusion ===
def forward_diffusion_sample(x_0, t, noise=None):
   if noise is None:
       noise = torch.randn_like(x_0)
   sqrt_alpha = sqrt_alphas_cumprod[t][:, None, None, None]
   sqrt_one_minus_alpha = sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
   return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise, noise

# === Simple CNN for denoising ===
class SimpleUNet(nn.Module):
   def __init__(self):
       super().__init__()
       self.net = nn.Sequential(
           nn.Conv2d(2, 32, 3, padding=1),
           nn.ReLU(),
           nn.Conv2d(32, 64, 3, padding=1),
           nn.ReLU(),
           nn.Conv2d(64, 32, 3, padding=1),
           nn.ReLU(),
           nn.Conv2d(32, 1, 3, padding=1),
       )

   def forward(self, x, t):
       t_emb = t[:, None, None, None].float() / T  # normalize timestep
       t_emb = t_emb.expand(-1, 1, 28, 28)
       x_input = torch.cat([x, t_emb], dim=1)
       return self.net(x_input)

# === Data ===
transform = transforms.Compose([
   transforms.ToTensor(),
   transforms.Lambda(lambda x: (x - 0.5) * 2),  # scale to [-1, 1]
])
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

# === Model, optimizer ===
model = SimpleUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# === Training loop ===
def train(epochs=10):
   model.train()
   for epoch in range(epochs):
       pbar = tqdm(loader)
       for batch, _ in pbar:
           batch = batch.to(device)
           t = torch.randint(0, T, (batch.size(0),), device=device).long()
           x_noisy, noise = forward_diffusion_sample(batch, t)
           noise_pred = model(x_noisy, t)
           loss = F.mse_loss(noise_pred, noise)

           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           pbar.set_description(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

# === Sampling loop ===
@torch.no_grad()
def sample():
   model.eval()
   img = torch.randn((16, 1, 28, 28), device=device)
   for t in reversed(range(T)):
       t_batch = torch.full((img.shape[0],), t, device=device, dtype=torch.long)
       noise_pred = model(img, t_batch)
       beta = betas[t]
       alpha = alphas[t]
       alpha_cumprod = alphas_cumprod[t]
       coef1 = 1 / torch.sqrt(alpha)
       coef2 = (1 - alpha) / torch.sqrt(1 - alpha_cumprod)
       if t > 0:
           noise = torch.randn_like(img)
       else:
           noise = 0
       img = coef1 * (img - coef2 * noise_pred) + torch.sqrt(beta) * noise
   return img

# === Plotting generated samples ===
def show_samples(imgs):
   imgs = imgs.cpu().clamp(-1, 1)
   imgs = (imgs + 1) / 2  # back to [0, 1]
   grid = torch.cat([img for img in imgs], dim=2).squeeze()
   plt.figure(figsize=(12, 2))
   plt.imshow(grid, cmap="gray")
   plt.axis('off')
   plt.title("Generated Samples")
   plt.show()

# === Run training and generate ===
if __name__ == "__main__":
   train(epochs=10)
   samples = sample()
   show_samples(samples)
