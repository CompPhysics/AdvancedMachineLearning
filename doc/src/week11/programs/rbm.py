import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

# Hyperparameters
input_size = 784  # 28x28 images
hidden_size = 128
batch_size = 64
learning_rate = 0.01
num_epochs = 10
k = 1  # Number of Gibbs sampling steps

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Restricted Boltzmann Machine
class RBM(nn.Module):
   def __init__(self, visible_dim, hidden_dim):
       super(RBM, self).__init__()
       self.W = nn.Parameter(torch.randn(hidden_dim, visible_dim) * 0.01)
       self.h_bias = nn.Parameter(torch.zeros(hidden_dim))
       self.v_bias = nn.Parameter(torch.zeros(visible_dim))

   def sample_from_p(self, p):
       return torch.bernoulli(p)

   def v_to_h(self, v):
       p_h_given_v = torch.sigmoid(torch.matmul(v, self.W.t()) + self.h_bias)
       return p_h_given_v, self.sample_from_p(p_h_given_v)

   def h_to_v(self, h):
       p_v_given_h = torch.sigmoid(torch.matmul(h, self.W) + self.v_bias)
       return p_v_given_h, self.sample_from_p(p_v_given_h)

   def forward(self, v):
       # Gibbs sampling
       h_prob, h_sample = self.v_to_h(v)
       for _ in range(k):
           v_prob, v_sample = self.h_to_v(h_sample)
           h_prob, h_sample = self.v_to_h(v_sample)
       return v, v_prob

   def free_energy(self, v):
       vbias_term = torch.matmul(v, self.v_bias.unsqueeze(1)).squeeze()
       wx_b = torch.matmul(v, self.W.t()) + self.h_bias
       hidden_term = torch.sum(torch.log(1 + torch.exp(wx_b)), dim=1)
       return -hidden_term - vbias_term

# Initialize RBM
rbm = RBM(visible_dim=input_size, hidden_dim=hidden_size).to(device)

# Optimizer
optimizer = optim.SGD(rbm.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
   epoch_loss = 0
   for batch_idx, (data, _) in enumerate(train_loader):
       data = data.to(device)

       # Forward pass
       v, v_prob = rbm(data)

       # Compute loss (contrastive divergence)
       loss = rbm.free_energy(data) - rbm.free_energy(v_prob)
       loss = loss.mean()

       # Backward pass
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

       epoch_loss += loss.item()

   print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')

# Function to visualize reconstructed images
def visualize_reconstructions(rbm, data_loader, num_images=5):
   rbm.eval()
   with torch.no_grad():
       for batch_idx, (data, _) in enumerate(data_loader):
           data = data.to(device)
           _, v_prob = rbm(data)
           v_prob = v_prob.view(-1, 28, 28).cpu()
           data = data.view(-1, 28, 28).cpu()

           for i in range(num_images):
               plt.figure(figsize=(6, 3))
               plt.subplot(1, 2, 1)
               plt.imshow(data[i], cmap='gray')
               plt.title('Original')
               plt.axis('off')

               plt.subplot(1, 2, 2)
               plt.imshow(v_prob[i], cmap='gray')
               plt.title('Reconstructed')
               plt.axis('off')

               plt.show()
           break

# Visualize some reconstructed images
visualize_reconstructions(rbm, train_loader)


"""
### Explanation:
1. **RBM Class**:
  - The `RBM` class defines the weights (`W`), hidden biases (`h_bias`), and visible biases (`v_bias`).
  - It includes methods for sampling from probabilities (`sample_from_p`), converting visible to hidden units (`v_to_h`), and converting hidden to visible units (`h_to_v`).
  - The `forward` method performs Gibbs sampling to reconstruct the input.
  - The `free_energy` method computes the free energy of the RBM, which is used in the loss function.

2. **Training**:
  - The training loop uses Contrastive Divergence (CD-k) to update the weights and biases.
  - The loss is computed as the difference in free energy between the original data and the reconstructed data.

3. **Visualization**:
  - After training, the `visualize_reconstructions` function displays some original and reconstructed images to evaluate the RBM's performance.

### Notes:
- RBMs are unsupervised models, so we don't use labels during training.
- The number of Gibbs sampling steps (`k`) is typically small (e.g., 1 or 2) for efficiency.
- You can experiment with different hyperparameters like `hidden_size`, `learning_rate`, and `num_epochs` to improve performance.
"""
