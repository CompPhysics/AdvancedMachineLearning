# Simple NN code using PyTorch on the MNIST dataset (this time the 28 x 28 set)
# The MNIST dataset is loaded using `torchvision.datasets`. The images are transformed to tensors and normalized.
# A simple feedforward neural network with one hidden layer is defined using `nn.Module`.
# The model is trained using the Adam optimizer and CrossEntropyLoss. The training loop iterates over the dataset for a specified number of epochs.
# Note that we don't include additional hyperparameters and the learning rate is set to 0.001.  
# After training, the model is evaluated on the test dataset to compute accuracy.
# The trained model's weights are saved to a file for later use.
# To do: add loops over hyperparameters and learning rates

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Hyperparameters
input_size = 784  # 28x28 images
hidden_size = 128
num_classes = 10
num_epochs = 5
batch_size = 64
learning_rate = 0.001

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

# Data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
   def __init__(self, input_size, hidden_size, num_classes):
       super(NeuralNet, self).__init__()
       self.fc1 = nn.Linear(input_size, hidden_size)
       self.relu = nn.ReLU()
       self.fc2 = nn.Linear(hidden_size, num_classes)  

   def forward(self, x):
       out = self.fc1(x)
       out = self.relu(out)
       out = self.fc2(out)
       return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
   for i, (images, labels) in enumerate(train_loader):  
       # Move tensors to the configured device
       images = images.reshape(-1, 28*28).to(device)
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
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
   correct = 0
   total = 0
   for images, labels in test_loader:
       images = images.reshape(-1, 28*28).to(device)
       labels = labels.to(device)
       outputs = model(images)
       _, predicted = torch.max(outputs.data, 1)
       total += labels.size(0)
       correct += (predicted == labels).sum().item()

   print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
