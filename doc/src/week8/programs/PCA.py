import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

# Load MNIST dataset from OpenML
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data / 255.0  # Normalize pixel values to [0, 1]
y = mnist.target.astype(int)

# Apply PCA to reduce dimensions to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualize the first two principal components
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5, s=10)
plt.colorbar(scatter, label='Digit Label')
plt.title('PCA Projection of MNIST Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()
