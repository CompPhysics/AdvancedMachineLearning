import numpy as np

class BinaryBinaryRBM:
    def __init__(self, n_visible, n_hidden, learning_rate=0.1):
        """
        Initialize the RBM with given parameters.
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.weights = np.random.normal(0, 0.01, (n_visible, n_hidden))  # Weights
        self.visible_bias = np.zeros(n_visible)  # Bias for visible units
        self.hidden_bias = np.zeros(n_hidden)  # Bias for hidden units

    def sigmoid(self, x):
        """
        Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))

    def sample_hidden(self, v):
        """
        Sample hidden units given visible units.
        """
        activation = np.dot(v, self.weights) + self.hidden_bias
        prob_h = self.sigmoid(activation)
        return np.random.binomial(n=1, p=prob_h), prob_h

    def sample_visible(self, h):
        """
        Sample visible units given hidden units.
        """
        activation = np.dot(h, self.weights.T) + self.visible_bias
        prob_v = self.sigmoid(activation)
        return np.random.binomial(n=1, p=prob_v), prob_v

    def contrastive_divergence(self, v0, k=1):
        """
        Contrastive Divergence (CD-k) algorithm for training the RBM.
        """
        # Positive phase
        h0, prob_h0 = self.sample_hidden(v0)
        pos_associations = np.outer(v0, prob_h0)

        # Gibbs Sampling (k steps)
        v_k = v0
        for _ in range(k):
            h_k, _ = self.sample_hidden(v_k)
            v_k, _ = self.sample_visible(h_k)

        # Negative phase
        h_k, prob_h_k = self.sample_hidden(v_k)
        neg_associations = np.outer(v_k, prob_h_k)

        # Update weights and biases
        self.weights += self.learning_rate * (pos_associations - neg_associations)
        self.visible_bias += self.learning_rate * (v0 - v_k)
        self.hidden_bias += self.learning_rate * (prob_h0 - prob_h_k)

    def train(self, data, epochs=1000, batch_size=10):
        """
        Train the RBM using mini-batch gradient descent.
        """
        n_samples = data.shape[0]
        for epoch in range(epochs):
            np.random.shuffle(data)
            for i in range(0, n_samples, batch_size):
                batch = data[i:i+batch_size]
                for v in batch:
                    self.contrastive_divergence(v)
            if (epoch + 1) % 100 == 0:
                error = self.reconstruction_error(data)
                print(f"Epoch {epoch + 1}/{epochs} - Reconstruction Error: {error:.4f}")

    def reconstruction_error(self, data):
        """
        Compute reconstruction error for the dataset.
        """
        error = 0
        for v in data:
            _, prob_h = self.sample_hidden(v)
            _, prob_v = self.sample_visible(prob_h)
            error += np.linalg.norm(v - prob_v)
        return error / len(data)

    def reconstruct(self, v):
        """
        Reconstruct a visible vector after one pass through hidden units.
        """
        _, prob_h = self.sample_hidden(v)
        _, prob_v = self.sample_visible(prob_h)
        return prob_v

# Generate synthetic binary data
np.random.seed(42)
data = np.random.binomial(n=1, p=0.5, size=(100, 6))  # 100 samples, 6 visible units

# Initialize and train the RBM
rbm = BinaryBinaryRBM(n_visible=6, n_hidden=3, learning_rate=0.1)
rbm.train(data, epochs=1000, batch_size=10)

# Test the reconstruction
sample = np.array([1, 0, 1, 0, 1, 0])
reconstructed = rbm.reconstruct(sample)
print(f"\nOriginal: {sample}")
print(f"Reconstructed: {np.round(reconstructed, 2)}")
