# import necessary packages
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def forwardpropagation(x):
    # weighted sum of inputs to the hidden layer
    z_1 = np.matmul(x, w_1) + b_1
    # activation in the hidden layer
    a_1 = sigmoid(z_1)
    # weighted sum of inputs to the output layer
    z_2 = np.matmul(a_1, w_2) + b_2
    a_2 = z_2
    return a_1, a_2

def backpropagation(x, y):
    a_1, a_2 = forwardpropagation(x)
    # parameter delta for the output layer, note that a_2=z_2 and its derivative wrt z_2 is just 1
    delta_2 = np.sum(a_2 - y)
    print(a_2)
    print(delta_2)
#    print(0.5*((a_2-y)**2))
    # delta for  the hidden layer
    delta_1 = np.matmul(delta_2, w_2.T) * a_1 * (1 - a_1)
    # gradients for the output layer
    output_weights_gradient = np.matmul(a_1.T, delta_2)
    output_bias_gradient = np.sum(delta_2, axis=0)
    # gradient for the hidden layer
    hidden_weights_gradient = np.matmul(x.T, delta_1)
    hidden_bias_gradient = np.sum(delta_1, axis=0)
    return output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient


# ensure the same random numbers appear every time
np.random.seed(0)
# Input variable
# Design matrix
X = np.array([ [1.0, 2.0], [3.0, 4.0], [5.0, 6.0],[7.0, 8.0]],dtype=np.float64)

# The XOR gate
y = np.array( [ 0, 1 ,1, 0])
# Defining the neural network, only scalars
n_inputs, n_features = X.shape
n_hidden_neurons = 2
n_outputs = 1

# Initialize the network
# weights and bias in the hidden layer
w_1 = np.random.randn(n_features, n_hidden_neurons)
b_1 = np.zeros(n_hidden_neurons) + 0.01
# weights and bias in the output layer
w_2 = np.random.randn(n_hidden_neurons, n_outputs)
b_2 = np.zeros(n_outputs) + 0.01
eta = 0.1
for i in range(100):
    # calculate gradients
    derivW2, derivB2, derivW1, derivB1 = backpropagation(X, y)
    # update weights and biases
    w_2 -= eta * derivW2
    b_2 -= eta * derivB2
    w_1 -= eta * derivW1
    b_1 -= eta * derivB1

