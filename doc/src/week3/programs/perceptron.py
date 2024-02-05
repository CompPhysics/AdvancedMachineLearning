# import necessary packages
import numpy as np
import matplotlib.pyplot as plt

def feed_forward(x):
    # weighted sum of inputs to the output layer
    z_o = x*output_weights + output_bias
    # Output from output node (one node only)
    # Here the output is equal to the input
    a_o = z_o
    return a_o

def backpropagation(x, y):
    a_o = feed_forward(x)
    # derivative of cost function
    derivative_c = a_o - y
    # the variable delta in the equations, note that output a_0 = z_0, its derivatives wrt z_o is thus 1
    delta_o = derivative_c
    # gradients for the output layer
    output_weights_gradient = delta_o*x
    output_bias_gradient = delta_o
    # The cost function is 0.5*(a_o-y)^2. This gives a measure of the error for each iteration
    return output_weights_gradient, output_bias_gradient

# ensure the same random numbers appear every time
np.random.seed(0)
# Input variable
x = 4.0
# Target values
y = 2*x+1.0

# Defining the neural network
n_inputs = 1
n_outputs = 1
# Initialize the network
# weights and bias in the output layer
output_weights = np.random.randn()
output_bias = np.random.randn()

# implementing a simple gradient descent approach with fixed learning rate
eta = 0.01
for i in range(40):
    # calculate gradients from back propagation
    dWo, dBo = backpropagation(x, y)
    # update weights and biases
    output_weights -= eta * dWo
    output_bias -= eta * dBo
# our final prediction after training
ytilde = output_weights*x+output_bias
print(0.5*((ytilde-y)**2))
