import numpy as np

class Activation:
    def evaluate(self, x):
        pass

    def derivative(self, x):
        pass

class Logistic(Activation):
    def evaluate(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        f = self.evaluate(x)
        return f * (1 - f)

class Tanh(Activation):
    def evaluate(self, x):
        return np.tanh(x)

    def derivative(self, x):
        f = np.tanh(x)
        return 1 - f ** 2

class ReLU(Activation):
    def evaluate(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x > 0, 1, 0)

class LeakyReLU(Activation):
    def evaluate(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

class Layer:
    def __init__(self, nodes, inputs, activation_function):
        self.nodes = nodes
        self.weights = np.random.randn(inputs, nodes) * 0.1
        self.biases = np.zeros((1, nodes))
        self.activation_function = activation_function

    def forward(self, input_data):
        z = input_data @ self.weights + self.biases
        return self.activation_function.evaluate(z)

class Network:
    def __init__(self):
        self.layers = []

    def append(self, layer):
        self.layers.append(layer)

    def forward(self, data_in):
        out = data_in
        for layer in self.layers:
            out = layer.forward(out)
        return out
