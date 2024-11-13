import numpy as np

class Activation:
    def evaluate(self, x):
        pass

    def derivative(self, x):
        pass

class Logistic(Activation):
    def evaluate(self, x):
        f = 1 / (1 + np.exp(-x))
        return f

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

class leakyReLU(Activation):
    def evaluate(self, x, alpha=0.01):
       return np.where(x > 0, x, alpha * x)

    def derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

class Elu(Activation):
    def evaluate(self, x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    def derivative(self, x, alpha=0.01):
       return np.where(x > 0, 1, alpha * np.exp(x))

class Layer:
    def __init__(self, nodes, inputs, activationFunction):
        self.nodes = nodes
        self.weights = np.random.randn(inputs, nodes) * 0.1
        self.biases = np.zeros((1, nodes))
        self.activationFunction = activationFunction

    def forward(self, input):
        self.input = input
        z = np.dot(input, self.weights) + self.biases
        self.output = self.activationFunction.evaluate(z)
        return self.output

#defining loss functions
#class Loss:
# get dimensions as well


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

class ANNBuilder:
    def build(nb_layers, list_nb_nodes, list_functions): 
        ann = Network()
        for i in range(nb_layers):
            layer = Layer(list_nb_nodes[i], list_functions[i]) 
            ann.append(layer)
            return ann