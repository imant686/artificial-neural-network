import math
import numpy as np

class activationFunction:
    def logisticFunction(x):
        f = (1 + (math.e ** -x)) ** -1
        return f

    def reLuFunction(x):
        f = np.maximum(0,x)
        return f

    def hyperbolicFunction(x):
        f = math.tanh(x)
        return f

    def leakyReLU(x):
        alpha=0.01
        f = np.maximum(x, x * alpha)
        return f

class ArtificialNeuralNetwork:
    def __init__(self, layer_sizes, activation_functions):
        if len(activation_functions) != len(layer_sizes) - 1:
            raise ValueError("The number of activation functions needs to be number of layers less  by 1")

        self.layer_sizes = layer_sizes
        self.activation_functions = activation_functions

        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1 for i in range(len(layer_sizes) - 1)]
        self.biases = [np.random.randn(1, size) * 0.1 for size in layer_sizes[1:]]

    def forward(self, x):
        output = x
        for i, (weights, biases, activation_func) in enumerate(zip(self.weights, self.biases, self.activation_functions)):
            output = activation_func(output @ weights + biases)
        return output
