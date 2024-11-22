import pandas as pd
import numpy as np
import random

class activationFunction:
    def logisticFunction(x):
        return 1 / (1 + np.exp(-x))

    def reluFunction(x):
        return np.maximum(0, x)

    def hyperbolicFunction(x):
        return np.tanh(x)

    def leakyReLU(x, alpha=0.01):
        return np.maximum(x, alpha * x)

    def linearFunction(x):  # Identity function for regression output
        return x
class ArtificialNeuralNetwork:
    def __init__(self, layerSize, activationFunction):
        self.layerSize = layerSize
        self.activationFunction = activationFunction
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i]) for i in range(len(layer_sizes) - 1)]
        self.biases = [np.random.randn(1, layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]

    def visualize_layers(self):
        for i in range(len(self.layerSize)-1):
            print("LAYER COUNT:  ", i,"    NODES: ",self.layerSize[i])
            print(f"  Weights Shape:   ", self.weights[i].shape)
            print(f"  Biases Shape:    ",self.biases[i].shape )
            print(f"  Weights= : {self.weights[i]}")
            print(f"  Biases= {self.biases[i]}")

    def forwardPropagation(self, x):
        output = x
        for i in range(len(self.weights)):
            matrix_total = np.dot(output, self.weights[i]) + self.biases[i]
            output = self.activationFunction[i](matrix_total)
        return output