import pandas as pd
import numpy as np
import random

class activationFunction:
    def logisticFunction(x): #For calculating the sigmoid function that always returns a value between 0 to 1
        f=1 / (1 + np.exp(-x))
        return f

    def reluFunction(x):
        f=np.maximum(0, x)  #Returns a value between 0 to infinite
        return f

    def hyperbolicFunction(x):
        f=np.tanh(x)
        return f

    def leakyReLU(x, alpha=0.01):
        f= np.maximum(x, alpha * x)
        return f

    def linearFunction(x):
        return x

# Creates a neural network based on the layerSize and activationFunction entered
class ArtificialNeuralNetwork:
    def __init__(self, layerSize, activationFunction):
        self.layerSize=layerSize
        self.activationFunction = activationFunction
        self.weights=[np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i]) for i in range(len(layer_sizes) - 1)]
        self.biases=[np.random.randn(1, layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]

#Prints the layers and values of the neural network
    def visualizeLayers(self):
        for i in range(len(self.layerSize)-1):
            print("LAYER COUNT:  ", i,"    NODES: ",self.layerSize[i])
            print("  WEIGHT SHAPE:   ", self.weights[i].shape)
            print("  BIAS SHAPE:    ",self.biases[i].shape )
            print("  WEIGHTS :      ",self.weights[i])
            print("  BIASES        ",self.biases[i])

    def forwardPropagation(self, x):
        output=x
        for i in range(len(self.weights)):
            #Calculate the dot product of each value
            matrix_total = np.dot(output, self.weights[i]) + self.biases[i]
            output= self.activationFunction[i](matrix_total)
        return output
