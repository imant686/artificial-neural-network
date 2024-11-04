import math
import numpy as np

#size of layer =  no of nodes in each layer (input+output)
#if there are n layers in the network, there will be n - 1 activation functions becauise output layer does not require an activation function

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

    ''' extra functions:
        - maxout (needs weights and biases though)
        - if possible: elu
    '''
class ArtificialNeuralNetwork:
    def __init__(self, layerSize, activationFunction):

        if len(activationFunction) != len(layerSize) - 1:
            raise ValueError("The number of activation functions needs to be number of layers less  by 1")

        self.layerSize = layerSize
        self.activationFunction = activationFunction

        self.weights = [np.random.randn(self.layer_sizes[i], self.layerSize[i + 1]) for i in range(len(self.layerSize) - 1)]
        self.biases = [np.random.randn(1, size) for size in self.layerSize[1:]]

# x = input data
    def forwardPropagation(self, x):
        output = x
        for i in range(len(self.weights)):
            matrix_total = np.dot(output, self.weights[i]) + self.biases[i]
            output = self.activation_functions[i](matrix_total)
        return output

# Example usage
layer_sizes = [2, 3, 1]
activation_functions = [ActivationFunction.logisticFunction, ActivationFunction.hyperbolicFunction]

network = ArtificialNeuralNetwork(layer_sizes, activation_functions)
input_data = np.array([[0.5, -0.2]])
output = network.forwardPropagation(input_data)
print(output)