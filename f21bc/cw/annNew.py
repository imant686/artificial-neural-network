import numpy as np

class Activation:
    def evaluate(self, x):
        pass

    def derivative(self, x):
        pass

class Sigmoid(Activation):
    def evaluate(self, x):
        f = 1 /(1 + (np.e ** -x))
        return f

    def derivative(self, x):
        f = 1 /(1 + (np.e ** -x))
        d=f * (1 - f)
        return d

class Tanh(Activation):
    def evaluate(self, x):
        f=np.tanh(x)
        return f

    def derivative(self, x):
        f = np.tanh(x)
        d = 1 - f ** 2
        return d


class Layer:
    def __init__(self,nodes,inputs,activationFunction):
        self.nodes=nodes
        self.weights=np.random.randn(inputs,nodes) * 0.1
        self.biases=np.zeros((1,nodes))
        self.activationFunction=activationFunction
    def forward(self,input):
        self.input=input
        z = np.dot(input,self.weights)+self.biases
        self.output= self.activationFunction.evaluate(z)
        return self.output

class Network:
    def __init__(self):
        self.layers = []
    def append(self, layer):
        self.layers.append(layer)
    def forward(self, data_in):
        out=data_in
        for layer in self.layers:
            out = layer.forward(out)
        return out

network = Network()

network.append(Layer(nodes=3, inputs=2, activationFunction=Sigmoid()))
network.append(Layer(nodes=1, inputs=3, activationFunction=Tanh()))

input_data = np.array([[0.5, -0.2]])
output = network.forward(input_data)

print( output)