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

network = Network()

network.append(Layer(nodes=3, inputs=2, activationFunction=Logistic()))
network.append(Layer(nodes=1, inputs=3, activationFunction=Tanh()))

input_data = np.array([[0.5, -0.2]])

output = network.forward(input_data)
print("Network output:", output)

class Particle:
    def __init__(self, vectorSize, ):
            self.position = np.random.rand(vectorSize)
            self.velocity = np.random.rand(vectorSize)
            self.bestPosition = np.zeros(vectorSize)
            self.best # fitness
            self.bestErr

    def updateVelocity (self, bestPosition):

            cognitiveVelocity = c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social
'''
Update velocity:
    inertia weight
    social weight
    cognitive weight

    to get:
        cognitive
        social
        self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

Update position:
'''

#class PSO:
    #def
    #informants go in here
    # global values go in
    # fitness funciton goes in
    #testing