import numpy as np
from ann import ArtificialNeuralNetwork

# Initialising the particle object
class Particle:
    def __init__(self,vectorSize):
            
            self.particlePosition=np.random.rand(vectorSize) #initial position of the particle
            self.particleVelocity=np.random.rand(vectorSize) #initial velocity of the particle
        
            self.bestPosition=np.copy(self.particlePosition)
            self.informants=[]   # array to store all the informants of the particle

# Code to convert the particle to an ANN
    def particleToAnn(particle, annLayers, activationFunctions):
        
        neuralNetwork = ArtificialNeuralNetwork(layerSize=annLayers, activationFunction=activationFunctions)
        weightBiasIndexCount = 0
        
        for i in range(len(annLayers) - 1):
            # input for each neuron layer
            prevValue = annLayers[i]
            
            # output for each neuron layer
            nextValue = annLayers[i + 1]
            
            # mutliplying the layer counts
            weightRange = prevValue * nextValue
            
            # calculate weights
            weight = particle.particlePosition[weightBiasIndexCount:weightBiasIndexCount + weightRange].reshape((prevValue, nextValue))
            
            weightBiasIndexCount += weightRange
            biases = particle.particlePosition[weightBiasIndexCount:weightBiasIndexCount + nextValue].reshape((1, nextValue))
            weightBiasIndexCount += nextValue
            
            # setting the activationFunctions for the particle's ANN
            activation = activationFunctions[i]
            
            # setting the weights and biases for the particle's ANN
            neuralNetwork.weights[i] = weight
            neuralNetwork.biases[i] = biases
        return neuralNetwork

    def assessFitness(particle, dataset, annLayers, activationFunctions, loss_function):
        x, y = dataset
        ann = particleToAnn(particle, annLayers, activationFunctions)  # converting the ann to a particle
        predictions = ann.forwardPropagation(x)  # perform forward propagation 
        loss = loss_function.evaluate(predictions, y.reshape(-1, 1))  # calculate the loss

        return loss
