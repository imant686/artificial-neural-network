from Particle import Particle
from ANNMethods import Layer, Network, Activation, ANNBuilder

def particleToAnn(particle,annLayers,activationFunctions):
    neuralNetwork=Network()
    weightBiasindexCount=0
    for i in range(annLayers):
        if i==0:
            prevValue=annLayers[i]
        prevValue=annLayers[i-1]
        nextValue=annLayers[i]

        weightRange=prevValue*nextValue
        weight=particle.particlePosition[weightBiasindexCount:weightBiasindexCount+weightRange].reshape((prevValue,nextValue))
        weightBiasindexCount+=weightRange
        biases=particle.particlePosition[weightBiasindexCount:weightBiasindexCount+nextValue].reshape((1,nextValue))
        position_idx+=nextValue
        activation = activationFunctions[i - 1]
        layer = Layer(nodes=nextValue,inputs=prevValue,activationFunction=activation)
        layer.weights = weight
        layer.biases = biases
        neuralNetwork.append(layer)
    return neuralNetwork
#must refactor this further