import pandas as pd
import numpy as np
import random
from particle import Particle


class ParticleSwarmOptimisation:
    # initialise the class variables
    def __init__(self, swarmSize, alpha, beta, delta, gamma, jumpSize, informantCount, vectorSize):
        self.swarmSize = swarmSize
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma =gamma
        self.jumpSize = jumpSize
        self.informantCount = informantCount
        self.vectorSize = vectorSize
        self.global_best = None
        self.global_best_fitness = float('inf')

    # initialise the informants for each particle
    def initInformants(self, informantCount, particleArray):
        for p in particleArray:
            # blank array for informants
            informants=[]
            # iterate every particle in the array again
            for p in particleArray:
                # to store every particle except itself
                potentialInformants=[]
                # to check if the array is the array itself
                for potInf in particleArray:
                    if potInf!=p:
                        # assign to potential particle informant
                        potentialInformants.append(potInf)
                for i in range(informantCount):
                    nformants.append(random.choice(potentialInformants))
                p.informants=informants  # assign all informants to the the particle

    def get_best_informant(self, particle, dataset, annLayers, activationFunctions, loss_function):
        bestInf = None
        bestFitnessInf = float('-inf')
        for i in particle.informants:
            # Assess the fitness of each informant
            fitness = assessFitness(i, dataset, annLayers, activationFunctions, loss_function)
            
            if fitness >  bestFitnessInf:
                bestFitnessInf = fitness
                bestInf = i
        return bestInf.particlePosition
        
    # Optimisation method
    def psoOptimisation(self, swarmSize, alpha, beta, gamma, jumpSize, informantCount, vectorSize,
                        dataset, annLayers, activationFunctions, loss_function, max_iterations=100):

        particleArray=[]
        # creating particles
        for i in range(swarmSize):
                particleArray.append(Particle(vectorSize))

        self.initInformants(informantCount, particleArray)
        best = None
        iteration = 0
        while iteration < max_iterations:
            # Update best particle
            for p in particleArray:
                # assessing fitness for p
                particleFitness = assessFitness(p, dataset, annLayers, activationFunctions, loss_function)
                # assigning the particle as the best
                if best is None or particleFitness < assessFitness(best, dataset, annLayers, activationFunctions, loss_function):
                    best = p
            # calculate the velocity
            for p in particleArray:
                # initialising the values
                previousBest = p.bestPosition
                informantsBest = self.get_best_informant(p, dataset, annLayers, activationFunctions, loss_function)
                allBest = best.bestPosition
                b = np.random.uniform(0.0, beta)
                c = np.random.uniform(0.0, gamma)
                d = np.random.uniform(0.0, delta)
                
                # calculate the new velocity
                newVelocity = (
                    alpha * p.particleVelocity +
                    b * (previousBest - p.particlePosition) +
                    c * (informantsBest - p.particlePosition) +
                    d * (allBest - p.particlePosition)
                )
                print("updated v:",  newVelocity)
                p.particleVelocity =  newVelocity
                p.particlePosition += jumpSize * newVelocity
                
            iteration += 1
        return best.particlePosition
