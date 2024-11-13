from Particle import Particle
class ParticleSwarmOptimisation:
    # initialise variables, use init function: constructor method
    def __init__(self,swarmSize,alpha,beta,delta,omega,jumpSize,informantCount,vectorSize):
        self.swarmSize = swarmSize
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.omega = omega
        self.jumpSize = jumpSize
        self.informantCount = informantCount
        self.vectorSize = vectorSize
        self.global_best = None
        self.global_best_fitness = float('inf')


        # assign informants
        def initInformants(informantCount,particleArray):
            pass

        # Example of assessFitness
        def assessFitness(particle, X, y, loss_function,predictions):
            # Implement forward propagation for the ANN represented by this particle
            # Use particle's position as ANN weights/biases
            # Calculate the loss (or error) using the provided loss_function
            # Return the computed fitness
            predictions = particle.forward_prop(X)  # Suppose each particle has a forward_prop method
            fitness = loss_function(predictions, y)
            return fitness

        
        def psoOptimisation(swarmSize,alpha,beta,delta,omega,jumpSize,informantCount,vectorSize):
            # stores all of the particles
            particleArray=[]
            for i in range(swarmSize):
                particleArray.append(Particle(vectorSize))
            best=None
            #initialising informants for the particles
            '''write code'''
            while(True): #will change to do while loop
                # compare fitness
                for p in particleArray:
                    particleFitness=assessFitness(p)
                    bestFitness=assessFitness(best)
                    if best is None or particleFitness<bestFitness:
                        best=p
                for p in particleArray:
                    previousBest=p.bestPosition
                    #informantsBest=
                    allBest=best.bestPosition








