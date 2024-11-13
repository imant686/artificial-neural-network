import numpy 
class Particle:
    def __init__(self,vectorSize):
            self.particlePosition=numpy.random.rand(vectorSize)
            self.particleVelocity=numpy.random.rand(vectorSize)
            self.bestPosition=numpy.copy(self.particlePosition)


