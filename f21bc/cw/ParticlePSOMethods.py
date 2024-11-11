import numpy as np

class Particle:
    def __init__(self, dimensions):
        self.position = np.random.uniform(-1, 1, dimensions)
        self.velocity = np.random.uniform(-0.1, 0.1, dimensions)
        self.best_position = self.position.copy()
        self.best_error = float('inf')
        self.current_error = float('inf')

    def update_velocity(self, global_best_position, w=0.5, c1=1.5, c2=1.5):
        inertia = w * self.velocity
        cognitive = c1 * np.random.rand() * (self.best_position - self.position)
        social = c2 * np.random.rand() * (global_best_position - self.position)
        self.velocity = inertia + cognitive + social

    def update_position(self):
        self.position += self.velocity

class PSO:
    def __init__(self, n_particles, dimensions, fitness_function, n_iterations=10):
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.fitness_function = fitness_function
        self.n_iterations = n_iterations
        self.particles = [Particle(dimensions) for _ in range(n_particles)]
        self.global_best_position = np.random.uniform(-1, 1, dimensions)
        self.global_best_error = float('inf')

    def optimize(self):
        for _ in range(self.n_iterations):
            for particle in self.particles:
                particle.current_error = self.fitness_function(particle.position)
                if particle.current_error < particle.best_error:
                    particle.best_error = particle.current_error
                    particle.best_position = particle.position.copy()

                if particle.current_error < self.global_best_error:
                    self.global_best_error = particle.current_error
                    self.global_best_position = particle.position.copy()

            for particle in self.particles:
                particle.update_velocity(self.global_best_position)
                particle.update_position()

        return self.global_best_position, self.global_best_error

