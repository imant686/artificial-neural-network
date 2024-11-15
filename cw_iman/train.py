import numpy as np
from ann import ANN, logistic
from pso import PSO
from fitness_function import fitness_function

dataset = load_dataset("data_banknote_authentication.txt")

# run some random hyperparameters
# Run PSO
#pso = PSO(swarm_size, dimension, bounds, fitness, num_iterations)

pso.optimize()

print("Best fitness:", pso.global_best_fitness)
print("Best parameters:", pso.global_best_position)
