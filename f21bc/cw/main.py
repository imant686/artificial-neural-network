import numpy as np
from ANNMethods import Network, Layer, Logistic, Tanh, ReLU  # Import activation functions, Network, and Layer
from ParticlePSOMethods import PSO

# Step 1: Define the ANN architecture
def build_ann():
    network = Network()
    network.append(Layer(nodes=3, inputs=2, activation_function=Logistic()))
    network.append(Layer(nodes=1, inputs=3, activation_function=Tanh()))
    return network

# Step 2: Define the fitness function for PSO
def fitness_function(weights_biases):
    # Reshape weights_biases into the ANN's weight and bias structure
    network = build_ann()
    weight_idx = 0

    for layer in network.layers:
        num_weights = layer.weights.size
        layer.weights = weights_biases[weight_idx:weight_idx + num_weights].reshape(layer.weights.shape)
        weight_idx += num_weights
        num_biases = layer.biases.size
        layer.biases = weights_biases[weight_idx:weight_idx + num_biases].reshape(layer.biases.shape)
        weight_idx += num_biases

    # Example input and target output for testing
    input_data = np.array([[0.5, -0.2]])
    target_output = np.array([[0.4]])  # Adjust based on your specific regression task

    # Calculate network output and error
    network_output = network.forward(input_data)
    error = np.mean((network_output - target_output) ** 2)  # Mean Squared Error (MSE)

    return error

# Step 3: Determine the dimensions needed for PSO based on the ANN structure
def get_dimensions(network):
    total_params = sum(layer.weights.size + layer.biases.size for layer in network.layers)
    return total_params

# Step 4: Run PSO
if __name__ == "__main__":
    ann_network = build_ann()
    dimensions = get_dimensions(ann_network)  # Number of parameters to optimize
    pso = PSO(n_particles=20, dimensions=dimensions, fitness_function=fitness_function, n_iterations=100)

    # Run PSO to find optimal weights and biases
    best_position, best_error = pso.optimize()

    print("Best Position:", best_position)
    print("Best Error:", best_error)

    # Test the optimized ANN
    optimized_ann = build_ann()
    weight_idx = 0

    for layer in optimized_ann.layers:
        num_weights = layer.weights.size
        layer.weights = best_position[weight_idx:weight_idx + num_weights].reshape(layer.weights.shape)
        weight_idx += num_weights
        num_biases = layer.biases.size
        layer.biases = best_position[weight_idx:weight_idx + num_biases].reshape(layer.biases.shape)
        weight_idx += num_biases

    # Print the optimized output for test input data
    optimized_output = optimized_ann.forward(np.array([[0.5, -0.2]]))
    print("Optimized Network Output:", optimized_output)
