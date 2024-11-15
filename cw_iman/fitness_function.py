import numpy as np
from ann import ANN

def fitness_function(params, ann, dataset):
    #load training set and test score
    x, y = dataset
    ann.set_parameters(params)
    predictions = ann.forward(x.T)  # Forward pass through ANN

    #calculate accuracy
    predicted_classes = (predictions > 0.5).astype(int)
    accuracy = np.mean(predicted_classes == y.reshape(-1, 1))
    return accuracy

