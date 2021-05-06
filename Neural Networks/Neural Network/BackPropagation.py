'''
Author: Sergio Remigio
Date: 5/3/2021

'''
import csv
import math
import random


def get_initial_weights():
    x0 = [-1, 1]
    x1 = [-2, 2]
    x2 = [-3, 3]
    layer0 = list()
    layer0.append(x0)
    layer0.append(x1)
    layer0.append(x2)
    z10 = [-1, 1]
    z11 = [-2, 2]
    z12 = [-3, 3]
    layer1 = list()
    layer1.append(z10)
    layer1.append(z11)
    layer1.append(z12)
    z20 = [-1]
    z21 = [2]
    z22 = [-1.5]
    layer2 = list()
    layer2.append(z20)
    layer2.append(z21)
    layer2.append(z22)
    initial_weights = list()
    initial_weights.append(layer0)
    initial_weights.append(layer1)
    initial_weights.append(layer2)
    return initial_weights


def get_random_initial_weights(width, layers, feature_vector, b):
    """
    :param width:
    :param layers:
    :param feature_vector:
    :return: Weight vector. Will be converted into augmented weight vector.
             3D array with the following categories:
             1st dimension: All layers in Neural Network - 1
             2nd dimension: Neurons for the given layer
             3rd dimension: Weight vectors for the given neuron.
    """
    random_weights = []
    # Add random weights for hidden layers
    for layer in range(layers):
        # Create this layer weights and add the bias parameter
        layer_weights = []
        # Only if this is the input layer
        if layer == 0:
            for neuron in range(len(feature_vector) + 1):
                neuron_weights = []
                for parent in range(width):
                    neuron_weights.append(random.gauss(0, 1))
                layer_weights.append(neuron_weights)
        else:
            for neuron in range(width + 1):
                neuron_weights = []
                for parent in range(width):
                    neuron_weights.append(random.gauss(0, 1))
                layer_weights.append(neuron_weights)
        random_weights.append(layer_weights)
    # Finally add the output layer weights
    output_layer_weights = []
    for neuron in range(width + 1):
        neuron_weights = [random.gauss(0, 1)]
        output_layer_weights.append(neuron_weights)
    random_weights.append(output_layer_weights)
    return random_weights


def import_examples(file):
    """
    Copies file contents into a list that holds every row in the csv file.
    Every column needs to correspond to an attribute while the last column of
    a row needs to be the ground truth value of the example.
    The ground truth label of 0 is converted to a -1 here.
    :param file: File to copy contents from, must be a csv file
    :return: An array containing two arrays:
             [0]: array of every training feature vector
             [1]: array of every ground truth label
    """
    feature_vectors = []
    ground_truth_labels = []
    with open(file, 'r') as f:
        for line in f:
            # Convert all terms to floats
            terms = list(map(float, line.strip().split(',')))
            # Convert the ground truth label to integers and convert 0 to -1 label.
            label_index = len(terms) - 1
            label_value = terms[label_index]
            if label_value == 0:
                ground_truth_labels.append(-1)
            else:
                ground_truth_labels.append(1)
            # Remove the ground truth label from the terms to separate the feature vector from the label
            terms.pop(label_index)
            feature_vectors.append(terms)
    # Return both feature vectors and ground truth labels
    examples_set = list()
    examples_set.append(feature_vectors)
    examples_set.append(ground_truth_labels)
    return examples_set


def inner_product(arr1, arr2):
    """
    Gets the inner product between two arrays
    :param arr1:
    :param arr2:
    :return: inner product value
    """
    output = 0
    for i in range(len(arr1)):
        output = output + arr1[i] * arr2[i]
    return output


def sigmoid(s):
    """
    sigmoid(s) = \frac{1}{1 - e^{-s}}
    :param s:
    :return:
    """
    answer = 1 / (1 - math.exp(-s))
    if answer > 100:
        print('here')
    return answer


def forward_pass(x, w, width, layers, b):
    """
    Does forward pass and returns a 2D array containing all the neuron values.
    :param x: Input feature vector
    :param w: Weights off all neurons
    :param width: Width of each layer. Not including the bias parameter.
    :param layers: Number of hidden layers.
    :param b: Value the bias parameter will take.
    :return: 2D array of all neuron values.
    """
    width_and_bias = width + 1
    # Current neuron values including the bias parameter
    curr_z_values = x
    curr_z_values.insert(0, 1)
    # 2D array holding the neuron values for every layer in the neural network
    all_neuron_values = []
    # Append the feature vector to represent the 0th layer
    all_neuron_values.append(x)
    # Forward pass to get loss of prediction
    # For every layer in the neural network. +1 to include the output layer
    for curr_layer in range(layers + 1):
        layer_weights = w[curr_layer]
        z_plus_1_values = []
        z_plus_1_without_activation = []
        # If the next layer isn't the output layer
        if curr_layer < layers:
            # Add the bias parameter to the next layer of z neurons only
            z_plus_1_values.append(b)
            z_plus_1_without_activation.append(b)
        if curr_layer == layers:
            width = 1
        # For every next level z not including the bias parameter
        for z_plus_1 in range(width):
            neuron = 0
            # The 0th layer only has len(x) nodes
            if curr_layer == 0:
                for i in range(len(x)):
                    # Multiply weights with current z value
                    neuron = neuron + (layer_weights[i][z_plus_1] * curr_z_values[i])
                z_plus_1_without_activation.append(neuron)
            # All other layers have width + 1 nodes
            else:
                # Add weights needed to calculate z_plus_1. Adding 1 because bias parameter has weights.
                for i in range(width_and_bias):
                    # Multiply weights with current z value
                    neuron = neuron + (layer_weights[i][z_plus_1] * curr_z_values[i])
                z_plus_1_without_activation.append(neuron)
            # If the next layer isn't the output layer
            if curr_layer < layers:
                # Use activation function on neuron after weights have been calculated
                neuron = sigmoid(neuron)
            z_plus_1_values.append(neuron)
        curr_z_values = z_plus_1_values
        all_neuron_values.append(z_plus_1_without_activation)
    # Last element contains the output
    return all_neuron_values


def get_loss(all_neuron_values, y_prime):
    """
    Loss(y, y*) = (1 / 2) * (y - y *) ^ 2
    :param all_neuron_values:
    :param y_prime: this is the ground truth label
    :return: Loss of output
    """
    loss = 0
    # Last element contains the output
    y = all_neuron_values[len(all_neuron_values) - 1][0]
    return (1/2) * ((y - y_prime)**2)


def back_propagate(y, w, width, layers, all_neuron_values, prediction):
    # Will hold all of the partial weights that will be returned
    partial_weights = []
    # Contains the partials of each node needed to calculated the partials at this layer.
    partials_of_parents = [prediction - y]

    first_layer_weights = []
    first_children = []
    # For every child node (i.e., for every node in the next layer)
    for z in range(width + 1):
        child_partial_weights = []

        # Get weight partial of every child node
        weight_partial = partials_of_parents[0] * all_neuron_values[layers][z]
        child_partial_weights.append(weight_partial)

        # Get partial of every child node if its not the bias child
        if z != 0:
            z_partial = partials_of_parents[0] * w[layers][z][0]
            first_children.append(z_partial)

        first_layer_weights.append(child_partial_weights)
    partial_weights.append(first_layer_weights)
    # Setting up for finding the partials of the hidden layers
    partials_of_parents = first_children

    # Counting down for all the hidden layers
    for curr_layer in range(layers, 0, -1):
        curr_layer_weights = []
        partials_of_children = []
        # For every child node (i.e., for every node in the next layer)
        for z in range(width + 1):
            child_partial_weights = []

            # Get partial of every weight this child points to
            for z_parent in range(width):
                weight_partial = partials_of_parents[z_parent] * all_neuron_values[curr_layer][z]
                s = all_neuron_values[curr_layer][z_parent]   # Parent of current node
                weight_partial = weight_partial * sigmoid(s) * (1 - sigmoid(s))
                child_partial_weights.append(weight_partial)

            # Get the partial of the current node, only if its not the bias node and if its not an input node
            if z != 0 and curr_layer != 1:
                # For every parent this child node has
                for parent in range(len(partials_of_parents)):
                    z_partial = z_partial + partials_of_parents[parent] * w[curr_layer - 1][z][parent]
                    s = all_neuron_values[curr_layer][z_parent]  # Parent of current node
                    z_partial = z_partial * sigmoid(s) * (1 - sigmoid(s))
                partials_of_children.append(z_partial)

            # Save all weights this child node had
            curr_layer_weights.append(child_partial_weights)
            # Weights are counting up
        # In the next layer, children will become parents
        partials_of_parents = partials_of_children
        # Save all partial weights that were calculated at this layer
        partial_weights.insert(0, curr_layer_weights)
    return partial_weights


def get_all_partial_weights(x, y, w, width, layers, b):
    """
    :param x: Feature vector. Will be converted into augmented feature vector.
    :param y: Ground truth label
    :param w: Weight vector. Will be converted into augmented weight vector.
              3D array with the following categories:
              1st dimension: All layers in Neural Network - 1
              2nd dimension: Neurons for the given layer
              3rd dimension: Weight vectors for the given neuron.
    :param width: Width of neurons in each layer. Does not includes the bias parameter.
    :param layers: Desired number of layers for neural network
    :param b: Desired bias value
    :return: 2D list holding every weights for every layer:
             [0]: Layer zero weights
             [1]: Layer 1 weights
             .
             .
             .
             [layer]: Last hidden layer weights
    """
    # Get loss and neuron values after forward pass
    all_neuron_values = forward_pass(x, w, width, layers, b)
    loss_of_forward_pass = get_loss(all_neuron_values, y)
    # Last element contains the prediction
    prediction = all_neuron_values[len(all_neuron_values) - 1][0]
    # Back propagation to get all weight values
    return back_propagate(y, w, width, layers, all_neuron_values, prediction)


def update_wt(wt, feature_vector, ground_truth_label, gamma, width, layers, bias):
    """
    𝒘 ← 𝒘 − 𝛾t𝛻𝐿(𝑁𝑁 𝒙i,𝒘), 𝑦&))
    :param wt: Initial weights
    :param feature_vector:
    :param ground_truth_label:
    :param gamma:
    :param width:
    :param layers:
    :param bias:
    :return: Updated wt that takes a step towards converging
    """
    # Number of nodes for this layer
    num_nodes = width
    # 𝛻𝐿(𝑁𝑁 𝒙i,𝒘), 𝑦&))
    partials = get_all_partial_weights(feature_vector, ground_truth_label, wt, width, layers, bias)
    # For every hidden layer including the output layer
    for layer in range(layers + 1):
        # Number of neurons in this layer
        for neuron in range(width + 1):
            num_nodes = width
            # Number of parents this neuron has
            if layer == layers:
                width = 1
            elif layer == 0:
                num_nodes = len(feature_vector)
            for weight in range(num_nodes):
                # 𝒘 ← 𝒘 − 𝛾t𝛻𝐿(𝑁𝑁 𝒙i,𝒘), 𝑦&))
                wt[layer][neuron][weight] -= (gamma * partials[layer][neuron][weight])
    return wt


def stochastic_gradient_descent_for_NN(initial_gamma, d, initial_wt, feature_vectors, ground_truth_labels, width, layers, bias):
    """
    :param initial_gamma: Learning rate
    :param d: Cost hyper-parameter
    :param initial_wt: starting position
    :param feature_vectors: Examples to train learning algorithm.
    :param ground_truth_labels:
    """
    # Initialize weight vector at t iteration
    wt = initial_wt
    # Initialize initial gamma
    gamma = initial_gamma
    # Try to make the gradient converge but give up after 100 tries
    for T in range(100):
        # Shuffle training set
        temp = list(zip(feature_vectors, ground_truth_labels))
        random.shuffle(temp)
        feature_vectors, ground_truth_labels = zip(*temp)
        # For each training example.
        for i in range(len(feature_vectors)):
            # 𝒘 ← 𝒘 − 𝛾t𝛻𝐿(𝑁𝑁 𝒙i,𝒘), 𝑦&))
            wt = update_wt(wt, feature_vectors[i], ground_truth_labels[i], gamma, width, layers, bias)
            # Learning rate takes the next step
            gamma = gamma / (1 + (gamma * i) / d)
    return wt


def main():
    training_examples = import_examples('train.csv')
    feature_vectors = training_examples[0]
    ground_truth_labels = training_examples[1]
    initial_weights = get_initial_weights()
    feature_vector = [1, 1]
    ground_truth_label = 1
    width = 10
    layers = 2
    bias_value = 1
    gamma = .5
    d = 5

    random_weights = get_random_initial_weights(width, layers, feature_vectors[0], bias_value)
    learned_weights = stochastic_gradient_descent_for_NN(gamma, d, random_weights, feature_vectors, ground_truth_labels, width, layers, bias_value)
#    test_examples = import_examples('test.csv')


if __name__ == "__main__":
    main()
