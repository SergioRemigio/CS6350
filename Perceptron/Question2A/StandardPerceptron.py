'''
Author: Sergio Remigio
Date: 3/27/2021

This file contains the implementation of the perceptron algorithm.
After the perceptron algorithm has found a separating hyperplane u given the training examples,
the file prints to the console:
the separating hyperplane as a unit vector u,
the learning rate r,
and the average prediction success of the given test examples.
'''
import csv
import math
import random

def import_examples(file):
    """
    Copies file contents into a list that holds every row in the csv file
    Every column needs correspond to an attribute while the last column of
    a row needs to be the ground truth value of the example.
    :param file: File to copy contents from, must be a csv file.
    #The ground truth label will also be changed from 0 to -1.
    :return: List containing the training examples and ground truth label.
             Every feature in every example will also be converted to a float
    """
    examplesSet = list()
    with open(file, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            # Make every feature an number
            for i in range(len(terms)):
                terms[i] = float(terms[i])
            # The ground truth label will also be changed from 0 to -1.
            if terms[len(terms) - 1] == 0:
                terms[len(terms) - 1] = -1
            examplesSet.append(terms)
    return examplesSet

def get_zero_vector(training_set):
    """
    :param training_set: Used to figure out how many features are in every example
    :return: Zero vector
    """
    zero_vector = list()
    # Number of attributes in the feature vectors and -1 because the example contains the ground truth label at the end
    num_attributes = len(training_set[0]) - 1
    # Simply create the 0 vector
    for x in range(num_attributes):
        zero_vector.append(0)
    return zero_vector


def get_inner_product(v, xi):
    """
    Returns the inner product of any n dimensional vector
    and a training example.
    Note: Vector dimensions = feature vector dimensions
    :param v: Any vector
    :param xi: Must be a training example
    :return:
    """
    inner_product = 0
    for i in range(len(xi) - 1):
        inner_product += (v[i] * xi[i])
    return inner_product


def get_prediction(w, xi):
    """
    Returns the prediction using the separating unit vector w
    :param w: separating unit vector
    :param xi: feature vector
    :return: yi(w T xi)
    """
    # Ground truth label is at the end of the example
    yi = xi[len(xi) - 1]
    # w inner product xi
    w_T_xi = get_inner_product(w, xi)
    prediction = yi * w_T_xi
    return prediction


def normalize(w):
    """
    Returns the vector w normalized.
    W PARAMETER IS MODIFIED
    :param w: vector to normalize. This parameter is modified
    """
    w_T_w = 0
    # w inner product w
    for i in range(len(w)):
        w_T_w += w[i] * w[i]
    norm_w = math.sqrt(w_T_w)
    # To normalize, divide norm_w by every feature in w
    for i in range(len(w)):
        w[i] = w[i] / norm_w


def update_separating_vector(w, r, xi):
    """
    update w ← w + r*yi*xi
    :param w: Separating unit vector
    :param r: learning rate
    :param xi: Feature vector
    :return: Returns the updated separating vector as a unit vector
    """
    # Ground truth label is at the end of the example
    yi = xi[len(xi) - 1]
    updated_w = list()
    # Update w ← w + r*yi*xi
    for i in range(len(w)):
        updated_w.append(w[i] + (r * yi * xi[i]))
    normalize(updated_w)
    return updated_w


def get_separating_hyperplane(D, w, T, r):
    """
    Using the perceptron algorithm, a unit vector separating hyperplane is returned
    :param D: Training set D = {(xi , yi )}, xi ∈ ℜ^n , yi ∈ {-1,1}
    :param w: w = 0 ∈ ℜ^n
    :param T: Number of epochs
    :param r: Learning rate
    :return: Unit vector separating hyperplane u
    """
    # For epoch = 1 … T:
    for t in range(T):
        # Shuffle Data
        random.shuffle(D)
        # For each training example (x i , y i ) ∈ D:
        for xi in D:
            # Prediction = yi(w T xi)
            prediction = get_prediction(w, xi)
            # If prediction is wrong
            if prediction <= 0:
                # update w ← w + r*yi*xi
                w = update_separating_vector(w, r, xi)
        # Update the learning rate
        r = r / 2
    return w


def print_prediction_success_percentage(u, r, test_examples):
    """
    After the perceptron algorithm has found a separating hyperplane u given the training examples,
    the file prints to the console:
    the separating hyperplane as a unit vector u,
    the learning rate r,
    and the average prediction success of the given test examples.
    :param u: separating hyperplane u
    :param test_examples: Test examples to predict
    """
    success_count = 0
    for xi in test_examples:
        prediction = get_prediction(u, xi)
        # if predicted correctly
        if prediction > 0:
            success_count += 1

    success_rate = success_count / len(test_examples)

    print("Unit vector separating hyperplane: " + str(u))
    print("Learning rate: " + str(r))
    print("Success percent: " + str(success_rate))


if __name__ == "__main__":
    # Given a training set D = {(xi , yi )}, xi ∈ ℜ^n , yi ∈ {-1,1}
    D = import_examples("train.csv")
    # Initialize w = 0 ∈ ℜ^n
    w = get_zero_vector(D)
    # Number of epochs
    T = 10
    # Learning Rate
    r = .5
    separating_hyperplane_u = get_separating_hyperplane(D, w, T, r)

    test_examples = import_examples("test.csv")
    print_prediction_success_percentage(separating_hyperplane_u, r, test_examples)

