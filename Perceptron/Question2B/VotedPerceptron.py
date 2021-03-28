'''
Author: Sergio Remigio
Date: 3/27/2021

This file contains the implementation of the voted perceptron algorithm.
After the perceptron algorithm has gone through all of the epochs in the given training set,
a list of weighted unit vector separating hyperplanes and their respective counts is returned.
Finally, the returned list is printed and the prediction success rate for the given test set is printed to the console.
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
                 The ground truth label will also be changed from 0 to -1.
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


def get_voted_list(D, w_m, T, r):
    """

    :param D: Training set D = {(xi , yi )}, xi ∈ ℜ^n , yi ∈ {-1,1}
    :param w_m: w_m = 0 ∈ ℜ^n
    :param T: Number of epochs
    :param r: Learning rate
    :return:
    """
    # List of all unit vector separating hyperplanes for m number of updates
    all_hypothesized_u = list()
    # List of number of correct predictions made by u at the mth update
    c_m = list()
    c = 0

    # For epoch = 1 … T:
    for t in range(T):
        # Shuffle Data
        random.shuffle(D)
        # For each training example (x i , y i ) ∈ D:
        for xi in D:
            # Prediction = yi(w_m T xi)
            prediction = get_prediction(w_m, xi)
            # If prediction is wrong
            if prediction <= 0:
                # Add w_m to list of hypothesized u that incorrectly predicted a feature vector
                all_hypothesized_u.append(w_m)
                # Add its count of corrected prediction to c_m
                c_m.append(c)
                # update w_m to a new hypothesized w_m: w_m ← w_m + r*yi*xi
                w_m = update_separating_vector(w_m, r, xi)
                c = 1
            # Prediction was correct
            else:
                # Increase correctly predicted count + 1
                c += 1
        # Update the learning rate
        r = r / 2
    list_of_votes = list()
    list_of_votes.append(all_hypothesized_u)
    list_of_votes.append(c_m)
    return list_of_votes


def get_sign_of_prediction(w, xi):
    """
    Returns the sign of the prediction using the separating unit vector w
    :param w: separating unit vector
    :param xi: feature vector
    :return: sign(yi(w T xi))
    """
    # Ground truth label is at the end of the example
    yi = xi[len(xi) - 1]
    # w inner product xi
    w_T_xi = get_inner_product(w, xi)
    prediction = yi * w_T_xi
    # Return the sign of the prediction
    if prediction <= 0:
        return -1
    return 1


def get_voted_prediction(list_of_votes, xi):
    """
    Returns a prediction by using a weighted vote of every single hypothesized unit vector separating hyperplane.
    :param list_of_votes: Contains:
                          [0] = all hypothesized unit vector separating hyperplanes
                          [1] = count of correct prediction for mth separating hyperplane
    :param xi: ith example
    :return: Voted prediction using list_of_votes
    """
    voted_prediction = 0
    # For every unit vector hypothesized
    for m in range(len(list_of_votes[0])):
        # Get the sign of the prediction using the mth separating unit vector
        sign_prediction = get_sign_of_prediction(list_of_votes[0][m], xi)
        correct_prediction_count = list_of_votes[1][m]
        voted_prediction += correct_prediction_count * sign_prediction
    return voted_prediction


def print_prediction_success_percentage(list_of_votes, r, test_examples):
    success_count = 0
    for xi in test_examples:
        prediction = get_voted_prediction(list_of_votes, xi)
        # if predicted correctly
        if prediction > 0:
            success_count += 1
    for m in range(len(list_of_votes[0])):
        print("m = " + str(m) + "\t Weighted vector: " + str(list_of_votes[0][m])
              + "\t Correct Predictions: " + str(list_of_votes[1][m]))
    success_rate = success_count / len(test_examples)
    print("Success percent: " + str(success_rate))


if __name__ == "__main__":
    # Given a training set D = {(xi , yi )}, xi ∈ ℜ^n , yi ∈ {-1,1}
    D = import_examples("train.csv")
    # Initialize w_m = 0 ∈ ℜ^n
    w_m = get_zero_vector(D)
    # Number of epochs
    T = 10
    # Learning Rate
    r = .5
    list_of_votes = get_voted_list(D, w_m, T, r)

    test_examples = import_examples("test.csv")
    print_prediction_success_percentage(list_of_votes, r, test_examples)
