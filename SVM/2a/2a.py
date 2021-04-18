'''
Author: Sergio Remigio
Date: 4/17/2021

Implements the Stochastic Sub-gradient Descent
given the learning rate: gamma = gamma / (1 + (gamma * t) / C)
'''
import csv
import math
import random


def import_examples(file):
    """
    Copies file contents into a list that holds every row in the csv file
    Every column needs to correspond to an attribute while the last column of
    a row needs to be the ground truth value of the example.
    The ground truth label of 0 is converted to a -1 here.
    :param file: File to copy contents from, must be a csv file
    :return:
    """
    examples_set = list()
    with open(file, 'r') as f:
        for line in f:
            terms = list(map(float, line.strip().split(',')))
            # Convert the ground truth label to integers and convert 0 to -1 label.
            label_index = len(terms) - 1
            label_value = terms[label_index]
            if label_value == 0:
                terms[label_index] = -1
            else:
                terms[label_index] = 1
            examples_set.append(terms)
    return examples_set


def getExamplePrediction(W, example):
    """
    Prediction = y_i W\top x_i
    :param W: Weight vector
    :param example: Example with ground truth label appended at the end
    :return: Predicition
    """
    prediction = 0
    # prediction = y_i W\top x_i
    # Remember that the ground truth label is appended to the example
    for i in range(len(example) - 1):
        prediction += W[i] * example[i]
    # Multiplying by the ground truth label at the end.
    prediction = prediction * example[len(example) - 1]
    return prediction


def update_W(W, wt, gamma, C, N, example):
    """
    Update: W ← W − 𝛾_t [wt; 0] + 𝛾_t * C * N * yi * xi
          : W ← W − a + b 
    :param W: Weight vector
    :param wt: current weight vector
    :param gamma: learning rate
    :param C: hyper-parameter
    :param N: training examples size
    :param example: current example
    :return: Updated W
    """
    # 𝛾_t [w0; 0]
    for i in range(len(wt)):
        wt[i] = gamma * wt[i]
    # a = 𝛾_t [w0; 0]
    a = wt
    # 𝛾_t * C * N * yi
    product_of_scalars = gamma * C * N * example[len(example) - 1]
    # product of scalars(𝛾_t * C * N * yi) * vector(example)
    # Remember that example has the ground truth label appended at the end.
    for i in range(len(example) - 1):
        example[i] = product_of_scalars * example[i]
    # b = 𝛾_t * C * N * yi
    b = example
    #: W ← W − a + b
    for i in range(len(wt)):
        W[i] = W[i] - a[i] + b[i]
        
    return W


def update_wt(wt, gamma):
    """
    update: w0 ← (1- 𝛾t) w0
    :param wt: Current weight vector
    :param gamma: learning rate
    :return: Updated wt
    """
    for i in range(len(wt)):
        wt[i] = (1 - gamma) * wt[i]
    return wt


def stochastic_sub_gradient_descent_for_SVM(initial_gamma, C, initial_wt, training_examples):
    """
    :param initial_gamma: Learning rate
    :param initial_wt: starting position
    :param training_examples: Examples to train learning algorithm.
    """
    # Initialize weight vector at t iteration
    wt = initial_wt
    # Initialize final weight  vector. Both are the zero vector.
    W = [0, 0, 0, 0]
    # Initialize initial gamma
    gamma = initial_gamma

    # Try to make the gradient converge but give up after 100 tries
    for T in range(100):
        # Shuffle training set
        random.shuffle(training_examples)
        # Iteration count
        t = 0
        # For each training example.
        for example in training_examples:
            # Get prediction of current example
            prediction = getExamplePrediction(W, example)
            if prediction <= 1:
                # W ← W − 𝛾_t [w0; 0] + 𝛾_t * C * N * yi * xi
                W = update_W(W, wt, gamma, C, len(training_examples), example)
            else:
                # w0 ← (1- 𝛾t) w0
                wt = update_wt(wt, gamma)
            # Learning rate takes the next step
            gamma = gamma / (1 + (gamma * t) / C)
            # Increase trail count
            t += 1
    return W


def get_error_percentage(W, training_examples):
    """
    Gets percentage success of weight vector
    :param W: Given weight vector
    :param training_examples: Examples
    :return: Percentage error
    """
    correct_count = 0
    for example in training_examples:
        prediction = getExamplePrediction(W, example)
        if prediction > 0:
            correct_count += 1
    return correct_count / len(training_examples)


if __name__ == "__main__":
    training_examples = import_examples('train.csv')
    test_examples = import_examples('test.csv')

    # Initial gamma learning rate
    initial_gamma = .5
    # Hyper-parameter that controls the tradeoff
    C = 10
    initial_wt = [0, 0, 0, 0]
    W = stochastic_sub_gradient_descent_for_SVM(initial_gamma, C, initial_wt, training_examples)
    success_percentage = get_error_percentage(W, training_examples)

    # Following simply prints the calculations in the console
    best = 0
    best_percentage = 0
    for i in range(1, 200):
        initial_wt = [0, 0, 0, 0]
        initial_gamma = .5
        training_examples = import_examples('train.csv')
        W = stochastic_sub_gradient_descent_for_SVM(initial_gamma, i, initial_wt, training_examples)
        success_percentage = get_error_percentage(W, training_examples)
        print("C:", i, "\t", success_percentage)
        if success_percentage > best_percentage:
            best = i
            best_percentage = success_percentage
    print("Best C:", best)

