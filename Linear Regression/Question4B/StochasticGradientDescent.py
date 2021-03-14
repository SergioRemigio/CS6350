'''
Author: Sergio Remigio
Date: 3/13/2021

This file contains the implementation of the Stochastic Gradient Descent Algorithm.
This file prints to the console the weight vector, r, and loss function values
of the training examples for every t interval.

After the Stochastic Gradient Descent Algorithm has converged, the final
weight vector, r, and los function values of the test examples is printed to the console.
'''
import csv
import math
import random

def import_examples(file):
    """
    Copies file contents into a list that holds every row in the csv file
    Every column needs correspond to an attribute while the last column of
    a row needs to be the ground truth value of the example
    :param file: File to copy contents from, must be a csv file
    :return:
    """
    examplesSet = list()

    with open(file, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            examplesSet.append(terms)
    return examplesSet


def get_gradient_of_loss_function_wt(wt, training_examples):
    """
    Function in latex: $\frac{\nabla J}{\nabla \w} =
         [\frac{\partial J}{\partial w_1}, \frac{\partial J}{\partial w_2},... \frac{\partial J}{\partial W_d}]$
         where d = number of features in training examples

         Latex of partial of loss function with respect to gradient j:
         $$ \frac{\partial J}{\partial w_j} = -\sum_{i = 1}^m (y_i - w^\top x_i) (x_{ij}) $$
    :param wt: Weight vector
    :param training_examples: examples
    :return: returns the gradient of the loss function
    """
    stochastic_examples = list()
    stochastic_examples.append(get_random_example(training_examples))
    stochastic_examples.append(get_random_example(training_examples))

    # gradient to return
    gradient = list()
    num_of_features = len(stochastic_examples[0]) - 1
    # Get the partial of every feature
    for j in range(num_of_features):
        partial = 0

        # Take the sum of all examples where of (yi - w(inner product) xi)(xij)
        for i in stochastic_examples:
            # Ground truth value is always the last column of the example row
            yi = float(stochastic_examples[0][len(i) - 1])
            # Feature vector weight (wt) inner product example at i (xi)
            wt_dot_xi = 0
            for nextFeature in range(num_of_features):
                wt_dot_xi += wt[nextFeature] * float(i[nextFeature])
            # Cost of prediction
            cost = yi - wt_dot_xi
            # cost * (xij)
            cost = cost * float(i[j])
            # Add up all the costs
            partial += cost
        # Make sure to multiply by -1 at the end
        partial = -partial
        # Add jth partial and repeat until all partials of each feature is gotten
        gradient.append(partial)

    return gradient


def print_regression_of_all_examples(w, examples):
    """
    Prints the regression of the examples using
    w as the linear classifier.
    regression = w inner product ith example(xi)
    :param w: weight vector
    :param examples: need to calculate regression of each example
    """
    num_of_features = len(training_examples[0]) - 1
    count = 1
    for example in examples:
        regression = 0
        # inner product is the regression of the example
        for nextFeature in range(num_of_features):
            regression += w[nextFeature] * float(example[nextFeature])
        print("i = " + str(count) + ": \t " + str(regression))
        count += 1



def predict_new_weight_vector(wt, r, nabla_J_wt):
    """
    Predicts the w_t+1 weight vector to try to converge the loss function
    at the t+1 iteration.
    w_t+1 = w_t - (r)(nabla_J_wt)
    :param wt: weight vector
    :param r: learning rate
    :param nabla_J_wt: Gradient of the loss function
    :return: returns weight vector w_t+1
    """
    # (r)(nabla_J_wt)
    scaled_gradient = list()
    for i in nabla_J_wt:
        scaled_gradient.append(i * r)

    new_weight_vector = list()
    # wt - (r)(nabla_J_wt) This is a vector subtraction
    for feature in range(len(wt)):
        new_weight = wt[feature] - scaled_gradient[feature]
        new_weight_vector.append(new_weight)

    # Return the new weight vector after subtracting wt -(r)(nabla_J_wt)
    return new_weight_vector


def get_change_in_gradients(wt, wt_plus1):
    """
    Norm of (w_t+1 - w_t)
    Norm = (v inner product v)^(1/2)
    :param wt: weight vector
    :param wt_plus1: new weight vector
    :return: change in gradient
    """
    norm = 0
    difference_vector = list()
    # Subtract the vectors w_t+1 - w_t
    for feature in range(len(wt)):
        difference_vector.append(wt_plus1[feature] - wt[feature])
    # Take the norm of the difference vector
    for feature in range(len(difference_vector)):
        norm += difference_vector[feature]**2
    # At the end take the square root
    norm = math.sqrt(norm)
    return norm


def get_random_example(training_examples):
    rng = random.randint(0, 52)
    return training_examples[rng]


def stochastic_gradient_descent(tolerance_level, learning_rate, initial_r, initial_wt, training_examples, test_examples):
    """
    Implementation of the gradient descent algorithm.
    Prints to the console the weight vector, r, and loss function values
    of the training examples for every t interval.
    After the Gradient Descent Algorithm has converged, the final
    weight vector, r, and los function values of the test examples is printed to the console.
    :param tolerance_level:
    :param learning_rate:
    :param initial_r:
    :param training_examples:
    :param test_examples:
    """
    # Initialize initial weight vector
    wt = initial_wt
    # Learning rate r
    r = initial_r

    # Try to make the gradient converge but give up after 100 tries
    for t in range(100):
        # Gradient  of the loss function
        nabla_J_wt = get_gradient_of_loss_function_wt(wt, training_examples)
        # New weight vector that approaches the convergence of the loss function
        wt_plus1 = predict_new_weight_vector(wt, r, nabla_J_wt)

        # Simply prints the result at t iteration
        print("At " + str(t) + " iteration r = " + str(r) + " weight vector wt = " + str(wt))
        print("For all examples at i, this is their loss function value:")
        print_regression_of_all_examples(wt, training_examples)

        # Check if loss function has converged
        delta_gradient = get_change_in_gradients(wt, wt_plus1)
        print("delta" + str(delta_gradient))
        if delta_gradient < tolerance_level:
            # Loss function finally converged, quit searching
            break

        # Set up for t+1 iteration
        r = r * learning_rate
        wt = wt_plus1


    # Finally print out the r, weight vector, and loss function values of the test examples
    print("Final values for Test Examples: ")
    print("At " + str(t) + " iteration r = " + str(r) + " weight vector wt = " + str(wt))
    print("For all examples at i, this is their loss function value:")
    print_regression_of_all_examples(wt, test_examples)


if __name__ == "__main__":
    training_examples = import_examples('train.csv')
    test_examples = import_examples('test.csv')
    tolerance_level = .00001
    learning_rate = .5
    initial_r = .5
    initial_wt = [0, 0, 0, 0, 0, 0, 0]
    stochastic_gradient_descent(tolerance_level, learning_rate, initial_r, initial_wt, training_examples, test_examples)
