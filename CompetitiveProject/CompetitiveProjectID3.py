"""
Author: Sergio Remigio

This file implements a Decision tree with a naive bayes classifier
at the leaf nodes.
Naive bayes is incorporated into the decision tree by finding the entropy
at a node and deciding to create a naive bayes node only if the entropy at the current
node is less than a hyper parameter threshold.
"""
import csv
import math
import copy

maximum_tree_depth = 1
gain_threshold = 0
histogram_count = 0

attribute_values = {
    "age": [36, 36.0000001],
    "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov",
                  "Without-pay", "Never-worked", "?"],
    "fnlwgt": [111567, 111567.000001],
    "education": ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th",
                  "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool", "?"],
    "education_num": [9, 9.000001],
    "marital_status": ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed",
                       "Married-spouse-absent", "Married-AF-spouse", "?"],
    "occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty",
                   "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving",
                   "Priv-house-serv", "Protective-serv", "Armed-Forces", "?"],
    "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried", "?"],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black", "?"],
    "sex": ["Female", "Male", "?"],
    "capital_gain": [0, 0.0000001],
    "capital_loss": [0, 0.0000001],
    "hours_per_week": [40, 40.0000000001],
    "native_country": ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
                       "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran",
                       "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal",
                       "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia",
                       "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador",
                       "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands", "?"],
}
# Add 1 to every possible event to prevent multiplying by zero
features = {
    "age": {
        36: 1,
        36.0000001: 1
    },
    "workclass": {
        "Private": 1,
        "Self-emp-not-inc": 1,
        "Self-emp-inc": 1,
        "Federal-gov": 1,
        "Local-gov": 1,
        "State-gov": 1,
        "Without-pay": 1,
        "Never-worked": 1,
        "?": 1
    },
    "fnlwgt": {
        111567: 1,
        111567.000001: 1
    },
    "education": {
        "Bachelors": 1,
        "Some-college": 1,
        "11th": 1,
        "HS-grad": 1,
        "Prof-school": 1,
        "Assoc-acdm": 1,
        "Assoc-voc": 1,
        "9th": 1,
        "7th-8th": 1,
        "12th": 1,
        "Masters": 1,
        "1st-4th": 1,
        "10th": 1,
        "Doctorate": 1,
        "5th-6th": 1,
        "Preschool": 1,
        "?": 1
    },
    "education_num": {
        9: 1,
        9.000001: 1
    },
    "marital_status": {
        "Married-civ-spouse": 1,
        "Divorced": 1,
        "Never-married": 1,
        "Separated": 1,
        "Widowed": 1,
        "Married-spouse-absent": 1,
        "Married-AF-spouse": 1,
        "?": 1
    },
    "occupation": {
        "Tech-support": 1,
        "Craft-repair": 1,
        "Other-service": 1,
        "Sales": 1,
        "Exec-managerial": 1,
        "Prof-specialty": 1,
        "Handlers-cleaners": 1,
        "Machine-op-inspct": 1,
        "Adm-clerical": 1,
        "Farming-fishing": 1,
        "Transport-moving": 1,
        "Priv-house-serv": 1,
        "Protective-serv": 1,
        "Armed-Forces": 1,
        "?": 1
    },
    "relationship": {
        "Wife": 1,
        "Own-child": 1,
        "Husband": 1,
        "Not-in-family": 1,
        "Other-relative": 1,
        "Unmarried": 1,
        "?": 1
    },
    "race": {
        "White": 1,
        "Asian-Pac-Islander": 1,
        "Amer-Indian-Eskimo": 1,
        "Other": 1,
        "Black": 1,
        "?": 1
    },
    "sex": {
        "Female": 1,
        "Male": 1,
        "?": 1
    },
    "capital_gain": {
        0: 1,
        0.0000001: 1
    },
    "capital_loss": {
        0: 1,
        0.0000001: 1
    },
    "hours_per_week": {
        40: 1,
        40.0000000001: 1
    },
    "native_country": {
        "United-States": 1,
        "Cambodia": 1,
        "England": 1,
        "Puerto-Rico": 1,
        "Canada": 1,
        "Germany": 1,
        "Outlying-US(Guam-USVI-etc)": 1,
        "India": 1,
        "Japan": 1,
        "Greece": 1,
        "South": 1,
        "China": 1,
        "Cuba": 1,
        "Iran": 1,
        "Honduras": 1,
        "Philippines": 1,
        "Italy": 1,
        "Poland": 1,
        "Jamaica": 1,
        "Vietnam": 1,
        "Mexico": 1,
        "Portugal": 1,
        "Ireland": 1,
        "France": 1,
        "Dominican-Republic": 1,
        "Laos": 1,
        "Ecuador": 1,
        "Taiwan": 1,
        "Haiti": 1,
        "Columbia": 1,
        "Hungary": 1,
        "Guatemala": 1,
        "Nicaragua": 1,
        "Scotland": 1,
        "Thailand": 1,
        "Yugoslavia": 1,
        "El-Salvador": 1,
        "Trinadad&Tobago": 1,
        "Peru": 1,
        "Hong": 1,
        "Holand-Netherlands": 1,
        "?": 1
    }
}


attribute_index = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
]

def is_integer(x):
    """
    Checks whether input is an integer.
    :param x: Object to check
    :return: True if integer, False if not
    """
    try:
        int(x)
        return True
    except ValueError:
        return False


class Tree:
    """
    """

    def __init__(self):
        # Holds the list of child trees this tree is a parent of
        self.children = list()
        # The attribute of this tree
        self.attribute = None
        self.branch = None
        self.nextTree = None
        self.histogram = []


def find_probabilities_of_labels(examples_set):
    """
    Given a set, returns a list containing
    the probabilities of every label in order:
    1. "yes"
    2. "no"
    :param examples_set: examples to calculate
    :return: List contains probabilities: [0]:yes, [1]:no
    """
    labelsCount = {
        "yes": 0,
        "no": 0
    }
    probOfLabels = list()

    if isinstance(examples_set[0], list):
        totalCount = len(examples_set)

        for example in examples_set:
            nextLabel = example[14]
            labelsCount[nextLabel] += 1

    # There is only one example in the set
    else:
        totalCount = 1
        nextLabel = examples_set[14]
        labelsCount[nextLabel] += 1

    for label in labelsCount.keys():
        labelProbability = (labelsCount[label] / totalCount)
        probOfLabels.append(labelProbability)

    return probOfLabels


def find_entropy(examples_set):
    """
    Entropy{p1, p2, ..., pk} = -(\sum_{i=1}(pilog_4(pi)))
    where p = the probability of kth label
    (i.e., probability of "unacc", "acc".. etc.)
    :param examples_set: Set to be calculated
    :return: Entropy of the given examples set
    """
    pk = find_probabilities_of_labels(examples_set)
    result = 0
    for p in pk:
        if p != 0:
            result = result + (p * math.log(p, 2.0))

    result = -result
    return result


def get_value_subsets_for_every_attribute(examples_set, attributes):
    """
    Returns a Dictionary of size n attributes containing:
    A dictionary of values of a single attribute with a list of
    all example subsets for that attribute
    :param examples_set:
    :param attributes:
    :return:
    """
    dict = {}

    # Loop through all of the examples for all attributes
    # and fill dictionary in with subsets of all value examples
    for i in range(14):
        if attributes[i] != -1:
            # Dictionary containing a subset(list of examples) for every value in attributes
            valueSubSets = {}

            # Create empty dictionary that will hold an attribute(key) and subset examples(value)
            for value in attribute_values[attributes[i]]:
                valueSubSets.update({value: list()})

            # Fill the dictionary
            # The example is a numeric type
            for example in examples_set:
                if is_integer(example[i]):
                    currNum = int(example[i])

                    upperBound = attribute_values[attributes[i]][0]
                    lowerBound = attribute_values[attributes[i]][1]
                    if currNum <= upperBound:
                        valueSubSets[upperBound].append(example)
                    if currNum > lowerBound:
                        valueSubSets[lowerBound].append(example)
                else:
                    valueSubSets[example[i]].append(example)

            # update dict to return later
            dict.update({attributes[i]: valueSubSets})

    return dict


def find_best_attribute_to_split_on_entropy(examples_set, attributes):
    """
    Returns the best attribute to split the example set on
    using Entropy
    :param examples_set: example set to calculate
    :param attributes: Available attributes to split on
    :return: Best attribute to split the decision tree using entropy
    """
    current_entropy = find_entropy(examples_set)

    gains_to_calculate = get_value_subsets_for_every_attribute(examples_set, attributes)
    largest_gain = 0
    best_attribute = None
    # Return a random attribute if all are the same
    for x in range(14):
        if attributes[x] != -1:
            best_attribute = attributes[x]

    best_gain = 0
    for attribute in gains_to_calculate.keys():
        gain = 0
        for value in gains_to_calculate[attribute].keys():
            value_sub_set = gains_to_calculate[attribute][value]
            numerator = len(value_sub_set)
            denominator = 0
            for attributeValue in gains_to_calculate[attribute].keys():
                denominator = denominator + len(gains_to_calculate[attribute][attributeValue])
            weight = numerator / denominator
            if not value_sub_set:
                continue
            gain = gain + (weight * find_entropy(value_sub_set))
        # Return a random attribute if all attributes have 0 gain
        gain = current_entropy - gain
        if gain >= largest_gain:
            largest_gain = gain
            best_attribute = attribute
            best_gain = gain
    if best_gain < gain_threshold:
        return "None"
    return best_attribute


def import_examples(file):
    """
    Copies file contents into a list that holds every row in the csv file
    Every column needs correspond to an attribute while the last column of
    a row needs to be the ground truth value of the example.
    :param file: File to copy contents from, must be a csv file.
    :return: List containing the training examples and ground truth label.
    """
    examples_set = list()

    with open(file, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            # Convert label 0 into no and 1 into yes
            if terms[14] == "0":
                terms[14] = "no"
            else:
                terms[14] = "yes"
            examples_set.append(terms)
    return examples_set


def import_test_examples(file):
    examples_set = list()

    with open(file, 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            terms.pop(0)
            examples_set.append(terms)
    return examples_set


def predict_label(examples_set):
    """
    Given a set of examples, predicts label outcome.
    :param examples_set: All examples
    :return: Predicted label outcome.
    """
    all_labels = list(('yes', 'no'))
    prediction = 'no'

    for label in all_labels:
        all_same_label = True
        for example in examples_set:
            if example[14] != label:
                all_same_label = False
                break
        if all_same_label:
            prediction = label
            break
    return prediction


def check_all_same_label(examplesSet, label):
    """
    Checks whether all labels in the given example set are the same
    to the input parameter label.
    :param examplesSet: All examples
    :param label: Label to match examples to.
    :return: True if all examples have the same label, false otherwise
    """
    for example in examplesSet:
        if example[14] != label:
            return False
    return True


def find_most_common_label_in_s(examples_set):
    """
    Returns the most common label in the given examples.
    :param examples_set: All examples
    :return: Most common label in s
    """
    labels_count = {
        "yes": 0,
        "no": 0
    }

    # Go through every example
    for example in examples_set:
        next_label = example[14]
        # Add its count to the label_count dictionary
        labels_count[next_label] += 1
    # Set a default most common label
    most_common_label = "yes"
    # Figure out which label is the most common
    for next_label in labels_count.keys():
        if labels_count[most_common_label] < labels_count[next_label]:
            most_common_label = next_label
    return most_common_label


def get_tree_depth(attributes):
    """
    Counts the number of -1 in the
    attributes list as that correlates to the
    the current tree depth
    :param attributes: All attributes of the data
    :return: Int of the current tree depth
    """
    depth = 0
    for attribute in attributes:
        if attribute == -1:
            depth = depth + 1
    return depth


def ID3_algorithm(examples_set, label, attributes):
    """
    Decision tree algorithm
    :param examples_set: s: the set of examples
    :param label: The target attribute(the prediction)
    :param attributes: The set of measured attributes
    :return: Root node containing a decision tree
    """
    # Checks whether maximum depth has been reached
    tree_depth = get_tree_depth(attributes)
    if tree_depth == maximum_tree_depth:
        # Return a leaf node with the most common label
        leaf = find_most_common_label_in_s(examples_set)
        return leaf

    # If all examples have the same label
    if check_all_same_label(examples_set, label):
        # If attributes empty
        if not attributes:
            # Return a leaf node with the most common label
            leaf = find_most_common_label_in_s(examples_set)
            return leaf

        # Else return a leaf node with the label
        return label

    # Create a root node for tree
    root = Tree()

    A = find_best_attribute_to_split_on_entropy(examples_set, attributes)

# -----------------------------------------------Naive Bayes-------------------------------------------------

    # If gain threshold was met. i.e. if there wasn't a good attribute to split on.
    if A == "None":
        child = Tree()
        child.branch = "Naive"
        histogram_yes = features
        histogram_no = features
        yes_examples = []
        no_examples = []
        for example in examples_set:
            # Get array only containing 1s
            if example[len(example) - 1] == "yes":
                yes_examples.append(example)
            else:
                # Get array only containing 0s
                no_examples.append(example)

        # Add every probability in the rest of the array given that its 1
        for example in yes_examples:
            # index
            i = 0
            for feature in example:
                if feature == "yes" or "no":
                    continue
                if i == 0 or i == 2 or i == 4 or i == 10 or i == 11 or i == 12:
                    j = 0
                    # Its a float so check
                    for key in histogram_yes[attribute_index[i]]:
                        if float(feature) <= float(key) and j == 0:
                            histogram_yes[attribute_index[i]][key] += 1
                        elif j == 1:
                            histogram_yes[attribute_index[i]][key] += 1
                        j += 1
                else:
                    histogram_yes[attribute_index[i]][feature] += 1
                i += 1

        yes_count = 0
        # Set probabilities
        for key in histogram_yes.keys():
            count = 0
            for second_key in histogram_yes[key]:
                count += histogram_yes[key][second_key]
            for second_key in histogram_yes[key]:
                histogram_yes[key][second_key] = histogram_yes[key][second_key] / count
            yes_count += count

        # Add every probability in the rest of the array given that its 0
        for example in no_examples:
            # index
            i = 0
            for feature in example:
                if feature == "yes" or "no":
                    continue
                if i == 0 or i == 2 or i == 4 or i == 10 or i == 11 or i == 12:
                    j = 0
                    # Its a float so check
                    for key in histogram_no[attribute_index[i]]:
                        if float(feature) <= float(key) and j == 0:
                            histogram_no[attribute_index[i]][key] += 1
                        elif j == 1:
                            histogram_no[attribute_index[i]][key] += 1
                        j += 1
                else:
                    histogram_no[attribute_index[i]][feature] += 1
                i += 1

        no_count = 0
        # Set probabilities
        for key in histogram_no.keys():
            count = 0
            for second_key in histogram_no[key]:
                count += histogram_no[key][second_key]
            for second_key in histogram_no[key]:
                histogram_no[key][second_key] = histogram_no[key][second_key] / count
            no_count = count

        yes_count = len(yes_examples)
        no_count = len(no_examples)
        total_count = yes_count + no_count

        child.histogram.append(yes_count / total_count)
        child.histogram.append(histogram_yes)
        child.histogram.append(no_count / total_count)
        child.histogram.append(histogram_no)

        root.children.append(child)

        root.attribute = "age"

        return root
# -----------------------------------------------Naive Bayes-------------------------------------------------

    attribute_column = list(attribute_values.keys()).index(A)

    root.attribute = A

    # For each possible value v of that A can take:
    for value in attribute_values[A]:
        child = Tree()
        # Add a new tree branch corresponding to A=v
        child.branch = value
        root.children.append(child)
        Sv = list()
        if isinstance(value, str):
            # Let Sv be the subset of examples in S with A = v
            for example in examples_set:
                if example[attribute_column] == value:
                    Sv.append(example)
        # This means the value is a numeric type
        else:
            # If the value tag is an int, then we check for example values less or equal to the value tag.
            if isinstance(value, int):
                for example in examples_set:
                    if int(example[attribute_column]) <= value:
                        Sv.append(example)
            # If the value tag is a float, then we check for example values greater than the value tag.
            if isinstance(value, float):
                for example in examples_set:
                    if int(example[attribute_column]) > value:
                        Sv.append(example)

        # If Sv is empty:
        if not Sv:
            # Add leaf node with the most common value of Label in S
            leaf = find_most_common_label_in_s(examples_set)
            child.nextTree = leaf
        else:
            # Below this branch add the next subtree
            label_prediction = predict_label(Sv)
            # Remove attribute A from all attributes
            new_attributes = copy.deepcopy(attributes)
            new_attributes[attributes.index(A)] = -1
            child.nextTree = ID3_algorithm(Sv, label_prediction, new_attributes)
    # Return Root node
    return root


def get_index(attribute, attributes):
    """
    Returns the index of the attribute given a list of attributes.
    :param attribute: Attribute to find index of.
    :param attributes: List of all attributes.
    :return: Index position in attributes of the given parameter attribute.
    """
    for i in range(14):
        if attribute == attributes[i]:
            return i


def decision_tree_prediction(example, root, attributes):
    """
    Returns a predicted label based on the decision tree given
    :param example: Given example
    :param root: Given decision tree
    :param attributes: All data attributes
    :return: Decision tree prediction
    """
    # If reached a leaf node, return the label
    if isinstance(root, str):
        return root

    # Attribute that was split on
    attribute = root.attribute
    # Column of the attribute that was split on
    i = get_index(attribute, attributes)
    testValue = example[i]
    # Check every child to see what path the example must take in the decision tree
    for child in root.children:
        if isinstance(child.branch, int):
            if int(testValue) <= child.branch:
                return decision_tree_prediction(example, child.nextTree, attributes)
        elif isinstance(child.branch, float):
            if int(testValue) > child.branch:
                return decision_tree_prediction(example, child.nextTree, attributes)
# -----------------------------------------------Naive Bayes-------------------------------------------------
        # Naive bayes
        elif child.branch == "Naive":
            yes_probability = child.histogram[0]
            no_probability = child.histogram[2]
            i = 0
            for feature in example:
                if feature == "yes" or feature == "no":
                    continue
                if i == 0 or i == 2 or i == 4 or i == 10 or i == 11 or i == 12:
                    j = 0
                    # Its a float so check
                    for key in child.histogram[1][attribute_index[i]]:
                        if float(feature) <= float(key) and j == 0:
                            yes_probability = yes_probability * child.histogram[1][attribute_index[i]][key]
                        elif j == 1:
                            yes_probability = yes_probability * child.histogram[1][attribute_index[i]][key]
                        j += 1
                    for key in child.histogram[1][attribute_index[i]]:
                        if float(feature) <= float(key) and j == 0:
                            no_probability = no_probability * child.histogram[1][attribute_index[i]][key]
                        elif j == 1:
                            no_probability = no_probability * child.histogram[1][attribute_index[i]][key]
                        j += 1
                else:
                    yes_probability = yes_probability * child.histogram[1][attribute_index[i]][feature]
                    no_probability = no_probability * child.histogram[3][attribute_index[i]][feature]
                i += 1
            if yes_probability > no_probability:
                return "yes"
            elif no_probability >= yes_probability:
                return "no"
# -----------------------------------------------End Naive Bayes-------------------------------------------------
        else:
            if child.branch == testValue:
                return decision_tree_prediction(example, child.nextTree, attributes)


def find_prediction_success_rate(decision_tree, test_examples, attributes):
    """
    Finds the prediction success rate of the given test data using a decision tree.
    :param decision_tree: Decision tree used to predict example labels.
    :param test_examples: Given test data.
    :param attributes: All attributes in the data.
    :return: Prediction success rate of the input test data.
    """
    totalCorrect = 0
    for example in test_examples:
        actualResult = example[14]
        prediction = decision_tree_prediction(example, decision_tree, attributes)
        if prediction == actualResult:
            totalCorrect = totalCorrect + 1
    return totalCorrect / len(test_examples)


def print_test_predictions(name, decision_tree, test_examples, attributes):
    with open(name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Prediction"])
        id = 1
        for example in test_examples:
            prediction = decision_tree_prediction(example, decision_tree, attributes)
            if prediction == "yes":
                writer.writerow([str(id), "1"])
            elif prediction == "no":
                writer.writerow([str(id), "0"])
            else:
                print("error")
            id += 1


def new_tests(decision_tree, test_examples, attributes):
    """
    :param decision_tree:
    :param test_examples:
    :param attributes:
    :return:
    """
    success_count = 0
    for example in test_examples:
        correct_answer = example[len(example) - 1]
        prediction = decision_tree_prediction(example, decision_tree, attributes)
        if prediction == correct_answer:
            success_count += 1
        elif prediction == "yes":
            pass
        elif prediction == "no":
            pass
        else:
            print("error")
    return success_count/len(test_examples)


if __name__ == "__main__":
    attributes = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
                  "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week",
                  "native_country"]
    examples_set = import_examples('train_final.csv')
    label_prediction = predict_label(examples_set)
    test_examples = import_test_examples('test_final.csv')
    decision_tree = None
    # Prints prediction data for every depth
    name = "NaiveBayes.csv"
    decision_tree = ID3_algorithm(examples_set, label_prediction, attributes)
    print("Maximum tree depth: " + str(maximum_tree_depth))
    print("Average prediction Error for Training Data:" +
          str(find_prediction_success_rate(decision_tree, examples_set, attributes)))
    print_test_predictions(name, decision_tree, test_examples, attributes)

    for i in range(13):
        test_set = import_examples("personal_tests.csv")
        decision_tree = ID3_algorithm(examples_set, label_prediction, attributes)
        result = new_tests(decision_tree, test_set, attributes)
        print(str(maximum_tree_depth) + "\t\t\t\t\t\t\t\t" + str(result))
        maximum_tree_depth += 1
#        gain_threshold = gain_threshold - .0025
