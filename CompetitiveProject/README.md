This is the competitive project developed by Sergio Remigio
for CS5350/6350 at the University of Utah
github url: https://github.com/SergioRemigio/DecisionTree

This file implements a Decision tree with a naive bayes classifier at the leaf nodes.
Naive bayes is incorporated into the decision tree by finding the entropy
at a node and deciding to create a naive bayes node only if the entropy at the current
node is less than a hyper parameter threshold.