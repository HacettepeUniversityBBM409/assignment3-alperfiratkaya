import numpy as np


##
# Sigmoid Function
# f(x) =  1 / (1 + e ^ -x) #
def sigmoid_function(x):
    return 1 / (1 + np.exp(x))


def derivative_sigmoid_function(x):
    return sigmoid_function(x) * (1 - sigmoid_function(x))


##
# Hyperbolic Tangent Function
# f(x) = (1 - e ^ -2x) / (1 + e ^ -2x)
def tanh_function(X):
    return np.tanh(X)


def derivative_tanh_function(x):
    return 1.0 - np.tanh(x) ** 2


def relu(X):
    return X * (X > 0)


def derivative_relu(X):
    X[X <= 0] = 0
    X[X > 0] = 1
    return X


##
# SoftMax #
def softmax_function(X):
    expA = np.exp(X - np.max(X))
    return expA / expA.sum(axis=1, keepdims=True)


def derivative_softmax_function(X, y):
    y_ = np.argmax(y, axis=1)
    num_example = X.shape[0]
    probs = softmax_function(X)
    for i in range(num_example):
        probs[i, y_[i]] -= 1
    return probs
