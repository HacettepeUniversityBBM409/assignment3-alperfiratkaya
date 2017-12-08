import numpy as np

from activation_functions import softmax_function, derivative_softmax_function, sigmoid_function, \
    derivative_sigmoid_function, tanh_function, derivative_tanh_function, relu
from loss_functions import nll
import pickle


class NeuralNetwork:
    def __init__(self, input_size, output_size, learning_rate=0.01,
                 batch_size=50, epoch=30, hidden_layer_count=0,
                 activation_func='softmax', loss_func='nll'):
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_layer_count = hidden_layer_count
        self.model = self.initialize_weights()

        if activation_func == 'softmax':
            self.activation_func = softmax_function
            self.derivative_activation_func = derivative_softmax_function
        elif activation_func == 'sigmoid':
            self.activation_func = sigmoid_function
            self.derivative_activation_func = derivative_sigmoid_function
        elif activation_func == 'tanh':
            self.activation_func = tanh_function
            self.derivative_activation_func = derivative_tanh_function

        if loss_func == 'nll':
            self.lost_function = nll

    def fit(self, training_data, training_labels):
        for epoch in range(self.epoch):
            epoch_loss = list()
            for i in range(0, len(training_data), self.batch_size):
                # Batches
                X = training_data[i:i + self.batch_size]
                Y_true = training_labels[i:i + self.batch_size]
                # Feed Forward
                Z, Y_predict = self.feedforward(X)
                # Calculate Loss
                loss = nll(Y_true, Y_predict)
                epoch_loss.append(loss)
                # Backpropagation
                dB, dW = self.backpropagation(X, Y_true, Z)
                # Update Weights
                self.updateWeights(dB, dW)
            print(f"epoch :{epoch} loss {np.mean(epoch_loss)}")

    # Returns list of neuron counts
    # For example if no hidden_layer_count = 0
    # it will return [784,10]
    # if hidden_layer_count = 1
    # it will return [784,598,10]
    # and so on
    def hidden_layer_neuron_counts(self):
        neurons = list()
        neurons.append(self.input_size)
        for i in range(self.hidden_layer_count):
            neurons.append(int((neurons[-1] * 0.75) + 10))
        neurons.append(self.output_size)
        return neurons

    # initializes weights and biases
    # between 0 - 0.01
    def initialize_weights(self):
        neuronCounts = self.hidden_layer_neuron_counts()
        model = {}
        for i in range(len(neuronCounts)):
            if i + 1 == len(neuronCounts):
                break
            model[f"W{i}"] = np.random.normal(0, 0.01, (neuronCounts[i], neuronCounts[i + 1])).astype(np.float64)
            model[f"B{i}"] = np.random.normal(0, 0.01, neuronCounts[i + 1]).astype(np.float64)
        return model

    # feed forwards through network
    # relu(sum(xi * wi) + b) until the last layer
    # on last layer uses softmax
    def feedforward(self, X):
        Z = dict()
        Z["L0"] = X.dot(self.model['W0']) + self.model['B0']
        if self.hidden_layer_count == 0:
            Y_predict = self.activation_func(Z['L0'])
            return Z, Y_predict
        for i in range(self.hidden_layer_count):
            Z[f"L{i+1}"] = Z[f"L{i}"].dot(self.model[f"W{i+1}"]) + self.model[f"B{i+1}"]
            Z[f"L{i+1}"] = relu(Z[f"L{i+1}"])
        Y_predict = self.activation_func(Z[f"L{self.hidden_layer_count}"])
        return Z, Y_predict

    # its too long to explain
    def backpropagation(self, X, Y_true, Z):
        dB = list()
        dW = list()
        delta = self.derivative_activation_func(Z[f"L{self.hidden_layer_count}"], Y_true)
        if self.hidden_layer_count == 0:
            dB.append(self.learning_rate * delta)
            dW.append(self.learning_rate * X.T.dot(delta))
            return dB, dW
        dB.append(self.learning_rate * delta)
        dW.append(self.learning_rate * Z[f"L{self.hidden_layer_count-1}"].T.dot(delta))
        for i in reversed(range(self.hidden_layer_count)):
            if i - 1 == -1:
                break
            delta = relu(Z[f"L{i}"])
            dB.append(self.learning_rate * delta)
            dW.append(self.learning_rate * Z[f"L{i-1}"].T.dot(delta))
        delta = relu(Z["L0"])
        dB.append(self.learning_rate * delta)
        dW.append(self.learning_rate * X.T.dot(delta))

        return dB, dW

    # updates the weights according to
    # db and dW
    def updateWeights(self, dB, dW):
        for i in range(self.hidden_layer_count + 1):
            self.model[f"W{i}"] = self.model[f"W{i}"] - dW[-i - 1]
            self.model[f"B{i}"] = self.model[f"B{i}"] - dB[-i - 1]

    # predicts the given data
    # batch by batch
    def predict(self, data):
        res = list()
        for i in range(0, len(data), self.batch_size):
            X = data[i:i + self.batch_size]
            a, pred = self.feedforward(X)
            for i in pred:
                res.append(i.argmax())
        return res

    def score(self, test_data, test_label):
        res = self.predict(test_data)
        correct_res = 0
        for i in range(len(test_data)):
            if test_label[i] == res[i]:
                correct_res = correct_res + 1

        return correct_res / len(test_data)

    def save_model(self, filename):
        pickle.dump(self.model, open(f"{filename}.p", 'wb'))

    def load_model(self, filename):
        self.model = pickle.load(open(f"{filename}.p", 'rb'))
