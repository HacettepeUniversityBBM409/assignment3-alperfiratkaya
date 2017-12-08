import argparse

from network import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt


def visualize(W1):
    for i in range(10):
        img = np.reshape(W1[:, i], (28, 28))
        plt.imshow(img, cmap='Greys', interpolation='nearest')
        plt.show()


#
# Simple Argument Parser
#
def argumentParser():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("-td", "--traindata", default="train-data.npy", help="Path to train data")
    ap.add_argument("-tl", "--trainlabel", default="train-label.npy", help="Path to train labels")
    ap.add_argument("-vd", "--validationdata", default="validation-data.npy", help="Path to validation data")
    ap.add_argument("-vl", "--validationlabel", default="validation-label.npy", help="Path to validation labels")
    ap.add_argument("-ttd", "--testdata", default="test-data.npy", help="Path to test data -o can be used as well")
    ap.add_argument("-b", "--batchsize", default='1280', help="Value of the batch size")
    ap.add_argument("-e", "--epoch", default='10', help="Value of the epoch size")
    ap.add_argument("-lr", "--learningrate", default='0.01', help="Value of the learning rate")
    ap.add_argument('-hl', "--hiddenlayer", default=1, help="Value of the hidden layer size")
    ap.add_argument('-a', "--activationfunction", default='softmax', help="Specify the activation function choices "
                                                                          "are sigmoid tanh and softmax")
    ap.add_argument('-s', '--savemodel', default='none', help="saves the model for future use")
    ap.add_argument('-ld', '--loadmodel', default='none', help="loads the model")
    ap.add_argument('-o', "--output", default='output.csv', help="output file for test data")
    ap.add_argument("-v", "--visualize", default=False, help="Shows the visualizations of the weights set true if you "
                                                             "want to see it")
    return vars(ap.parse_args())


#
# normalizes the pixel values from 0-255
# to  0 - 1 #
# #
def normalize_values(train_data, validation_data, test_data):
    train_data = train_data.astype(np.float64)
    validation_data = validation_data.astype(np.float64)
    test_data = test_data.astype(np.float64)

    train_data = (train_data - np.mean(train_data)) / (10.0 * np.std(train_data))
    validation_data = (validation_data - np.mean(validation_data)) / (10.0 * np.std(validation_data))
    test_data = (test_data - np.mean(test_data)) / (10.0 * np.std(test_data))

    return train_data, validation_data, test_data


##
# Loads npy file
# and creates simple
# training label in the
# form of one-hot#
def load_data(args):
    train_data = np.load(args['traindata'])
    train_label = np.load(args['trainlabel'])

    validation_data = np.load(args['validationdata'])
    validation_label = np.load(args['validationlabel'])

    test_data = np.load(args['testdata'])
    train_data, validation_data, test_data = normalize_values(train_data, validation_data, test_data)
    new_train_label = list()
    for label in train_label:
        tmp = list()
        tmp.extend([0] * label)
        tmp.extend([1])
        tmp.extend([0] * (9 - label))
        new_train_label.append(tmp)
    train_label = np.array(new_train_label)
    return train_data, train_label, validation_data, validation_label, test_data


# predict the test file predictions
def predictTestFile(nn, X_test_data):
    prediction_data = nn.predict(X_test_data)
    # write predicted data to cvs file
    cvsFile = open(f"{args['output']}", 'w')
    for i in range(1, 10001):
        cvsFile.write(f'{i},{int(prediction_data[i-1])}\n')
    cvsFile.close()


def main():
    # Get the arguments
    args = argumentParser()
    # Load the data
    X_train_data, Y_train_label, X_validation_data, Y_validation_label, X_test_data = load_data(args)
    # initialize network with given parameters
    nn = NeuralNetwork(input_size=len(X_train_data[0]), output_size=10,
                       epoch=int(args['epoch']), batch_size=10,
                       activation_func=args['activationfunction'],
                       learning_rate=float(args['learningrate']),
                       hidden_layer_count=int(args['hiddenlayer']))
    # if needed load a saved model
    if args['loadmodel'] != 'none':
        nn.load_model(args['loadmodel'])
    else:
        # shuffle the data
        order = np.random.permutation(X_train_data.shape[0])
        X_train_data = X_train_data[order, :]
        Y_train_label = Y_train_label[order, :]
        # create the model
        nn.fit(X_train_data, Y_train_label)

    # test with validation data
    print("accuracy : ", nn.score(X_validation_data, Y_validation_label))

    # Uncomment if you want to predict the test file
    # predictTestFile(nn,X_test_data)

    # save the model if needed
    if args['savemodel'] != 'none':
        nn.save_model(args['savemodel'])

    if args['visualize'] and nn.hidden_layer_count == 0:
        visualize(nn.model["WO"])
    else:
        print("Visualization is usable when hidden layer count is 1")


if __name__ == '__main__':
    main()
