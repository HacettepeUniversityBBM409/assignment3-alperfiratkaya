import argparse

from network import NeuralNetwork
import numpy as np


#
# Simple Argument Parser
#
def argumentParser():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("-b", "--batchsize", default='50')
    ap.add_argument("-e", "--epoch", default='30')
    ap.add_argument("-lr", "--learningrate", default='0.01')
    ap.add_argument('-hl', "--hiddenlayer", default=0)
    ap.add_argument('-a', "--activationfunction", default='softmax')
    ap.add_argument('-s', '--savemodel', default='none')
    ap.add_argument('-ld', '--loadmodel', default='none')
    ap.add_argument('-o', "--output", default='output.csv')
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
def load_data():
    train_data = np.load('train-data.npy')
    train_label = np.load('train-label.npy')

    validation_data = np.load('validation-data.npy')
    validation_label = np.load('validation-label.npy')

    test_data = np.load('test-data.npy')
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


def main():
    # Get the arguments
    args = argumentParser()
    # Load the data
    X_train_data, Y_train_label, X_validation_data, Y_validation_label, X_test_data = load_data()
    # initialize network with given parameters
    nn = NeuralNetwork(input_size=len(X_train_data[0]), output_size=10,
                       epoch=int(args['epoch']), batch_size=int(args['batchsize']),
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
    print(nn.score(X_validation_data, Y_validation_label))
    # predict the test file predictions
    prediction_data = nn.predict(X_test_data)
    # write predicted data to cvs file
    cvsFile = open(f"{args['output']}", 'w')
    for i in range(1, 10001):
        cvsFile.write(f'{i},{int(prediction_data[i-1])}\n')
    cvsFile.close()
    # save the model if needed
    if args['savemodel'] != 'none':
        nn.save_model(args['savemodel'])


if __name__ == '__main__':
    main()
