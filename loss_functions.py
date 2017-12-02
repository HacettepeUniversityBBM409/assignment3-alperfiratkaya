import numpy as np


##
# Negative Log Likelihood
# #
def nll(Y_true, Y_predict):
    _, ones = np.where(Y_true == 1)
    loss = 0
    for i in range(len(ones)):
        loss += np.log(Y_predict[i, ones[i]])
    return (-1 / Y_true.shape[0]) * loss
