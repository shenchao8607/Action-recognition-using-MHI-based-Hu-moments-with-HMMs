import math

__author__ = 'calp'

import matplotlib.pyplot as plt
import itertools
from sklearn import metrics
import numpy as np


def plotConfusionMatrix(expected, predicted, target_names):
    """
    Plot the confussion matrix.
    :param expected: Expected array
    :param predicted: Predicted array
    :param target_names: categories of data
    """
    cm = metrics.confusion_matrix(expected, predicted)
    np.set_printoptions(precision=2)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for x in range(cm.shape[0]):
        for y in range(cm.shape[1]):
            if math.isnan(cm[x][y]):
                cm[x][y] = 0
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "%.2f" % round(cm[i, j], 2), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def initByBakis(nComp, bakisLevel):
    """
    init start_prob and transmat_prob by Bakis model
    :param nComp: state number
    :param bakisLevel: bakis level
    :return: start and transmat matrix
    """
    startprobPrior = np.zeros(nComp)
    startprobPrior[0: bakisLevel - 1] = 1. / (bakisLevel - 1)
    transmatPrior = getTransmatPrior(nComp, bakisLevel)
    return startprobPrior, transmatPrior


def getTransmatPrior(nComp, bakisLevel):
    """
    get transmat prior
    :param nComp: state number
    :param bakisLevel: bakis level
    :return: transmat matrix
    """
    transmatPrior = (1. / bakisLevel) * np.eye(nComp)
    for i in range(nComp - (bakisLevel - 1)):
        for j in range(bakisLevel - 1):
            transmatPrior[i, i + j + 1] = 1. / bakisLevel

    for i in range(nComp - bakisLevel + 1, nComp):
        for j in range(nComp - i - j):
            transmatPrior[i, i + j] = 1. / (nComp - i)

    return transmatPrior
