"""hw1 k-Nearest Neighbors Implementation
completed as part of DSBA 6156 Machine Learning in the Spring of 2016 UNCC
Keith G. Williams
SID: 800690755
email: kwill229@uncc.edu
"""

import numpy as np
import pandas as pd
from scipy.stats import mode
import random

# constants
DATA_FILE = './letter-recognition.data' # should be in same directory as hw1.py
TRAIN_PROPORTION = 0.75

def euclidean_dist(x, y):
    """Returns the distance between two n-dimensional vectors
    inputs:
        x: m*n numpy array
        y: n-dimensional numpy array
    output:
        array of distances between tuples in x and vector y
    """
    return np.sqrt(np.sum((x - y) ** 2, 1))

# k-nearest neighbors algorithm
def testknn(trainX, trainY, testX, k=1):
    """implementation of k-NN
    inputs:
        trainX: n * D numpy array of training tuples
        trainY: n * 1 numpy array of training labels
        testX: n_test * D numpy array of test tuples
    returns:
        n_test * 1 numpy array of test labels
    """
    nTest = len(testX)
    testY = []
    
    # iterate over test tuples
    for i in xrange(nTest):
        # compute distances to every point in trainX
        distances = euclidean_dist(trainX, testX[i, :])
        
        # assign label based on minimum distance(s)        
        if k == 1:
            label = trainY[np.argmin(distances)] # optimized for k=1
        
        # for k > 1
        else:
            k_shortest_distances = np.sort(distances)[:k]
            nn_labels = [trainY[np.where(distances == distance)[0][0]] for distance in k_shortest_distances]
            label = mode(nn_labels)[0][0]       # assign label by majority vote
    
        testY.append(label)
    
    return np.array(testY)
    
# condensed 1-NN algorithm
def condenseData(trainX, trainY):
    """Finds a consistent subset of the training data whose 1-NN decision
        boundary correctly classifies all training data.
        inputs:
            trainX: n * D numpy array of training tuples
            trainY: n* 1 numpy array of training labels
        returns:
            numpy array of indices of consistent subset of training data
    """
    # initialize subset with a single, random training example
    mask = np.zeros(len(trainX)).astype(bool)
    mask[random.randint(len(trainX)] = True
    
    # Classify all remaining samples using the subset, and transfer
    # an incorrectly classified sample to the subset
    transfer = True
    while transfer:
        testY = testknn(trainX[mask, :], trainY[mask], trainX[-mask, :], k=1)
        no_errors = True
        i = 0
        while no_errors:
            if trainY[unused_indices[i]] == 
            i += 1
            no_errors = False
        
        
    
    # Continue until no transfers occur or the subset is full

def run_experiments():
    # read-in data file
    df = pd.read_csv(DATA_FILE, header=None)
    
    # split data into train and test
    nTrain = int(TRAIN_PROPORTION * len(df))
    
    trainX = df.values[:nTrain, 1:].astype(float)
    trainY = df.values[:nTrain, 0].astype(str)
    testX = df.values[nTrain:, 1:].astype(float)
    test_labels = df.values[nTrain:, 0].astype(str)

if __name__ == "__main__":
    run_experiments()