"""hw1 k-Nearest Neighbors Implementation
completed as part of DSBA 6156 Machine Learning in the Spring of 2016 UNCC
Keith G. Williams
SID: 800690755
email: kwill229@uncc.edu
"""
# imports
import numpy as np              # for arrays and math
import pandas as pd             # for importing/exporting csv in tidy format
from scipy.stats import mode   # for fast mode implementation
from scipy.spatial.distance import pdist, squareform  # for pairwise distances
from time import clock          # for timer function
import gc                        # for uninterrupted timing

# constants
DATA_FILE = './letter-recognition.data' # should be in same directory as hw1.py
TRAIN_PROPORTION = 0.75 # proportion of data used to train classifier

#################################################################################
##    Helper Code for Required Functions
#################################################################################

def euclidean_dist(X, y):
    """Returns the distances between all n-dimensional vectors in X
    and n-dimensional vector y.
    inputs:
        x: m*n numpy array
        y: n-dimensional numpy array
    output:
        m-dimensional array of distances between vectors in X and vector y
    """
    return np.sqrt(np.sum((X - y) ** 2, 1)) # broadcasted calculations

#################################################################################
##    Required Functions: testknn(), condenseData()
#################################################################################

def testknn(trainX, trainY, testX, k=1):
    """implementation of k-NN
    inputs:
        trainX: n * D numpy array of training tuples
        trainY: n * 1 numpy array of training labels
        testX: n_test * D numpy array of test tuples
    returns:
        n_test * 1 numpy array of test labels
    """
    nTest = testX.shape[0]
    testY = []
    
    # iterate over test tuples
    for i in xrange(nTest):
        # compute distances to every point in trainX
        distances = euclidean_dist(trainX, testX[i])
        
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

def condenseData(trainX, trainY):
    """Finds a consistent subset of the training data whose 1-NN decision
        boundary correctly classifies all training data.
        inputs:
            trainX: n * D numpy array of training tuples
            trainY: n * 1 numpy array of training labels
        returns:
            numpy array of indices of consistent subset of training data
    """
    # get euclidean distance matrix
    edm = squareform(pdist(trainX))
    
    # initialize prototype subset
    ntrain = trainX.shape[0]
    classes = np.unique(trainY)
    condensedIdx = np.zeros(ntrain).astype(bool)
    
    for cls in classes:
        mask = trainY == cls
        rep = np.random.randint(0, np.sum(mask))
        condensedIdx[np.where(mask)[0][rep]] = True
    
    # slice edm to include only prototype subset
    edm_p = edm[condensedIdx]
    
    # label remaining points using 1-NN
    labels_t = trainY[condensedIdx]
    labels_h = labels_t[np.argmin(edm_p, 0)]

    # iterate over remaining points
    for i in range(ntrain):
        # if point is misclassified, add to prototype subset
        if labels_h[i] != trainY[i]: 
            condensedIdx[i] = True
            edm_p = edm[condensedIdx]
            labels_t = trainY[condensedIdx]
            labels_h = labels_t[np.argmin(edm_p, 0)] # 1-NN w/new prototype

    return np.where(condensedIdx)[0]

#################################################################################
##    Code for Running Required 60 Experiments
#################################################################################

def timer(trainX, trainY, testX, k, condensed=False):
    """timer function for comparing running times of NN algorithms. 
    Returns a tuple of run-time and predicted labels"""
    
    gc.disable() # disable garbage collector for uninterrupted timing    
    initial = clock()
    if condensed:
        cnn = condenseData(trainX, trainY)
        testY = testknn(trainX[cnn], trainY[cnn], testX, k)
    else:
        testY = testknn(trainX, trainY, testX, k)
    final = clock()
    
    gc.enable() # turn garbage collector back on
    return ((final - initial), testY)
    
def confusion_matrix(predictions, truth):
    """computes confusion matrix
    Input:
        predictions: array of predicted labels
        truth: array of known labels, aligned with predictions
    Output:
        c * c pandas dataframe, where c is the number of classes in labels"""
    # get class labels
    classes = np.unique(truth) # sorted
    
    # create an index for class labels
    class_index = dict((idx, cls) for cls, idx in enumerate(classes))
    
    # convert predictions and truth labels to indices
    pred_to_index = np.array([class_index[label] for label in predictions])
    truth_to_index = np.array([class_index[label] for label in truth])
    
    # create confusion matrix
    cmx = np.zeros((classes.size, classes.size))
    for i in xrange(truth.size):
        cmx[truth_to_index[i], pred_to_index[i]] += 1
    
    # return pandas dataframe where i -> truth, j -> prediction
    return pd.DataFrame(cmx, index=classes, columns=classes)
    
def accuracy(confusion_matrix):
    """computes accuracy given a confusion matrix
    Input:
        confusion_matrix as c * c numpy array, where c is the number of classes
    Output:
        float accuracy = N_correct / N"""
    return confusion_matrix.diagonal().sum() / confusion_matrix.sum()

def results_to_df(ary, ks, ns):
    """converts 4-d array of experimental results 
    into a pandas data frame for easy storage and analysis.
    Inputs:
        ary: 4d numpy array of experimental results
        ks: list of k values used for experiments
        ns: list of sample sizes used for experiments
    Output:
        n * 5 pandas data frame, where n is the number of experiments run"""
        
    # create columns as dictionaries
    results = {}
    results['algorithm'] = ['knn' for i in range(ary.size / 4)] + ['cnn' for j in range(ary.size / 4)]
    results['sample_size'] = ns * (2 * len(ks))
    k = []
    for ii in range(len(ks)):
        k += [ks[ii] for jj in range(len(ns))]
    results['k'] = k + k
    results['run_time'] = ary[0].reshape(60)
    results['accuracy'] = ary[1].reshape(60)
    
    return pd.DataFrame(results)

def run_experiments():
    # read-in data file
    df = pd.read_csv(DATA_FILE, header=None)
    
    # split data into train and test
    nTrain = int(TRAIN_PROPORTION * len(df))
    
    trainX = df.values[:nTrain, 1:].astype(float)
    trainY = df.values[:nTrain, 0].astype(str)
    testX = df.values[nTrain:, 1:].astype(float)
    test_labels = df.values[nTrain:, 0].astype(str)
    
    """4D matrix for storing results (i, j, k, n)
    i: {0: runtime, 1: accuracy}
    j: {0: knn, 1: cnn}
    k: value of k in {1, 3, 5, 7, 9}
    n: sample size in {100, 1e3, 2e3, 5e3, 1e4, 1.5e4}"""
    k_vals = [1, 3, 5, 7, 9]
    sample_sizes = [100, 1000, 2000, 5000, 10000, 15000]
    times_accuracies = np.zeros((2, 2, 5, 6))
    
    # run experiments, collect runtime and accuracy
    for k in range(len(k_vals)):
        for n in range(len(sample_sizes)):
            sample_indices = np.random.choice(len(trainX), sample_sizes[n], replace=False)
            for j in range(2):
                if j == 0:
                    cnn = False # flag for condensed 1-NN algorithm
                else:
                    cnn = True
                result = timer(trainX[sample_indices], trainY[sample_indices], testX, k_vals[k], condensed=cnn)                 
                cm = confusion_matrix(result[1], test_labels)
                times_accuracies[0, j, k, n] = result[0]
                times_accuracies[1, j, k, n] = accuracy(cm.values)
                
                # progress check
                print 'k:', k_vals[k], ' n: {:5d}'.format(sample_sizes[n]), \
                ' CNN: {:6}'.format(str(cnn)), \
                '{:02d}:{:06.3f}'.format(int(result[0] // 60), result[0] % 60) 
       
    # export results data to csv
    RUN = 6
    dat = results_to_df(times_accuracies, k_vals, sample_sizes)
    dat.to_csv('hw1-results-run{}.csv'.format(RUN), index=False)
           
if __name__ == "__main__":
    run_experiments()