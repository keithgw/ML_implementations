"""hw1 k-Nearest Neighbors Implementation
completed as part of DSBA 6156 Machine Learning in the Spring of 2016 UNCC
Keith G. Williams
SID: 800690755
email: kwill229@uncc.edu
"""

import numpy as np              # for arrays and math
import pandas as pd             # for importing/exporting csv
from scipy.stats import mode   # for fast mode implementation
from scipy.spatial.distance import pdist, squareform
from time import clock          # for timer function
import gc                        # for uninterrupted timing

# constants
DATA_FILE = './letter-recognition.data' # should be in same directory as hw1.py
TRAIN_PROPORTION = 0.75 # proportion of data used to train classifier

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
    nTest = int(testX.shape[0])
    testY = []
    
    # iterate over test tuples
    for i in xrange(nTest):
        # compute distances to every point in trainX
        #distances = euclidean_dist(trainX, testX[i, :])
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

def distance_matrix(trainX, condensedIdx):
    dist_mtrx = np.zeros((np.sum(condensedIdx), np.sum(-condensedIdx)))
    for i in xrange(np.sum(condensedIdx)):
        dist_mtrx[i] = euclidean_dist(trainX[-condensedIdx], trainX[condensedIdx][i])
    return dist_mtrx

def cnn_centroids(trainX, trainY):
    # get euclidean distance matrix
    edm = squareform(pdist(trainX))
    #edm += (np.eye(edm.shape[0]) * (np.max(edm) + 1)) # replace diagonal with max + 1
    
    # initialize
    ntrain = trainX.shape[0]
    classes = np.unique(trainY)
    condensedIdx = np.zeros(ntrain).astype(bool)
    
    for cls in classes:
        mask = trainY == cls
        rep = np.random.randint(0, np.sum(mask))
        condensedIdx[np.where(mask)[0][rep]] = True
    
    # slice edm to include only prototype points
    edm_p = edm[condensedIdx]
    
    # knn
    labels_t = trainY[condensedIdx]
    labels_h = labels_t[np.argmin(edm_p, 0)]

    # iterate over remaining points
    for i in range(ntrain):
        # if point is misclassified, add to prototypes
        print i
        if labels_h[i] != trainY[i]: 
            print labels_h[i], trainY[i]
            condensedIdx[i] = True
            edm_p = edm[condensedIdx]
            labels_t = trainY[condensedIdx]
            labels_h = labels_t[np.argmin(edm_p, 0)]
    
    print np.sum(condensedIdx)
    return np.where(condensedIdx)[0]

def fcnn(trainX, trainY):
    # initialize
    condensedIdx = np.zeros(len(trainX)).astype(bool)
    condensedIdx[np.random.randint(len(trainX))] = True
    
    transfer = True
    run = 0
    while transfer:
        print 'run:', run
        run += 1
        # find distances of all prototypes to all absorbed points
        nproto = np.sum(condensedIdx)
        nabsorbed = np.sum(-condensedIdx)
        dist_mtrx = np.zeros((nproto, nabsorbed))
        for i in xrange(nproto):
            dist_mtrx[i] = euclidean_dist(trainX[-condensedIdx], trainX[condensedIdx][i])
            
        # label each prototype
        labels_t = trainY[condensedIdx]
        labels_h = labels_t[np.argmin(dist_mtrx, 0)]
        
        # find closest points to each prototype
        sorted_dist = np.sort(dist_mtrx.reshape(dist_mtrx.size))
        
        # check for incorrect label, and add to prototype set
        j = 0
        transfer = False
        while not transfer and j < dist_mtrx.size:
            p_a = np.where(dist_mtrx == sorted_dist[j])
            p_i, a_j = p_a[0][0], p_a[1][0]
            print trainY[np.where(-condensedIdx)[0][a_j]], labels_h[p_i]
            if trainY[np.where(-condensedIdx)[0][a_j]] != labels_h[p_i]:
                condensedIdx[np.where(-condensedIdx)[0][a_j]] = True
                transfer = True
                print np.sum(condensedIdx), transfer
            else:
                j += 1
                   
    return np.where(condensedIdx)[0]
                
            
            
# condensed 1-NN algorithm
def cnnf(trainX, trainY):
    """Finds a consistent subset of the training data whose 1-NN decision
        boundary correctly classifies all training data.
        inputs:
            trainX: n * D numpy array of training tuples
            trainY: n * 1 numpy array of training labels
        returns:
            numpy array of indices of consistent subset of training data
    """
    # initialize subset with a single, random training example
    condensedIdx = np.zeros(len(trainX)).astype(bool)
    condensedIdx[np.random.randint(len(trainX))] = True
    
    # loop over remaining training patterns
    transfer = True
    while transfer:
        # Check if unused subset is empty
        if np.sum(-condensedIdx) == 0:
            transfer = False
        else:
            # test absorbed points one at a time until a mislabeled one is found
            for i in np.where(-condensedIdx)[0]:
                #print trainX[i, :]
                #print trainX[condensedIdx]
                testY = testknn(trainX[condensedIdx, :], trainY[condensedIdx], trainX[i], k=1)
                transfer = False
                
                if testY[0] != trainY[i]:
                    condensedIdx[i] = True
                    transfer = True
                    break
            
    
    return np.where(condensedIdx)[0]       
            
    
    # if incorrectly classified, add it to consistent subset

# condensed 1-NN algorithm
def condenseData(trainX, trainY):
    """Finds a consistent subset of the training data whose 1-NN decision
        boundary correctly classifies all training data.
        inputs:
            trainX: n * D numpy array of training tuples
            trainY: n * 1 numpy array of training labels
        returns:
            numpy array of indices of consistent subset of training data
    """
    # initialize subset with a single, random training example
    condensedIdx = np.zeros(len(trainX)).astype(bool)
    condensedIdx[np.random.randint(len(trainX))] = True
    
    # add to subset until consistent subset is achieved
    transfer = True     # flag to test if a point is added to consistent set
    while transfer:
        # Check if unused subset is empty
        if np.sum(-condensedIdx) == 0:
            transfer = False
        else:
            # get predicted labels using 1-NN for points outside of consistent set
            testY = testknn(trainX[condensedIdx, :], trainY[condensedIdx], trainX[-condensedIdx, :], k=1)
            
            # create a dictionary to keep track of (index, label) tuples
            # where index is of trainX and not in consistent subset
            index_label = {}
            for idx, label in enumerate(testY):
                index_label[np.where(-condensedIdx)[0][idx]] = label
                
            # find a predicted label that is incorrect
            transfer = False # if no incorrect label is found, alg will finish
            for key in index_label:
                if index_label[key] != trainY[key]:
                    condensedIdx[key] = True   # add point to consistent subset
                    transfer = True
                    break
                
    return np.where(condensedIdx)[0]
    
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
    
    # convert predictions and truth to indexes
    pred_to_index = np.array([class_index[label] for label in predictions])
    truth_to_index = np.array([class_index[label] for label in truth])
    
    # create confusion matrix
    cmx = np.zeros((classes.size, classes.size))
    for i in xrange(truth.size):
        cmx[truth_to_index[i], pred_to_index[i]] += 1
    
    # return pandas dataframe indexed by classes x classes
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
        array: 4d array of experimental results
        ks: list of k values used for experiments
        ns: list of sample sizes used for experiments
    Output:
        n * 5 pandas data frame, where n is the number of experiments run"""
        
    # create columns as dictionary
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
    i: runtime/accuracy
    j: knn/cnn
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
                t = clock()
                print 'k:', k_vals[k], 'n:', sample_sizes[n], ', CNN:', cnn, '{} min'.format((clock() - t) / 60) 
                
                
    
    # export results data to csv
    run_number = 3
    dat = results_to_df(times_accuracies, k_vals, sample_sizes)
    dat.to_csv('hw1-results-run{}.csv'.format(run_number), index=False)

def timecnn(x, y, fxn):
    gc.disable() # disable garbage collector for uninterrupted timing    
    initial = clock()
    
    ci = fxn(x, y)
    
    final = clock()
    
    gc.enable() # turn garbage collector back on
    return final - initial

def testcnn(n):
    # read-in data file
    df = pd.read_csv(DATA_FILE, header=None)
    
    # split data into train and test
    nTrain = int(TRAIN_PROPORTION * len(df))
    
    trainX = df.values[:nTrain, 1:].astype(float)
    trainY = df.values[:nTrain, 0].astype(str)
    
    #old_times = []
    new_times = []
    for i in range(n):
        #old_times.append(timecnn(trainX, trainY, condenseData))
        new_times.append(timecnn(trainX, trainY, cnn_centroids))
        
    #return (np.median(old_times), np.median(new_times))
    return new_times
    
            
if __name__ == "__main__":
    #run_experiments()
    print testcnn(1)