"""
Homework 3
Completed as part of DSBA 6156 Machine Learning, Spring 2016
Keith G. Williams    kwill229@uncc.edu    800690755
"""
import numpy as np
import pandas as pd
from scipy import stats
import random as random

def split_train_test(x, y, p):
    """
    Partitions data into training and testing sets.
    
    Parameters
    ----------
    x : array like feature representation
    y : labels
    p : proportion of data to be used for training
    
    Returns
    -------
    list of arrays x_train, y_train, x_test, y_test
    """
    random.seed(6156)
    # listify x and y for list comprehension later
    xy = [x, y]
    
    # get number of training examples
    n = y.size
    cut = int(p * n)
    
    # shuffle the indices for slicing
    shuffled = np.random.permutation(n)
    
    # shuffle the data, and slice into p and 1-p partitions
    return [data[shuffled[:cut]] for data in xy] + [data[shuffled[cut:]] for data in xy]    
    

def p_label_subset(subset, weights=None):
    """given a subset, find the weighted probability of each value in the subset.
    
    Parameters
    ----------
    subset : array-like labels of a given subset of training examples
    weights : weight for each labeled example.
    
    Returns
    -------
    probs : float weighted probabilities of each label given the subset.
    """
    if weights is None:
        weights = np.array([1. / len(subset)] * len(subset))
        
    # get labels
    labels = np.unique(subset)
    probs = np.zeros(len(labels))
    
    for i in range(len(labels)):
        probs[i] = np.sum((subset == labels[i]) * weights)
        
    return (probs, labels)
        
def entropy(probs):
    """
    calculate the entropy given an array of probabilities
    """
    return np.sum(-probs * np.log2(probs))

def decision_stump(XTrain, YTrain, weights=None):
    """
    Creates a single-split decision tree for XTrain. 
    Assumes discrete features.
    
    Parameters
    ----------
    XTrain : the training examples, n*D numpy ndarray where N is the number of
        training examples and D is the dimensionality.
    YTrain : 1-D array of training labels.
    
    Returns
    -------
    split_idx : index for XTrain on which splitting results in the 
        lowest entropy.
    model : dictionary that maps feature value to prediction
    """    
    # get labels and training dimensions
    labels = np.unique(YTrain)
    n_examples = XTrain.shape[0]
    n_features = XTrain.shape[1]
    
    # set data weights if none provided
    if weights is None:
        weights = np.array([1. / n_examples] * n_examples)
    
    # initialize split entropy array
    entropies = np.zeros(n_features)
    
    # find weighted class entropy for each potential feature split
    for i in range(n_features):
        # get values in feature i
        entropy_sum = 0
        for _val in np.unique(XTrain[:, i]):
            mask = XTrain[:, i] == _val
            probs, dummy_l = p_label_subset(YTrain[np.where(mask)], weights[np.where(mask)])
            entropy_sum += mask.sum() / float(n_examples) * entropy(probs)
        entropies[i] = entropy_sum
    
    # index of feature whose split minimizes class entropy
    split_idx = np.where(entropies == np.min(entropies))[0][0]
    
    model = {}
    vals = np.unique(XTrain)
    for val in vals:
        mask = XTrain[:, split_idx] == val
        if mask.sum() == 0:
            p, l = p_label_subset(YTrain)
        else:
            p, l = p_label_subset(YTrain[np.where(mask)])
        
        model[val] = l[np.where(p == np.max(p))][0]
            
    return (split_idx, model)
    
def predict_stump(x, model):
    """
    Parameters
    ----------
    x : vector of x values
    model : dictionary that maps x to labels
    
    Returns
    -------
    hypothesis : vector of labels predicted by x given model
    """
    return np.array([model[x[i]] for i in range(len(x))])
    
#def labels_to_R(labels, label_vector):
#    """
#    Converts string labels to a real number in {-1, 1}
#    Parameters
#    ----------
#    labels : array like of string labels
#    label_vector : array like vector of labels to be converted
#    
#    Returns
#    -------
#    y : vector of -1 and 1 of same length as label_vector
#    """
#    mask = label_vector == labels[0]
#    y = np.ones(len(label_vector))
#    y[-mask] *= -1
#    return y

def adaPredict(model, XTest):
    """
    Parameters
    ----------
    model : model object returned by adaTrain()
    XTest : testing examples, n*D numpy ndarray where N is the number of testing
        examples and D is the dimensionality.
    
    Returns
    -------
    YTest : 1-D array of predicted labels corresponding to the provided test
        examples.
    """
    # Encode valuse as indices in model
    encoding = {'?' : 0, 'n' : 1, 'y' : 2}
    
    # compute ensemble predictions
    predict = np.zeros(XTest.shape)
    for i in range(XTest.shape[0]):
        for j in range(XTest.shape[1]):
            predict[i, j] = model[encoding[XTest[i, j]], j]
    
    # compute weighted predictions        
    weighted_predictions = np.sum(predict, 1)
    positive = weighted_predictions > 0
    negative = weighted_predictions < 0
    negative *= -1
    
    return negative + positive    

def adaTrain(XTrain, YTrain, version):
    """
    Parameters
    ----------
    XTrain : the training examples, n*D numpy ndarray where N is the number of
        training examples and D is the dimensionality.
    YTrain : 1-D array of training labels.
    version : option for learners, string 'stump', 'perceptron', or 'both'
        stump uses one-level decision trees that split on a single attribute
        perceptron uses pocket PLA
        
    Returns
    -------
    model : object containing the parameters of the trained model
    """
    # partition data into training and validation
    x_train, y_train, x_val, y_val = split_train_test(XTrain, YTrain, .75)
    
    # get shape of training data
    n = x_train.shape[0]
    p = x_train.shape[1]
    
    # set max number of iterations (T)
    TMAX = 50
    
    # initialize N vector of weights for each data sample
    weights = np.array([1. / n] * n)
        
    # get labels and convert to one of {-1, +1}
    labels = np.unique(y_train)
    label_to_R = {labels[0] : 1, labels[1] : -1}
    y = np.array([label_to_R[label] for label in y_train])
    
    # initialize model as a 3 * p matrix, where each value in XTrain
    # is indexed by i, and each feature is indexed by p
    vals = np.unique(x_train)
    encoding = {vals[i] : i for i in range(len(vals))}
    model = np.zeros((len(vals), p))
    
    # Train weak learners and update weights
    t = 0
    finished = False
    while not finished:
        # Train base learner using distribution weights
        feature, dt_model = decision_stump(x_train, y_train, weights)
        
        # Get base classifier h_t : X -> {-1, +1}
        predict = predict_stump(x_train[:, feature], dt_model)
        predictR = np.array([label_to_R[label] for label in predict])
        
        # Get classifier error
        compare = predictR * y
        misclassified = compare < 0
        e_train = np.mean(misclassified)
        
        # Choose alpha_t is a member of the reals
        alpha = np.log((1 - e_train) / e_train) / 2
        
        # Update ensemble model
        for key in dt_model:
            model[encoding[key], feature] += alpha * label_to_R[dt_model[key]]
        
        # Update distribution weights and renormalize
        weights *= np.exp(alpha * -compare)
        weights = weights / np.sum(weights)
                
        # Check if validation error is increasing or if t == TMAX
        predict_val = adaPredict(model, x_val)
        y_val_r = np.array([label_to_R[label] for label in y_val])
        e_val = np.mean(predict_val * y_val_r < 0)
        print e_val
            
        t += 1
        if t == TMAX:
            finished = True
        
    # Create model output
    return model
    
def run_experiments():
    pass

def testing():
    # test decision stump
    df = pd.read_csv('./house-votes-84.data', header=None)
    x, y = df.values[:, 1:], df.values[:,0]
    x[np.where(x=='y')] = 1
    x[np.where(x=='n')] = -1
    x[np.where(x=='?')] = 0
    #t = decision_stump(df.values[:,1:], df.values[:,0])
    #pred = [t[1][df.values[i, t[0] + 1]] for i in range(df.values.shape[0])]
    #print pred, df.values[:,0]
    #print np.mean(pred == df.values[:,0])
    #return t
    
    from sklearn.ensemble import AdaBoostClassifier
    bdt = AdaBoostClassifier(random_state = 6156)
    fitted = bdt.fit(x, y)

    return adaTrain(x, y, 'stump'), fitted
            
if __name__ == '__main__':
    run_experiments()