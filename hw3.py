"""
Homework 3
Completed as part of DSBA 6156 Machine Learning, Spring 2016
Keith G. Williams    kwill229@uncc.edu    800690755
"""
#################################################################################
##    Imports
#################################################################################

import numpy as np
import pandas as pd

TRAIN_PROPORTION = 0.75

#################################################################################
##    Helper Code for Required Functions
#################################################################################

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
    
def pseudoinverse(X, Y):
    """Given feature matrix X, and continuous vector Y, return the learned
    weight vector, such that y_hat = (w.T)X gives the line of best fit that
    minimizes the squared error.
    
    Parameters
    ----------
    X : nD numpy array
        Input Data.
    Y : 1D numpy array of Real values
        target variable (labels)
        
    Returns
    -------
    w : 1D numpy array
        Learned weight vector that minimizes squared error.
    """
    # add bias term
    x = np.copy(X)
    x = np.c_[np.ones(np.size(Y)), x]
    
    pinv = np.dot(np.linalg.inv(np.dot(x.T, x)), x.T)
    return np.dot(pinv, Y)

################################################################################
##    Weak Learner Functions: Decision Stumps and Pocket PLA
################################################################################

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
    # get training dimensions
    #labels = np.unique(YTrain)
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
            entropy_sum += entropy(probs) * mask.sum() / float(n_examples)
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
    
def pla(X, Y, w0=None):
    """Perceptron learning algorithm is a linear classifier.
    
    Parameters
    ----------
    X : nD numpy array
        Input data.
    Y : 1D numpy array of labels in {-1, +1}
    w0: optional initial weight vector
    
    Returns
    ------
    (w, iters) : tuple
        w is the learned weight vector that separates -1 from +1
        iters is the number of iterations needed for convergence.
    """
    n = np.size(Y)
    max_iters = n * 100
    
    # preappend bias term to X
    x = np.c_[np.ones(n), X]
    
    # initialize weights to zero if none provided
    if w0 is None:
        w = np.zeros(x.shape[1])
    else:
        w = w0
    
    # initialize pocket weight and error
    w_pocket = np.copy(w)
    e_pocket = n    
                
    # update weight vector until max iterations achieved
    for i in range(max_iters):
        # get error count
        hypothesis = np.sign(np.dot(x, w))
        check = hypothesis != Y
        e_count = check.sum()
        if e_count == 0:
            break
        
        # update pocket weight if current weight yields a better error
        if e_count < e_pocket:
            e_pocket = e_count
            w_pocket = np.copy(w)
            
        # choose a misclassified point
        j = np.random.choice(np.where(check)[0])
        # update weight vector
        w += Y[j] * x[j]
            
    return w_pocket

################################################################################
##    Required Functions: adaPredict, adaTrain
################################################################################
      
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
    model_dt = model[0]
    model_pla = model[1]
    alpha = model[2]
    # Encode values as indices in model
    x = np.copy(XTest)
    
    n = x.shape[0]
    
    predict_pla = np.zeros(x.shape[0])
    predict_dt = np.zeros(x.shape)
    
    # iterate over each example
    for i in range(n):
        # calculate pla predictions
        h_pla = np.sign(np.dot(np.concatenate((np.array([1]), x[i, :])), model_pla.T))
        # multiply each prediction by alpha
        predict_pla[i] = np.sum(h_pla * alpha)
        
        # calculate dt predictions
        for j in range(x.shape[1]):
            #x[i, j] + 1 = index in model_dt matrix
            predict_dt[i, j] = model_dt[x[i, j] + 1, j] 
    
    # compute weighted predictions        
    weighted_predictions = predict_pla + np.sum(predict_dt, 1)
    positive = weighted_predictions > 0
    negative = weighted_predictions < 0
    negative = negative.astype(int) * -1
    
    return negative + positive.astype(int)

def adaTrain(XTrain, YTrain, version):
    """
    Parameters
    ----------
    XTrain : the training examples, n*D numpy ndarray where N is the number of
        training examples and D is the dimensionality.
    YTrain : 1-D array of real numbered training labels.
    version : option for learners, string 'stump', 'perceptron', or 'both'
        stump uses one-level decision trees that split on a single attribute
        perceptron uses pocket PLA
        
    Returns
    -------
    model : object containing the parameters of the trained model
    """
    # partition data into training and validation
    x_train, y_train, x_val, y_val = split_train_test(XTrain, YTrain, TRAIN_PROPORTION)
    
    # get shape of training data
    n = x_train.shape[0]
    p = x_train.shape[1]
    
    # set max number of iterations (T)
    TMAX = 25
    
    # initialize alphas
    alpha = np.zeros(TMAX)
    
    # initialize N vector of weights for each data sample
    weights = np.array([1. / n] * n)
    
    # initialize boosted tree model as a 3 * p matrix, 
    # where each value in XTrain is indexed by i, 
    # and each feature is indexed by p
    vals = np.unique(x_train)
    encoding = {vals[i] : i for i in range(len(vals))}
    btd_dt = np.zeros((len(vals), p))
    
    # initialize perceptron model as TMAX * p + 1 matrix
    # for weight vectors learned by perceptron
    # and the weight each vector has on final prediction
    btd_pla = np.zeros((TMAX, p + 1))
    
    # Train weak learners and update weights
    t = 0
    validation_errors = [1]
    finished = False
    while not finished:
        # Train base learner using distribution weights
        # decision trees
        feature, dt_model = decision_stump(x_train, y_train, weights)
        
        # pocket pla
        # create weighted sample training set
        sample_idx = np.random.choice(range(n), size=n, p = weights)
        btd_pla[t, :] = pla(x_train[sample_idx], y_train[sample_idx])    
                    
        # get base classifier h_t : X -> {-1, +1}
        predict_dt = predict_stump(x_train[:, feature], dt_model)
        e_dt = np.mean(predict_dt * y_train < 0)
        predict_pla = np.sign(np.dot(np.c_[np.ones(n), x_train], btd_pla[t, :]))
        e_pla = np.mean(predict_pla * y_train < 0)
            
        # Get classifier error
        if version == 'stump' or (version == 'both' and e_dt < e_pla):
            e_train = e_dt
            predict = predict_dt
            btd_pla[t, :] *= 0
        else:
            e_train = e_pla
            predict = predict_pla
        
        # Choose alpha_t is a member of the reals
        alpha[t] = np.log((1 - e_train) / e_train) / 2
        
        # Update ensemble model
        if version == 'stump' or (version == 'both' and e_dt < e_pla):
            for key in dt_model:
                btd_dt[encoding[key], feature] += alpha[t] * dt_model[key] 
        
        # Update distribution weights and renormalize
        update = -1 * y_train * alpha[t] * predict
        weights *= np.exp(update.astype('float64'))
        weights /= np.sum(weights)
                
        # Calculate validation errors
        predict_val = adaPredict((btd_dt, btd_pla, alpha), x_val)
        e_val = np.mean(predict_val * y_val < 0)
        validation_errors.append(e_val)
        
        # update t and check finish conditions
        t += 1
        if t == TMAX:
            finished = True
            # find minimum CV error
            cut = np.where(np.array(validation_errors) == min(validation_errors))[0][0]
            
            # roll back dt adjustments
            if version in ['stump', 'both']:
                for trn_rnd in range(cut, TMAX):
                    for key in dt_model:
                        btd_dt[encoding[key], feature] -= alpha[trn_rnd] * dt_model[key] 
        
    # Create model output
    return (btd_dt, btd_pla[:cut,:], alpha[:cut])
   
################################################################################
##    Main Function
################################################################################ 
      
def run_experiments():
    DATA_FILE = './house-votes-84.data'
    TRIALS = 10
    
    df = pd.read_csv(DATA_FILE, header=None)
    x, y = df.values[:, 1:], df.values[:,0]
    x[np.where(x == 'y')] = 1
    x[np.where(x == 'n')] = -1
    x[np.where(x == '?')] = 0
    y[np.where(y == 'democrat')] = 1
    y[np.where(y == 'republican')] = -1
    
    accuracies = {'stump' : [], 'perceptron' : [], 'both' : []}
    for version in accuracies:
        for _i in range(TRIALS):
            xtrain, ytrain, xtest, ytest = split_train_test(x, y, TRAIN_PROPORTION)
            boosted_model = adaTrain(xtrain, ytrain, version)
            predictions = adaPredict(boosted_model, xtest)
            accuracies[version].append(np.mean(predictions == ytest))
    
    df = pd.DataFrame(accuracies)
    df.to_csv('hw3-results.csv', index=False)

def testing():
    # test decision stump
    df = pd.read_csv(DATA_FILE, header=None)
    x, y = df.values[:, 1:], df.values[:,0]
    x[np.where(x == 'y')] = 1
    x[np.where(x == 'n')] = -1
    x[np.where(x == '?')] = 0
    y[np.where(y == 'democrat')] = 1
    y[np.where(y == 'republican')] = -1
    
    from sklearn.ensemble import AdaBoostClassifier
    #bdt = AdaBoostClassifier(random_state = 6156)
    #fitted = bdt.fit(x, y)

    return adaTrain(x, y, 'perceptron')#, fitted
            
if __name__ == '__main__':
    run_experiments()