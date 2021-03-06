"""hw2 Linear Methods: PLA and Linear Regression Implementations
completed as part of DSBA 6156 Machine Learning in the Spring of 2016 UNCC
Keith G. Williams
SID: 800690755
email: kwill229@uncc.edu
"""

# imports
import numpy as np
import pandas as pd

################################################################################
##    Required Functions: generateData(N), pla(X, Y, w0), pseudoinverse(X, Y)
################################################################################

def generateData(N):
    """Generates N 2-d points randomly sampled from the uniform distribution.
    A linear seperator is chosen as the target function, and all points above
    the line are labeled +1 and -1 otherwise.
    
    Parameters
    ----------
    n : integer 
        number of data points
    
    Returns
    -------
    (X, y) : tuple 
        where X is a n x 2 numpy matrix of points, 
        and y is the label vector in {-1, +1}."""
    
    # create n x 2 matrix of 2D points uniformly sampled in [-1, 1] x [-1, 1]
    x = 2 * np.random.random_sample(2 * N) - 1 # in [-1, 1]
    x = x.reshape((N, 2))
    
    # create target, linear function
    # choose two random points
    true_pts = x[np.random.choice(range(N), 2, False)]
    
    # y = mx + b
    m = (true_pts[0, 1] - true_pts[1, 1]) / (true_pts[0, 0] - true_pts[1, 0])
    b = true_pts[0, 1] - m * true_pts[0, 0]
    
    # create labels for each point in {-1, +1}
    above = x[:, 1] >= (x[:, 0] * m + b)
    labels = above.astype(int)
    labels[np.where(-above)] = -1 # convert 0 to -1
    
    return (x, labels)
    
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
    
    # preappend bias term to X
    X = np.c_[np.ones(n), X]
    
    # initialize weights to zero if none provided
    if w0 is None:
        w = np.zeros(X.shape[1])
    else:
        w = w0
        
    # update weight vector until convergence is achieved
    iters = 0
    converged = False
    while not converged:
        # check for convergence
        hypothesis = np.sign(np.dot(X, w))
        check = hypothesis != Y
        if check.sum() != 0:
            iters += 1
            # choose a misclassified point
            i = np.random.choice(np.where(check)[0])
            # update weight vector
            w += Y[i] * X[i]
        else:
            converged = True
            
    return (w, iters)
                
    
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
    X = np.c_[np.ones(np.size(Y)), X]
    
    pinv = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
    return np.dot(pinv, Y)

#################################################################################
##    Code for Running Required 12 Experiments
#################################################################################

def results_to_df(results, ns):
    """Export 2d array to dataframe for exporting to CSV
    
    Parameters
    ----------
    results: 6x2 3d numpy array with average results from experiment
    
    Returns
    -------
    Tidy pandas dataframe"""
    d = {}
    n = []
    for i in range(results.shape[0]):
        n += [ns[i]] * results.shape[1]
    d['n'] = n
    d['initialized'] = ['no', 'yes'] * results.shape[0]
    d['mean_iters'] = results.reshape(results.size)
    
    return pd.DataFrame(d)
    

def run_experiments():
    """Run 12 experiments as specified in Homework. 
       Export results to CSV for reporting"""
    TRIALS = 100                              # trials to run for each N
    n_list = [10, 50, 100, 200, 500, 1000]    # n for each experiment
    initialized = [False, True]          # boolean for initialized weight vector
       
    # results matrix of average iterations needed for PLA convergence
    results = np.zeros((len(n_list), len(initialized)))

    # collect the average iterations to convergence for PLA with and without
    # initialized weight vector from pseudoinverse.
    np.random.seed(6156) # set seed for replicability
    for n in range(len(n_list)):
        raw = np.zeros((TRIALS, 2))
        for i in range(TRIALS):
            training_data = generateData(n_list[n])
            for j in range(2):
                if initialized[j]:
                    w0 = pseudoinverse(training_data[0], training_data[1])
                    w, iters = pla(training_data[0], training_data[1], w0)
                else:
                    w, iters = pla(training_data[0], training_data[1])
                raw[i, j] = iters
        results[n] = np.mean(raw, axis=0)
    
    df = results_to_df(results, n_list)
    df.to_csv('hw2-mean-results.csv', index=False)

        
if __name__ == '__main__':
    run_experiments()