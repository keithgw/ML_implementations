"""hw2 Linear Methods: PLA and Linear Regression Implementations
completed as part of DSBA 6156 Machine Learning in the Spring of 2016 UNCC
Keith G. Williams
SID: 800690755
email: kwill229@uncc.edu
"""

# imports
import numpy as np
import matplotlib.pyplot as plt

def generateData(N):
    """Generates n 2-d points randomly sampled from the uniform distribution.
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
    # add bias term to x
    X = np.c_[np.ones(n), X]
    
    # initialize weights to zero if none provided
    if w0 is None:
        w = np.zeros(X.shape[1])
    else:
        w = w0
        
    # check for convergence
    iters = 0
    converge = False
    while not converge:
        hypothesis = np.sign(np.dot(X, w))
        check = hypothesis != Y
        if check.sum() != 0:
            iters += 1
            for i in xrange(n):
                # update weight vector
                y_hat = np.sign(np.dot(w, X[i]))
                w += (Y[i] - y_hat) / 2 * X[i]     #if correct, y-yhat = 0
        else:
            converge = True
            
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

def get_pts(w):
    """Get two points on the line predicted by a weight vector.
    
    Parameters
    ----------
    w : array_like
        weight vector
    
    Returns
    -------
    2*2 numpy array [(-1, y), (1, y)]"""
    return np.array([[-1, (-w[0] + w[1]) / w[2]], [1, (-w[0] - w[1]) / w[2]]])

TEST = True

def test():
    dat = generateData(50)
    h = pla(dat[0], dat[1])
    w = h[0]
    hpts = get_pts(w)
    
    pos = dat[0][dat[1] > 0]
    neg = dat[0][dat[1] < 0]
    
    plt.figure()
    plt.plot(pos[:, 0], pos[:, 1], 'ro')
    plt.plot(neg[:, 0], neg[:, 1], 'bo')
    plt.plot(hpts[:, 0], hpts[:, 1], 'k-')
    plt.show()
         
if __name__ == '__main__':
    if TEST:
        test()
    else:
        run()