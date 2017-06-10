# Implementd by: Amitabh Nag
# amnag@uw.edu
# June 2017
# About: This module implements oja PCA algorithm.
# Module functions:
#       oja_pca - This function implements the stochastic gradient descent to compute the principal component using Oja's algorithm
#       deflate - After computing a principal component, to compute the next set of principal components/eigenvectors, technique called "deflation" is used
#       displayOjaPCAResult - display the PCA values for a particular component and the plot of objective vs iteration

from IPython.core.display import display
import numpy as np
import numpy.linalg as linalg
import matplotlib
import matplotlib.pyplot as plt
import copy
from sklearn.decomposition import PCA

def oja_pca(z,a0,neta0 = 0.001, t0 = 1,max_tries=1000):
    #Parameters:
        #z - centered data
        #a0 - starting point
        #neta0 - step size - numerator
        #t0 - step size - denominator
        #max-tries - maximum number of iterations
    t = 0
    z_oja = copy.deepcopy(z)
    a_oja = copy.deepcopy(a0)
    a_oja_10iter = np.zeros((10,np.size(z, 1)))
    iter = 0
    n = z_oja.shape[0]
    pca_objective_vals = np.zeros(max_tries)
    for k in range (0,max_tries):
        np.random.shuffle(z_oja)
        for i in range(0,n):
            z_i = z_oja[i, :]
            a_oja = a_oja + neta0/(t+t0)*np.dot(z_i,np.dot(z_i.T,a_oja))
            a_oja = a_oja / np.linalg.norm(a_oja)
            t = t+1
        if iter % 10 == 0:
            iter = 0
        a_oja_10iter[iter,:] = a_oja
        iter = iter + 1
        pca_objective_vals[k] = a_oja.dot(z_oja.T).dot(z_oja).dot(a_oja)/n
    a_oja = np.mean(a_oja_10iter,axis=0)
    return a_oja,pca_objective_vals

def deflate(z, a):
    #Parameters:
        #z - centered data
        #a - top eigenvectors/principal component
    return z - z.dot(np.outer(a, a))

def displayOjaPCAResult(a,pca_objective_vals,PCANum,max_iter = 1000):
    #Parameters:
        #a - top eigenvectors/principal component
        #pca_objective_vals - array that stores objective value per iteration
        #PCANum - principal component number
        #max_iter - max tries duing gradient descent
    print('Pincipal component #:', PCANum)
    plt.figure(PCANum)
    plt.plot(range(0,max_iter),pca_objective_vals)
    plt.xlabel('iteration')
    plt.ylabel('objective')
    plt.show()
    print('Pincipal component values:', a)
