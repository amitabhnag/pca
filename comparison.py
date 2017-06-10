# Implementd by: Amitabh Nag
# amnag@uw.edu
# June 2017
# About: This module compares oja_pca library's PCA vs sklearn PCA. First three components from both methods are printed.
#        Additionally for oja PCA the plot between gradient descent iteration and objective function is also displayed
# Module functions:
#       skLearn_Oja_PCA_Comparison - Function computes the first three principal components from sklearn and oja library.
#       This function uses synthetic data set

from IPython.core.display import display
import numpy as np
import numpy.linalg as linalg
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
import pandas as pd
import sklearn.preprocessing as pre
from sklearn import linear_model
import copy
from sklearn.decomposition import PCA
import oja_pca_lib

#Generate synthetic data
np.random.seed(0)
x_class1 = np.random.normal(10,5,(50,50))
x_class2 = np.random.normal(20,5,(50,50))
x_class3 = np.random.normal(30,5,(50,50))
x_class4 = np.random.normal(40,5,(50,50))
x_class5 = np.random.normal(50,5,(50,50))
x_synthetic = np.concatenate((x_class1,x_class2,x_class3,x_class4,x_class5),axis=0)

#Function to compare oja_pca library and sklearn's PCA implementations
def skLearn_Oja_PCA_Comparison(x = x_synthetic):
    # Parameter:
        # x - data set
    #First three PCA components using sklearn
    pca = PCA(n_components=50, svd_solver='randomized')
    pca.fit(x)
    print('SkLearn - First principal component:', pca.components_[0])
    print('SkLearn - Second principal component:', pca.components_[1])
    print('SkLearn - Third principal component:', pca.components_[2])

    #First three PCA components using oja_pca library
    # Center the data
    z1 = x - np.mean(x, axis=0)
    # Generate a random starting point
    a0 = np.random.randn(np.size(z1, 1))
    a0 = a0 / np.linalg.norm(a0, axis=0)
    # Compute the first component
    a1,pca_objective_vals = oja_pca_lib.oja_pca(z1, a0,0.4,1,1000)
    oja_pca_lib.displayOjaPCAResult(a1,pca_objective_vals,1)

    # Compute the second component
    z2 = oja_pca_lib.deflate(z1, a1)
    a2,pca_objective_vals = oja_pca_lib.oja_pca(copy.deepcopy(z2), a0,0.4,1,1000)
    oja_pca_lib.displayOjaPCAResult(a2,pca_objective_vals,2)

    # Compute the third component
    z3 = oja_pca_lib.deflate(z2, a2)
    a3,pca_objective_vals = oja_pca_lib.oja_pca(copy.deepcopy(z3), a0,0.4,1,1000)
    oja_pca_lib.displayOjaPCAResult(a3,pca_objective_vals,3)

skLearn_Oja_PCA_Comparison()
