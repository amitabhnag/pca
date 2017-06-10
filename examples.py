# Implementd by: Amitabh Nag
# amnag@uw.edu
# June 2017
# About: This module provides an example of invoking oja pca library.
#        This module uses both synthetic and a real world data set while calling the oja PCA library
# Module functions:
#   runOjaPCA - Invokes the Oja library and computes the first three principal components

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

np.random.seed(0)
x_class1 = np.random.normal(10,5,(50,50))
x_class2 = np.random.normal(20,5,(50,50))
x_class3 = np.random.normal(30,5,(50,50))
x_class4 = np.random.normal(40,5,(50,50))
x_class5 = np.random.normal(50,5,(50,50))
x_synthetic = np.concatenate((x_class1,x_class2,x_class3,x_class4,x_class5),axis=0)

hitters = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Hitters.csv', sep=',', header=0)
hitters = hitters.dropna()
hitters = hitters.drop('Salary', axis=1)
hitters = pd.get_dummies(hitters, drop_first=True)
x_RealWorld =np.asarray(hitters)

def runOjaPCA(x):
    #Parameters:
        #x - data set
    # Center the data
    z1 = x - np.mean(x, axis=0)
    # Generate a random starting point
    a0 = np.random.randn(np.size(z1, 1))
    a0 = a0 / np.linalg.norm(a0, axis=0)
    #compute 1st principal component
    a1,pca_objective_vals = oja_pca_lib.oja_pca(z1, a0,0.4,1,2000)
    oja_pca_lib.displayOjaPCAResult(a1,pca_objective_vals,1,2000)

    #compute 2nd principal component
    z2 = oja_pca_lib.deflate(z1, a1)
    a2,pca_objective_vals = oja_pca_lib.oja_pca(copy.deepcopy(z2), a0,0.4,1,2000)
    oja_pca_lib.displayOjaPCAResult(a2,pca_objective_vals,2,2000)

    #compute 3rd principal component
    z3 = oja_pca_lib.deflate(z2, a2)
    a3,pca_objective_vals = oja_pca_lib.oja_pca(copy.deepcopy(z3), a0,0.4,1,2000)
    oja_pca_lib.displayOjaPCAResult(a3,pca_objective_vals,3,2000)

runOjaPCA(x_synthetic)
#runOjaPCA(x_RealWorld)

