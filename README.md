This project provides Oja PCA implementation. Additionally it provides code comparison with sklearn's PCA implementation. Example code with synthetic and real world data is included. 

PCA involves computing the top eigenvectors of the empirical covariance matrix of the data. The data is assumed to be centered. Oja algorithm is one of the ways the eigenvectors, which is stochastic gradient descent applied to PCA.

Files included in this repository:

## oja_pca_lib.py 
This module implements oja PCA algorithm

## examples.py
This module provides an example of invoking oja pca library. This module uses both synthetic and a real world data set while invoking the oja PCA library 

## comparison.py
This module compares oja_pca library's PCA vs sklearn PCA. First three components from both methods are printed. Additionally for oja PCA the plot between gradient descent iteration and objective function is also displayed

## Quick Start:

```python
#Center the input data (x)
z1 = x - np.mean(x, axis=0)

#Generate a random starting point
a0 = np.random.randn(np.size(z1, 1))
a0 = a0 / np.linalg.norm(a0, axis=0)

#Compute the first component
#Parameters of oja_pca_lib.oja_pca
#   z - centered data
#   a0 - starting point
#   neta0 - step size - numerator
#   t0 - step size - denominator
#   max-tries - maximum number of iterations
a1,pca_objective_vals = oja_pca_lib.oja_pca(z1, a0,0.4,1,1000)
#Output of oja_pca_lib.oja_pca
#   a1- principal components
#   pca_objective_vals - array with objective value per iteration

#Display and print the results
#Parameters for oja_pca_lib.displayOjaPCAResult
#   a - top eigenvectors/principal component
#   pca_objective_vals - array with objective value per iteration
#   PCANum - principal component number
#   max_iter - max tries duing gradient descent
oja_pca_lib.displayOjaPCAResult(a1,pca_objective_vals,1)
```
