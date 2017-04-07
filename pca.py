#This is where I briefly implement my own PCA algorithm. I'll try to do this tonight, then later perhaps actually try it on some data to see how it works!

from __future__ import division
import numpy as np

x = [] # should be a NxD data matrix
K = 10 # the number of principal dimensions to extract!
def pca(x,K):
	#find the covariance matrix of X
	cov = np.dot(x.T,x)
	#find the eigenvalues of the covariance
	eigvals, eigvects = np.linalg.eig(cov)
	#get the first K eigenvectors and create our V
	V = []
	for i in xrange(K):
		V.append(eigvects[:,i])
	#reduce the data matrix through multiplication with V
	pca = np.dot(x,V)
	return pca

