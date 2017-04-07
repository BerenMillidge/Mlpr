# okay, here is where I do my logistic regression. Which should be fairly simple to implement once we have linear regression working!

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

#helper functions
def sigmoid(x):
	return 1/1-np.exp(x)

def calc_pred_at_point(x_point, w):
	return sigmoid(np.dot(w.T,x_point))

# DATA - put in your data here:
x = []
y = []

#get gradients for the negative log-likelihood cost function

def gradients_NLL(x,y,w):
	for i in xrange(len(y)):
		y[i] = 2*y[i] - 1 # reparametrisation trick, assumes that each ele of y is between 0 and 1
	grads = []
	for j in xrange(len(x)): # for each vector in the matrix
		grad = (1-sigmoid(np.dot(y[j]*w.T,x[j])))*y[j]*x[j]
		grads.append(grad)
	NLL_grad = -sum(grads)
	return NLL_grad

# get gradients using the least squares cost function
def gradients_least_squares(x,y,w):
	a = sigmoid(np.dot(x,w))
	b = sigmoid(np.dot(x.T, w.T))
	grad = -np.dot(np.dot(a,b),((1-a)-(1-b))) - np.dot(a,y) - np.dot(b,y)
	return grad


#solve by gradient descent!
def gradient_descent(x,y,w, N=100, var =0,lrate = 0.1):
	#initialise our weights with small gaussian noise values
	mu = np.zeros(len(x[0])
	cov = np.diag(np.ones(len(x[0])))
	np.fill_diagonal(cov, 0.1)
	w = np.random.multivariate_normal(mean,cov)

	#set the epochs running
	for i in xrange(N):
		if var ==0:
			grad = gradients_NLL(x,y,w)
		if var ==1:
			grad = gradient_least_squares(x,y,w)
		w = w + lrate * grad
	print w
	return w


def plot_result(w):
	#plot the graph of the regression line calculated against the true y values.
	z = np.linspace(min(x),max(x), 100*len(x))
	y_pred = []
	for i in xrange(len(z)):
		y_pred.append(sigmoid(np.dot(w.T,z[i])) # we're currently assuming x is 1D here, which is probably bad!
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.plot(y, label = "True function")
	ax.plot(y_pred, label = "Predicted function")
	ax.legend(loc = "top")
	fig.tight_layout()
	plt.show()
	return fig
	


