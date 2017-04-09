#Linear regression experiments and implementation to check I actually understand what's going on and can implement in code.

from __future__ import division
import numpy as np
import math
import matplotlib.pyplot as plt

#Our data and data functions

def quadratic_fn(x, a, b,c):
	return a*(x**2) + b*x + c

def cubic_fn(x,a,b,c,d):
	return a*(x**3) + b*(x**2) + c*x + d

def add_gaussian_noise(x, sigma):	
	#adds gaussian noise to a point
	rand = np.random.normal(0,sigma)
	return x + rand

def get_linreg_line(xs,w):
	ys = []
	for x in xs:
		ys.append(np.dot(w.T,x))
	#plt.plot(ys)
	#plt.show()
	return ys
	

# create the design matrix for a simple function
x = np.linspace(-10, 10, 100)

#create our ys
y_quad = []
for i in x:
	y = quadratic_fn(i, 3,-5,2)
	y_quad.append(add_gaussian_noise(y, 1))
y_cube = []
for i in x:
	y = cubic_fn(i,2,-3,5,-2)
	y_cube.append(add_gaussian_noise(y,1))

#Do array massaging to get them into the correct shape
y_quad = np.array(y_quad) # turn list into array
y_quad = np.reshape(y_quad, [len(y_quad),1]) # turn from vector into 2D matrix
y_cube = np.array(y_cube) # turn list into array
y_cube = np.reshape(y_cube, [len(y_cube),1]) # turn from vector into 2D matrix
x = np.reshape(x, [len(x), 1]) # turn from vector into 2D matrix

#add biases to the design matrix
#b = np.ones(len(x))
#b = np.reshape(b, [len(b),1])
#x = np.concatenate((b,x),1)

def solve_linreg(x,y):
	#solve using inbuilt python functions
	w_solve = np.linalg.lstsq(x,y)[0]
	return w_solve

def solve_normal_equations(x,y):
	#solve using the normal equations method - implements w = (X^TX)^-1 * (X^Ty)
	xty = np.dot(x.T,y_quad)
	xtx = np.linalg.inv(np.dot(x.T,x))
	#print xtx.shape, xty.shape
	w_ols = np.dot(xtx,xty)
	return w_ols

def solve_gradient_descent(x,y, epochs = 100, lrate = 0.1):
	# solve using gradient descent- this should give approximately the same result as the error-space of the linear model is convex, with a single global optimum.
	
	#initialise our weights
	w_grad = [0.0005, 0.0001] # do a proper random gauss initialisaiton
	w_grad = np.array(w_grad)
	w_grad = np.reshape(w_grad,[len(w_grad),1])

	#run the iterative solver
	for i in xrange(epochs):
		xtx = np.dot(x.T,x)
		#print xtx.shape
		xtxw = np.dot(xtx,w_grad)
		#print xtxw.shape
		xty = np.dot(x.T,y)
		#print xty.shape
		grad = 2*xtxw - 2*xty
		#print grad
		w_grad = w_grad - lrate*grad
	return w_grad

#print solve_gradient_descent(x,y_quad)


def basis_function_expansion(X,N):
	#we'll just have this as a polynomial expansion. We can expand it to arbitrary functions later, if we want. N is the order of the polynomial and must be >=1
	#we're assuming the input x is one dimensional simple list of numbers
	phi = []
	for xs in X:
		xs = int(xs)	# we have to cast to int else weitrd stuff happens with the arrays
		row = []
		for i in xrange(N+1):
			row.append(xs**i)
		phi.append(row)
	phi = np.array(phi)
	print phi[1]
	print phi.shape
	return phi
	#this should work


X = basis_function_expansion(x, 5)
w =  solve_normal_equations(X,y_cube)

#Now let's just implement the plotting functionality!
def predict(x,w):
	#for an x and a weight vector, this gets our linear regression predictions. This is easy
	ys = []
	for xs in x:
		ys.append(np.dot(w.T,xs)) #I love how easy linear regression is to calculate
	return np.array(ys)

preds =  predict(X,w)

wlin = solve_normal_equations(x,y_cube)
preds_lin = predict(x,wlin)

print type(y_cube)
y_cube = np.array(y_cube)
print y_cube.shape

print type(preds_lin)
print type(preds)

#okay, now we've got everything we need imho
def plot_lines(x,true,preds_linear, preds_basis):
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	print x.shape, true.shape, preds_linear.shape, preds_basis.shape
	ax.plot(x,true, label="True function")
	ax.plot(x,preds_linear, label = "Standard linear regression line")
	ax.plot(x,preds_basis, label= "Basis function regression line")
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.legend()
	fig.tight_layout()
	plt.show()
	return fig

plot_lines(x,y_cube,preds_lin, preds)


##.// huh. well that's interesting. this doesn't seem to work AT ALL! Which I'm pretty not sure about. I'll have to work on this tomorrow to see where the bug is. At least we gotthe plotting functionality to work though! let's do some more theory now.

