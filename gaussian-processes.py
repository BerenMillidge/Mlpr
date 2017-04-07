#Gaussian processes. Let's do this and make up a demo. First steps first, we don't even do regression, just some simple function draws. then we will move onto regression, and having fun with kernels etc. In my free time post IRP and revision I should work through he GP book and do some bayesian optimisation, because that is fun!

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

#data here
x = []
y = []

#data creation functions
def cubic(x,a,b,c,d):
	return a*(x**3) + b*(x**2) + c*x + d

def quintic(x,a,b,c,d,e,f):
	return a*(x**5) + b*(x**4) + c*(x**3) + d*(x**2) + e*x + f

#kernel functions here
def exponential_kernel(xi,xj,l):
	#where l is the lengthscale hyperparameter
	return np.exp((-(xi -xj)**2)/l)

def gaussian_kernel(xi,xj,sigma,l):
	return (sigma**2)*np.exp(0.5*((xi-xj)**2)/(l**2)


def plot_GP_prior(l, kernel=exponential_kernel):
	#it has the l hyperparameter for the lenthscale of the function and a parameter for the kernel function you want to use
	#so the aim of this will just be to plot possible functions from our really basic GP prior which uses a mean of zero and an exponential/gaussian kernel function
	xgrid = np.linspace(-5,5,100) # these values can be changed
	#print xgrid.shape
	N = len(xgrid)
	#get zero mean vector
	means = np.zeros(N)
	#initialise covariance matrix
	cov = np.zeros([N,N])
	#fill in covariance matrix with kernel-derived values
	for i in xrange(N):
		for j in xrange(N):
			cov[i][j] = kernel(xgrid[i],xgrid[j],l)
		
	#draw from the GP prior distribution
	fvals = np.random.multivariate_normal(means,cov)	
	#plot the function
	plt.plot(xgrid,fvals, label="example function drawn from GP prior")	
	plt.show()

def perform_GP_inference(sigma_noise,x,f_star, l, kernel = exponential_kernel):
	#Given a set of known values - the Data - infer the posterior distribution over a set of unknown values - the test points. As we're using GPs, this can all be done analytically without variational or MCMC approximations, which is great!

	#first we need to get the covariances:
	#k(X,X)
	kXX = np.zeros([len(x), len(x)]) # initialise kXX covariance
	#fill in the kernel derived values
	for i in xrange(len(x)):
		for j in xrange(len(x)):
			kXX[i][j] = kernel(x[i],x[j], l)
	#create noise variance
	noise = np.diag(np.ones(len(x)))
	np.fill_diagonal(noise, np.sqrt(sigma_noise))
	#add noise
	kXX = kXX+ noise

	#covariance for kF_*X
	kF_starX = np.zeros([len(x), len(f_star)])
	#fill in the kernel derived values
	for i in xrange(len(x)):
		for j in xrange(len(f_star)):
			kF_starX[i][j] = kernel(x[i],f_star[j],l)
	#covariance for kF*F*
	kFF = np.zeros([len(f_star), len(f_star)])
	#fill in the kernel derived values
	for i in xrange(len(f_star)):
		for j in xrange(len(f_star)):
			kFF[i][j] = kernel(f_star[i],f_star[j],l)

	#now we derive the posterior distribution of p(f* | X)
	mu = np.dot(np.dot(kF_starX, np.linalg.inv(kXX)),x)
	cov = kFF - np.dot(np.dot(kF_starX, np.linalg.inv(kXX)),kF_starX.T)

	#we can now draw from the posterior if we want
	draw = np.random.multivariate_gaussian(mu,cov)
	return draw

def get_joint_posterior(x,f_star,l,sigma_noise,kernel = exponential_kernel)
	#get the joint posterior distribution of the old datapoints and new datapoints, if we have them whichc can be updated in a fairly easy way, just requiring a bunch of matrix arrangements.
	pass

def get_GP_errors_bars():
	pass

plot_GP_prior(0.01)


#we'll need to finish this up later, which shuold be pretty easy to do overall!
