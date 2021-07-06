#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#import modules, libraries and packages
import numpy as np
import pandas as pd
from math import log, sqrt
import math
import scipy
from scipy import special
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def tail_body_likelihood(params,r,pen):

    l = params[0]; alpha = params[1]; xmin = params[2]; # l for lambda from the exponential distribution, alpha for the tail exponent, xmin the point we expect power law behaviour to start   
    
    C = 2 - np.exp(-l*xmin); # Normalization constant
    
    L = 0; # Initializing likelihood
   
    ### Finding returns lower than xmin
    r = np.sort(r);
    f = len(np.abs(r[r<xmin]))
    
    ### Computing log-likelihood values for the body of the distribution 
    for i in range(f):
        L = L - np.log(C) + np.log(l) - l*r[i];
        ### Computing log-likelihood values for the tail of the distribution     
    for i in range(f+1,len(r)):
        L = L - np.log(C) + np.log(alpha/xmin) - (alpha+1)*np.log(r[i]/xmin);
    
    ### Computing values of body and tail distributions at xmin
    Fbody = l*np.exp(-l*xmin)/C;
    Ftail = alpha/(C*xmin);
    
    ### Adding penalty term to enforce continuity
    L = L - pen*(Fbody-Ftail)**2;
    
    ### Changing sign to log-likelihood (fminsearch algorithm searches for minima)
    L = - L;
    
    return L

#Import data
file = open('cryptocurrency_prices.txt','r')
crypto = np.loadtxt(file)
file.close()

# Choose data
BTC = crypto[0]

log_returns = np.log(BTC[1:]/BTC[0:-1]);

r = np.abs(log_returns[log_returns<0]) 
# Selecting only negative log_returns. Then computing abs values  

# Maximum likelihood optimization

pen = 1000; # Penalty parameter (to enforce continuity in xmin)

lambda_ = 10; alpha = 2; xmin = 0.05; 
params = [lambda_, alpha, xmin]; # this is the first guess for the params vector

par = scipy.optimize.fmin(tail_body_likelihood,x0=params,args=(r,pen),maxiter=1000,maxfun=5000,xtol=1e-12) 

print('\nLambda:{0}\nAlpha: {1}\nxMin: {2}'.format(par[0],par[1],par[2]))

lambda_ = par[0]; alpha = par[1]; xmin = par[2];

C = 2 -np.exp(-lambda_*xmin) # Normalisation

NB = 40; # number of bins

plot1 = plt.figure(1,figsize=(7,4), dpi=100)
a,b = np.histogram(r,NB,density=True);
plt.loglog(b[1:],a,'b.');
x = np.linspace(0,xmin,1000);
plt.loglog(x,lambda_*np.exp(-lambda_*x)/C,'-r')
x = np.linspace(xmin,max(r),1000);
plt.loglog(x,alpha*(x/xmin)**(-alpha-1)/(C*xmin),'-r')
plt.axvline(xmin)
plt.xlim =(min(r),max(r));
plt.title('Histogram vs. analytical PDF');

plot1 = plt.figure(2,figsize=(7,4), dpi=100)
r = np.sort(r,axis=None); # Returns sorted in ascending order
y = np.arange(0,len(r),1); 
y = 1-y/(len(r)+1); # Calculating CCDF as rank-frequency plot
plt.plot(r,y,'b','LineWidth',1.5)
x = np.linspace(np.min(r),xmin,1000);
plt.plot(x,1-(1-np.exp(-lambda_*x))/C,'-r','LineWidth',1.5)
x = np.linspace(xmin,max(r)*10,1000);
plt.plot(x,1-(1-np.exp(-lambda_*xmin)+1-(xmin/x)**alpha)/C,'-r','LineWidth',1.5)
plt.axvline(xmin);
plt.xscale('log');
plt.xlim = ([np.min(r),np.max(r)]);
plt.ylim = ([0,1]);
plt.title('Empirical CDF vs. analytical CDF');
