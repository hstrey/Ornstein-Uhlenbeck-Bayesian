# script to analyze data using least-square fits of the autocorrelation function
# as more data is available, the fits are weighted by the standard error

import os
import collections
import pandas as pd
import numpy as np
import lmfit as lm
from lmfit.models import ExponentialModel
import matplotlib.pyplot as plt
from scipy.stats import invgamma


def myfunction(t,amp,D):
    return amp*np.exp(-D*t/amp)

MyModel=lm.Model(myfunction)

results_dir='results/data100/'
N=100 # data length that is used for analysis
M=250 # number of points of the autocorrelation function that is used for fitting
delta_t=0.01
data=pd.read_csv(results_dir+"data.csv")
data_length=len(data)
samples=int(data_length / N)

amplitude=[]
amplitude_var=[]
diffusion=[]
diffusion_var=[]

a=[]

v_sum=0 # sum of (x_i+1 - x_i)**2

a_sum=0 # sum of (x_i+1 - x_i)*x_i

for i in range(samples):
    if i<samples-1:
        x=data['x'][N*i:N*(i+1)+1]
        nu=N*(i+1)
    else:
        x = data['x'][N * i:N * (i + 1)]
        nu=data_length-1
    d=np.diff(x) # calculate the difference
    v_sum=v_sum+np.sum(d**2)

    a_sum=a_sum+np.sum(d*x[:-1])

    a.append(a_sum/nu)

    v=v_sum/nu
    mean=nu/(nu-2)*v
    var=2*nu**2/(nu-2)**2/(nu-4)*v**2

    diffusion.append(mean/2/delta_t)
    diffusion_var.append(np.sqrt(var)/2/delta_t)

n=np.arange(len(diffusion))

d=np.linspace(0.8,1.2,100)
pd=invgamma.pdf(d,nu/2.0,scale=nu*v/4.0/delta_t)
print(pd)

plt.figure()
plt.title('diffusion')
plt.errorbar(n,diffusion,yerr=diffusion_var,fmt="o")
plt.xlabel('iterations')
plt.ylabel('D')

plt.figure()
plt.title('D Variance')
plt.plot(n,diffusion_var,"o")
plt.xlabel('iterations')
plt.ylabel('D var')

plt.figure()
plt.title('D distribution')
plt.plot(d,pd)
plt.xlabel('d')
plt.ylabel('p(d)')

plt.figure()
plt.title('a')
plt.plot(a)
plt.xlabel('iterations')
plt.ylabel('a')

plt.show()


