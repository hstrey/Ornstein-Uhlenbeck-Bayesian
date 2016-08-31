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

results_dir='results/delta10-2/data2/'
N=100 # data length that is used for analysis
M=250 # number of points of the autocorrelation function that is used for fitting
delta_t=0.01
data=pd.read_csv(results_dir+"data.csv")
data_length=len(data)
samples=int(data_length / N)

alpha_A=[2.1]
beta_A=[1.1]
alpha_D=[2.1]
beta_D=[1.1]

for i in range(samples):
    x = np.array(data['x'][N * i:N * (i + 1)])

    d=np.diff(x) # calculate the difference
    v_sum=np.sum(d**2)

    a_sum=np.sum(d*x[:-1])+x[0]**2

    alpha_D.append(alpha_D[-1]+(N-1)/2.0)
    beta_D.append(beta_D[-1]+v_sum/4.0/delta_t)

    alpha_A.append(alpha_A[-1]+0.5)
    beta_A.append(beta_A[-1]+a_sum/2.0)

alpha_A=np.array(alpha_A)
beta_A=np.array(beta_A)

alpha_D=np.array(alpha_D)
beta_D=np.array(beta_D)

diffusion=beta_D/(alpha_D-1.0)
diffusion_var=beta_D**2/(alpha_D-2.0)/(alpha_D-1.0)**2

ampl=beta_A/(alpha_A-1.0)
ampl_var=beta_A**2/(alpha_A-2.0)/(alpha_A-1.0)**2

resultdict= dict(mean_A=ampl, std_A=np.sqrt(ampl_var), mean_D=diffusion, std_D=np.sqrt(diffusion_var))

df=pd.DataFrame(resultdict)
df.to_csv(results_dir+'results_analytic'+str(N)+'.csv',index=False)

n=np.arange(len(diffusion))

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
plt.title('Amplitude')
plt.errorbar(n,ampl,yerr=ampl_var,fmt="o")
plt.xlabel('A')
plt.ylabel('iterations')

plt.figure()
plt.title('A Variance')
plt.plot(n,ampl_var,"o")
plt.xlabel('iterations')
plt.ylabel('A var')

plt.show()


