# script to analyze data using least-square fits of the autocorrelation function
# as more data is available, the fits are weighted by the standard error

import os
import collections
import pandas as pd
import numpy as np
import lmfit as lm
from lmfit.models import ExponentialModel
import matplotlib.pyplot as plt

def myfunction(t,amp,D):
    return amp*np.exp(-D*t/amp)

MyModel=lm.Model(myfunction)

results_dir='results/data100/'
N=100 # data length that is used for analysis
M=250 # number of points of the autocorrelation function that is used for fitting

data=pd.read_csv(results_dir+"data.csv")
data_length=len(data)
samples=int(data_length / N)

amplitude=[]
amplitude_stderr=[]
diffusion=[]
diffusion_stderr=[]

acf_sum=np.zeros(M)
acf_sumsq=np.zeros(M)

for i in range(samples):
    x=data['x'][N*i:N*(i+1)]
    f = np.fft.rfft(x)
    acf = np.fft.irfft(f * np.conjugate(f))
    acf = np.fft.fftshift(acf) / N
    autocorr=acf[int(N/2):]

    acf_sum=acf_sum+autocorr[:M]
    acf_sumsq=acf_sumsq+autocorr[:M]**2

    y = acf_sum/float(i+1)
    t = np.arange(M)
    s = np.sqrt(acf_sumsq/float(i+1)-y**2)/np.sqrt(i+1)
    if i==0: # stderr is not defined for the first iteration
        s = np.ones(M)

    out  = MyModel.fit(y, t=t, amp=1.0, D=1.0, weights=1/s)
    print(out.fit_report())
    amplitude.append(out.best_values['amp'])
    amplitude_stderr.append(np.sqrt(out.covar[0][0]))
    diffusion.append(out.best_values['D']*100)
    diffusion_stderr.append(np.sqrt(out.covar[1][1])*100)

n=np.arange(len(amplitude))

plt.figure()
plt.errorbar(t,y,yerr=s,fmt="ob")
plt.plot(t,out.best_fit)
plt.title('acf')

plt.figure()
plt.title('amplitude')
plt.errorbar(n,amplitude,yerr=amplitude_stderr,fmt="o")
plt.xlabel('iterations')
plt.ylabel('amplitude')

plt.figure()
plt.title('D')
plt.errorbar(n,diffusion,yerr=diffusion_stderr,fmt="o")
plt.xlabel('iterations')
plt.ylabel('D')

plt.show()


