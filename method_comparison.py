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

results_dir='results/length100/'

filelist=os.listdir(results_dir)

trace_list=[]
data_list=[]

for file in filelist:
    if file.startswith('trace') and file.endswith('.csv'):
        trace_list.append(file)
    if file.startswith('data') and file.endswith('.csv'):
        data_list.append(file)

samples=len(trace_list)
print(data_list)

amplitude=[]
amplitude_stderr=[]
diffusion=[]
diffusion_stderr=[]

for i,datafile in enumerate(data_list):
    data=pd.read_csv(results_dir+datafile)
    if i==0:
        acf_sum=data['acf']
        acf_sum_squares=data['acf']**2
        acf_stderr=np.ones(len(data['acf']))
    else:
        acf_sum+=data['acf']
        acf_sum_squares+=data['acf']**2
        acf_stderr=np.sqrt(acf_sum_squares/float(i+1)-(acf_sum/float(i+1))**2)/np.sqrt(float(i+1))
        print(acf_stderr)

    y = acf_sum[:100]/float(i+1)
    t = np.arange(100)
    s = acf_stderr[:100]

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


