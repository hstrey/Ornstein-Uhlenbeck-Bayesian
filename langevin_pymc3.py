import numpy as np
import matplotlib.pyplot as plt, seaborn as sns
import lmfit as lm
from scipy import signal
from lmfit.models import ExponentialModel
import pandas as pd

k,gamma,D = 1.0,1.0,1.0
delta_t=0.01
ampl = np.sqrt(2*D*delta_t)
N=200

datadict={}

# random force
w=np.random.normal(0,1,N)

# differential equation x_i = x_(i-1) - k/gamma*x_(i-1) + sqrt(2*D*delta_t)*w_i
from itertools import accumulate
def next_point(x,y):
    return x - k/gamma*x*delta_t + ampl*y

x = np.fromiter(accumulate(w, next_point),np.float)

datadict['x']=x

plt.figure()
plt.plot(x)

print("std: ",x.std(),"mean: ",x.mean())

# see http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.signal.fftconvolve.html
autocorr = signal.fftconvolve(x, x[::-1], mode='full')
n=len(autocorr)
print(n)
autocorr=autocorr[int((n-1)/2):]*2.0/(n+1)
datadict['acf']=autocorr

df=pd.DataFrame(datadict)
df.to_csv('data.csv')

mod = ExponentialModel()
y = autocorr[:200]
t = np.arange(200)

pars = mod.guess(y, x=t)
out  = mod.fit(y, pars, x=t)
print(out.fit_report(min_correl=0.25))

plt.figure()
plt.plot(t,y,"o")
plt.plot(t,out.best_fit)

# now lets model this data using pymc
import pymc3 as pm
# define the model/function for diffusion in a harmonic potential
DHP_model = pm.Model()
with DHP_model:
    t = pm.Exponential('t',0.1)
    A = pm.Exponential('A', 0.2)
    
    S=1.0-pm.exp(-4.0*delta_t/t)
    
    ss=pm.exp(-2.0*delta_t/t)
        
    path=pm.Normal('path_0',mu=0.0, tau=1/A, observed=x[0])
    for i in range(1,N):
        path = pm.Normal('path_%i' % i,
                            mu=path*ss,
                            tau=1.0/A/S,
                            observed=x[i])

with DHP_model:
    start = pm.find_MAP()
    print(start)
    trace = pm.sample(100000, start=start)

# save the data
tracedict={}
tracedict['t']=trace['t']
tracedict['t_log']=trace['t_log']
tracedict['A']=trace['A']
tracedict['A_log']=trace['A_log']

tdf=pd.DataFrame(tracedict)
tdf.to_csv('trace.csv')

pm.traceplot(trace)
pm.summary(trace)
plt.show()
