import numpy as np
import matplotlib.pyplot as plt, seaborn as sns
import lmfit as lm
from scipy import signal
from lmfit.models import ExponentialModel

k,gamma,D = 1.0,1.0,1.0
delta_t=0.01
ampl = np.sqrt(2*D*delta_t)
N=500

# random force
w=np.random.normal(0,1,N)

# differential equation x_i = x_(i-1) - k/gamma*x_(i-1) + sqrt(2*D*delta_t)*w_i
from itertools import accumulate
def next_point(x,y):
    return x - k/gamma*x*delta_t + ampl*y

x = np.fromiter(accumulate(w, next_point),np.float)
plt.plot(x)

print("std: ",x.std(),"mean: ",x.mean())

# see http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.signal.fftconvolve.html
autocorr = signal.fftconvolve(x, x[::-1], mode='full')
n=len(autocorr)
print(n)
autocorr=autocorr[int((n-1)/2):]*2.0/(n+1)
plt.plot(autocorr[:1000])

mod = ExponentialModel()
y = autocorr[:100]
t = np.arange(100)

pars = mod.guess(y, x=t)
out  = mod.fit(y, pars, x=t)
print(out.fit_report(min_correl=0.25))

plt.plot(t,y,"o")
plt.plot(t,out.best_fit)

# now lets model this data using pymc
import pymc3 as pm
# define the model/function for diffusion in a harmonic potential
DHP_model = pm.Model()
with DHP_model:
    t = pm.Uniform('t', 0.1, 20)
    A = pm.Uniform('A', 0.1, 10)
    
    S=1-pm.exp(-4*delta_t/t)
    
    s=pm.exp(-2*delta_t/t)
        
    path=pm.Normal('path_0',mu=0, tau=1/A, observed=x[0])
    for i in range(1,N):
        path = pm.Normal('path_%i' % i,
                            mu=path*s,
                            tau=1/A/S,
                            observed=x[i])

with DHP_model:
    trace = pm.sample(10000)

pm.traceplot(trace)
pm.summary(trace)
plt.show()
