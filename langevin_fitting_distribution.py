import numpy as np
import matplotlib.pyplot as plt, seaborn as sns
import lmfit as lm
from scipy import signal
from lmfit.models import ExponentialModel
import pandas as pd
from itertools import accumulate

# differential equation x_i = x_(i-1) - k/gamma*x_(i-1) + sqrt(2*D*delta_t)*w_i
def next_point(x,y):
    return x - k/gamma*x*delta_t + ampl*y

k,gamma,D = 1.0,1.0,1.0
delta_t=0.01
ampl = np.sqrt(2*D*delta_t)
N=500
M=10000
t_list=[]
A_list=[]
mean_list=[]
std_list=[]
mod = ExponentialModel()

for i in range(M):
    # random force
    w=np.random.normal(0,1,N)
    x = np.fromiter(accumulate(w, next_point),np.float)

    # see http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.signal.fftconvolve.html
    autocorr = signal.fftconvolve(x, x[::-1], mode='full')
    n=len(autocorr)
    autocorr=autocorr[int((n-1)/2):]*2.0/(n+1)
    y = autocorr[:200]
    t = np.arange(200)

    pars = mod.guess(y, x=t)
    out  = mod.fit(y, pars, x=t)
    #print(out.fit_report(min_correl=0.25))
    t_list.append(out.values['decay'])
    A_list.append(out.values['amplitude'])
    mean_list.append(x.mean())
    std_list.append(x.std())
    print('mean: ',x.mean(),'std: ',x.std(),'amplitude: ',out.values['amplitude'],'decay: ',out.values['decay'])

t_list=np.array(t_list)
A_list=np.array(A_list)
mean_list=np.array(mean_list)
std_list=np.array(std_list)

#eliminate outliers e.g. negative decay and very long decay
t_list_pos=t_list[np.logical_and(t_list>=0,t_list<4000)]
A_list_pos=A_list[np.logical_and(t_list>=0,t_list<4000)]
mean_list_pos=mean_list[np.logical_and(t_list>=0,t_list<4000)]
std_list_pos=std_list[np.logical_and(t_list>=0,t_list<4000)]

plt.figure()
plt.hist(t_list_pos,100,normed=1)
plt.figure()
plt.hist(A_list_pos,100,normed=1)
plt.show()

datadict={}
datadict['t_list']=t_list
datadict['A_list']=A_list
datadict['mean_list']=mean_list
datadict['std_list']=std_list
df=pd.DataFrame(datadict)
df.to_csv('500.csv')
