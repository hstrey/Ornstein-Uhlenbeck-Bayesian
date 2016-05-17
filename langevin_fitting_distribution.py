# here we simulate a brownian particle in a harmonic potential
# the question is: if you only have access to a short time sequence
# how accurately can you say something about the parameters of the system
# Ultimately, we would like to compare Bayesian methods and standard
# approaches, such as the analysis of correlation functions
#
# it looks that calculating correlation functions does not give correct amplitudes
# decay times when the observed time sequence is shorter than a few relaxation times
#
import numpy as np
import matplotlib.pyplot as plt, seaborn as sns
import lmfit as lm
from scipy import signal
from lmfit.models import ExponentialModel
import pandas as pd
from itertools import accumulate

k,gamma,D = 1.0,1.0,1.0
delta_t=0.01
ampl = np.sqrt(2*D*delta_t)
a=k/gamma*delta_t

# differential equation x_i = x_(i-1) - k/gamma*x_(i-1) + sqrt(2*D*delta_t)*w_i
def next_point_euler(x,y):
    return x - a*x + ampl*y
    
# differential equation x_i = x_(i-1) - k/gamma*x_(i-1) + sqrt(2*D*delta_t)*w_i
# using 4th-order Runge-Kutta
def next_point_RK4(x,y):
    k0=-a*x
    k1=-a*(x+k0/2.0)
    k2=-a*(x+k1/2.0)
    k3=-a*(x+k2)
    return x + (k0+2*k1+2*k2+k3)/6 + ampl*y

N=500
M=10000
t_list=[]
A_list=[]
mean_list=[]
std_list=[]
mod = ExponentialModel()
acf_avg=np.zeros(int(N/2))
acf_std=np.zeros(int(N/2))
for i in range(M):
    # random force
    w=np.random.normal(0,1,N)
    x = np.fromiter(accumulate(w, next_point_euler),np.float)

    # see http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.signal.fftconvolve.html
    # autocorr = signal.fftconvolve(x, x[::-1], mode='full')
    f = np.fft.rfft(x)
    acf = np.fft.irfft(f * np.conjugate(f))
    acf = np.fft.fftshift(acf) / N
    autocorr=acf[int(N/2):]
#    n=len(autocorr)
#    autocorr=autocorr[int((n-1)/2):]*2.0/(n+1)
    acf_avg=acf_avg+autocorr
    acf_std=acf_std+autocorr**2
    y = autocorr[:100]
    t = np.arange(100)

    pars = mod.guess(y, x=t)
    out  = mod.fit(y, pars, x=t)
    #print(out.fit_report(min_correl=0.25))
    t_list.append(out.values['decay'])
    A_list.append(out.values['amplitude'])
    mean_list.append(x.mean())
    std_list.append(x.std())
    print('mean: ',x.mean(),'std: ',x.std(),'amplitude: ',out.values['amplitude'],'decay: ',out.values['decay'])

acf_avg=acf_avg/M
acf_stderr=np.sqrt((acf_std/M-(acf_avg/M)**2)/M)
y = acf_avg
dy=acf_stderr
t = np.arange(int(N/2))

pars = mod.guess(y, x=t)
out  = mod.fit(y, pars, x=t, weights=1./dy)
print(out.fit_report(min_correl=0.25))

plt.figure()
plt.errorbar(t,y,yerr=dy,fmt="ob")
plt.plot(t,out.best_fit)
plt.title('acf')

plt.figure()
plt.plot(t,dy)
plt.title('acf stddev')

t_list=np.array(t_list)
A_list=np.array(A_list)
mean_list=np.array(mean_list)
std_list=np.array(std_list)

#eliminate outliers e.g. negative decay and very long decay
t_list_pos=t_list[np.logical_and(t_list>=0,t_list<4000)]
A_list_pos=A_list[np.logical_and(t_list>=0,t_list<4000)]
mean_list_pos=mean_list[np.logical_and(t_list>=0,t_list<4000)]
std_list_pos=std_list[np.logical_and(t_list>=0,t_list<4000)]

# careful, I am overwriting gamma
from scipy.stats import gamma

# calculate diffusion coefficient from tau and amplitude
D=A_list_pos/t_list_pos/delta_t
mean_D=D.mean()
std_D=D.std()
print('D mean: ',mean_D,'std: ',std_D)
scale_D=std_D**2/mean_D
alpha_D=mean_D/scale_D
print('D alpha: ',alpha_D,'scale: ',scale_D)

xgt=np.linspace(0,D.max(),200)
g_tau=gamma.pdf(xgt,alpha_D,scale=scale_D)

mean_t=t_list_pos.mean()
std_t=t_list_pos.std()
print('tau mean: ',mean_t,'std: ',std_t)
scale_t=std_t**2/mean_t
alpha_t=mean_t/scale_t
print('tau alpha: ',alpha_t,'scale: ',scale_t)

xgt=np.linspace(0,t_list_pos.max(),200)
g_tau=gamma.pdf(xgt,alpha_t,scale=scale_t)

mean_A=A_list_pos.mean()
std_A=A_list_pos.std()
print('ampl mean: ',mean_A,'std: ',std_A)
scale_A=std_A**2/mean_A
alpha_A=mean_A/scale_A
print('ampl alpha: ',alpha_A,'scale: ',scale_A)

xgA=np.linspace(0,A_list_pos.max(),200)
g_A=gamma.pdf(xgA,alpha_A,scale=scale_A)

plt.figure()
plt.title('D histogramm')
plt.hist(D,100,normed=1)

plt.figure()
plt.title('tau histogramm')
plt.hist(t_list_pos,100,normed=1)
plt.plot(xgt,g_tau)

plt.figure()
plt.title('amplitude histogramm')
plt.hist(A_list_pos,100,normed=1)
plt.plot(xgA,g_A)
plt.show()

datadict={}
datadict['t_list']=t_list
datadict['A_list']=A_list
datadict['mean_list']=mean_list
datadict['std_list']=std_list
df=pd.DataFrame(datadict)
df.to_csv('5000.csv')

acf_dict={}
acf_dict['acf']=acf_avg
acf_dict['acf_stderr']=acf_stderr
df=pd.DataFrame(acf_dict)
df.to_csv('acf_5000.csv')
