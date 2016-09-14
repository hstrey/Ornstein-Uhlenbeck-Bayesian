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
import langevin

A,D = 1.0,1.0
delta_t=0.01

N=1000
M=10000
t_list=[]
tstd_list=[]
A_list=[]
Astd_list=[]
mean_list=[]
std_list=[]
mod = ExponentialModel()
acf_avg=np.zeros(int(N/2))
acf_var=np.zeros(int(N/2))
for i in range(M):
    # random force
    w=np.random.normal(0,1,N)
    x = langevin.time_series(A=A,D=D,delta_t=delta_t,N=N)

    # see http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.signal.fftconvolve.html
    # autocorr = signal.fftconvolve(x, x[::-1], mode='full')
    f = np.fft.rfft(x)
    acf = np.fft.irfft(f * np.conjugate(f))
    acf = np.fft.fftshift(acf) / N
    autocorr=acf[int(N/2):]
#    n=len(autocorr)
#    autocorr=autocorr[int((n-1)/2):]*2.0/(n+1)
    acf_avg=acf_avg+autocorr
    acf_var=acf_var+autocorr**2
    y = autocorr[:min(int(N/2),1000)]
    t = np.arange(min(int(N/2),1000))

    out  = mod.fit(y, amplitude=1.0, decay=100.0, x=t)
    #print(out.fit_report(min_correl=0.25))
    t_list.append(out.values['decay'])
    tstd_list.append(out.covar[0,0])
    A_list.append(out.values['amplitude'])
    Astd_list.append(out.covar[1,1])

    mean_list.append(x.mean())
    std_list.append(x.std())
    print('mean: ',x.mean(),'std: ',x.std(),'amplitude: ',out.values['amplitude'],'decay: ',out.values['decay'])

acf_avg=acf_avg/M
acf_stderr=np.sqrt((acf_var/M-(acf_avg/M)**2)/M)
y = acf_avg
dy=acf_stderr
t = np.arange(int(N/2))

out  = mod.fit(y, amplitude=1.0, decay=100.0, x=t, weights=1./dy)
print(out.fit_report(min_correl=0.25))
print('covar[0,0]: ',out.covar[0,0],'covar[1,1]: ',out.covar[1,1])

plt.figure()
plt.errorbar(t,y,yerr=dy,fmt="ob")
plt.plot(t,out.best_fit)
plt.title('acf')

plt.figure()
plt.plot(t,dy)
plt.title('acf stddev')

t_list=np.array(t_list)
A_list=np.array(A_list)
tstd_list=np.array(tstd_list)
Astd_list=np.array(Astd_list)
mean_list=np.array(mean_list)
std_list=np.array(std_list)

#eliminate outliers e.g. negative decay and very long decay
t_list_pos=t_list[np.logical_and(t_list>=0,t_list<1000)]
A_list_pos=A_list[np.logical_and(t_list>=0,t_list<1000)]

tstd_list_pos=np.sqrt(tstd_list[np.logical_and(t_list>=0,t_list<1000)])
Astd_list_pos=np.sqrt(Astd_list[np.logical_and(t_list>=0,t_list<1000)])

mean_list_pos=mean_list[np.logical_and(t_list>=0,t_list<1000)]
std_list_pos=std_list[np.logical_and(t_list>=0,t_list<1000)]

# careful, I am overwriting gamma
from scipy.stats import gamma

# calculate diffusion coefficient from tau and amplitude
D=A_list_pos/t_list_pos/delta_t
D=D[np.logical_and(D>=0,D<1000)] # remove outliers
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
plt.hist(D,bins='auto',normed=1)

plt.figure()
plt.title('tau histogramm')
plt.hist(t_list_pos,bins='auto', normed=True)
plt.plot(xgt,g_tau)

plt.figure()
plt.title('amplitude histogramm')
plt.hist(A_list_pos,bins='auto', normed=True)
plt.plot(xgA,g_A)

plt.figure()
plt.title('tau std histogramm')
plt.hist(tstd_list_pos, bins='auto', normed=True)

plt.figure()
plt.title('amplitude std histogramm')
plt.hist(Astd_list_pos,bins='auto',normed=1)
plt.show()

datadict=dict(t=t_list,
              t_std=tstd_list,
              A=A_list,
              A_std=Astd_list,
              mean=mean_list,
              std=std_list)
df=pd.DataFrame(datadict)
df.to_csv(str(N)+'x'+str(M)+'.csv',index=False)

acf_dict=dict(acf=acf_avg,
              acf_stderr=acf_stderr)
df=pd.DataFrame(acf_dict)
df.to_csv('acf_'+str(N)+'x'+str(M)+'.csv',index=False)
