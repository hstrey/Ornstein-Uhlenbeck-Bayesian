import numpy as np
import matplotlib.pyplot as plt, seaborn as sns
import lmfit as lm
from scipy import signal
from lmfit.models import ExponentialModel
import pandas as pd
from itertools import accumulate
import langevin

k,ga,D = 1.0,1.0,1.0
delta_t=0.01

datadir='results/data100/'

N=1000 # length of individual data sets = 1 relaxation time
G=100 # simulation oversampling
M=100 # create 100 data sets

P=250 # points to fit autocorrelation function

acf_sum=np.zeros(int(N/2))
acf_sumsq=np.zeros(int(N/2))

for i in range(M):
    x = langevin.time_series(k=k,ga=ga,diff=D,delta_t=delta_t,N=N,G=G)

    # calculate autocorrelation function
    f = np.fft.rfft(x)
    acf = np.fft.irfft(f * np.conjugate(f))
    acf = np.fft.fftshift(acf) / N
    autocorr=acf[int(N/2):]

    acf_sum=acf_sum+autocorr
    acf_sumsq=acf_sumsq+autocorr**2

    print(i)

#    df=pd.DataFrame({'x':x})
#    df.to_csv(datadir+'data'+str(i)+'.csv')

acf_avg=acf_sum/M
acf_stderr=np.sqrt(acf_sumsq/M-acf_avg**2)/np.sqrt(M)

y = acf_avg[:min(int(N/2),P)]
t = np.arange(min(int(N/2),P))

mod=ExponentialModel()
pars = mod.guess(y, x=t)
out  = mod.fit(y, pars, x=t, weights=1/acf_stderr[:min(int(N/2),P)])
print(out.fit_report(min_correl=0.25))

plt.figure()
plt.errorbar(t,y,yerr=acf_stderr[:min(int(N/2),P)],fmt="ob")
plt.plot(t,out.best_fit)
plt.title('acf')

plt.show()