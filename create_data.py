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

datadir='results/delta10-2/data2/'

N=10000 # length of data set

P=1000 # range to fit acf

x = langevin.time_series(A=A,D=D,delta_t=delta_t,N=N)

# calculate autocorrelation function
f = np.fft.rfft(x)
acf = np.fft.irfft(f * np.conjugate(f))
acf = np.fft.fftshift(acf) / N
autocorr=acf[int(N/2):]

y = autocorr[:min(int(N/2),P)]
t = np.arange(min(int(N/2),P))

mod=ExponentialModel()
pars = mod.guess(y, x=t)
out  = mod.fit(y, pars, x=t)
print(out.fit_report(min_correl=0.25))

plt.figure()
plt.plot(t,y,"o")
plt.plot(t,out.best_fit)
plt.title('acf')

plt.show()

df=pd.DataFrame({'x':x})
df.to_csv(datadir+'data.csv',index=False)
