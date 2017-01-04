import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import langevin_cached_model as lcm
import pymc3 as pm
import langevin
import lmfit as lm
from lmfit.models import ExponentialModel

A,D = 1.0,1.0
delta_t=0.01
M=1000 # number of data sets
N=10000 # length of data set
P=1000 # range to fit acf

# initial prior
# both D and A have mean 1 and std 10
alpha_A=0.01
beta_A=0.01
alpha_D=2.01
beta_D=1.01

# compile model for reuse
sm = lcm.LangevinIG()
sm.samples=10000

result_array = None

for i in range(M):
    data = langevin.time_series(A=A, D=D, delta_t=delta_t, N=N)
    # calculate autocorrelation function
    f = np.fft.rfft(data)
    acf = np.fft.irfft(f * np.conjugate(f))
    acf = np.fft.fftshift(acf) / N
    autocorr = acf[int(N / 2):]

    y = autocorr[:min(int(N / 2), P)]
    t = np.arange(min(int(N / 2), P))

    mod = ExponentialModel()
    pars = mod.guess(y, x=t)
    out = mod.fit(y, pars, x=t)

    fit_results = np.array([out.values['decay']*delta_t,
                            np.sqrt(out.covar[0,0])*delta_t,
                            out.values['amplitude'],
                            np.sqrt(out.covar[1,1])])

    print(out.fit_report(min_correl=0.25))

    trace = sm.run(x=data,
                    aD=alpha_D,
                    bD=beta_D,
                    aA=alpha_A,
                    bA=beta_A,
                    delta_t=0.01,
                    N=10000)

    pm.summary(trace)

    traceD_results = np.percentile(trace['D'],(2.5,25,50,75,97.5))
    traceD_results = np.concatenate((traceD_results, [np.std(trace['D'])], [np.mean(trace['D'])]))

    traceA_results=np.percentile(trace['A'],(2.5,25,50,75,97.5))
    traceA_results = np.concatenate((traceA_results, [np.std(trace['A'])], [np.mean(trace['A'])]))

    results = np.concatenate((fit_results, traceD_results, traceA_results))

    print(results)

    if result_array is None:
        result_array = results
    else:
        result_array = np.vstack((result_array, results))

print(np.mean(result_array, axis=0))

columns = ['decay',
           'decay_std',
           'amplitude',
           'amplitude_std',
           'D2.5',
           'D25',
           'D50',
           'D75',
           'D97.5',
           'Dstd',
           'Dmean',
           'A2.5',
           'A25',
           'A50',
           'A75',
           'A97.5',
           'Astd',
           'Amean',
           ]

df = pd.DataFrame(data=result_array, columns=columns)

df.to_csv('results/BM_'+str(delta_t)+'.csv',index=False)
