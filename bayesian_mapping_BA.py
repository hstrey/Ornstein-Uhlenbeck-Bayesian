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
P=500 # range to fit acf

# initial prior
# A have mean 1 and std 10
# B is uniform prior over [0,1] interval
alpha_A=2.1
beta_A=1.1
alpha_B=1
beta_B=1

# compile model for reuse
sm = lcm.OU_BA()
sm.samples=100000

result_array = None

for i in range(M):
    print("***** Iteration ",i," *****")
    data = langevin.time_series(A=A, D=D, delta_t=delta_t, N=N)

    data_results = [data[0]**2, data[-1]**2, np.sum(data[1:-2]**2), np.sum(data[:-1]*data[1:])]

    # calculate autocorrelation function
    f = np.fft.rfft(data)
    acf = np.fft.irfft(f * np.conjugate(f))
    acf = np.fft.fftshift(acf) / N
    autocorr = acf[int(N / 2):]

    y = autocorr[:min(int(N / 2), P)]
    t = np.arange(min(int(N / 2), P))

    mod = ExponentialModel()
    pars = mod.guess(y, x=t)
    try:
        out = mod.fit(y, pars, x=t)
    except:
        fit_results = np.zeros(4)
        print('fit did not work')
    else:
        fit_results = np.array([out.values['decay']*delta_t,
                            np.sqrt(out.covar[0,0])*delta_t,
                            out.values['amplitude'],
                            np.sqrt(out.covar[1,1])])
        print(out.fit_report(min_correl=0.25))

    trace = sm.run(x=data,
                    aB=alpha_B,
                    bB=beta_B,
                    aA=alpha_A,
                    bA=beta_A,
                    delta_t=delta_t,
                    N=N)

    pm.summary(trace)

    traceB_results = np.percentile(trace['B'],(2.5,25,50,75,97.5))
    traceB_results = np.concatenate((traceB_results, [np.std(trace['B'])], [np.mean(trace['B'])]))

    traceA_results=np.percentile(trace['A'],(2.5,25,50,75,97.5))
    traceA_results = np.concatenate((traceA_results, [np.std(trace['A'])], [np.mean(trace['A'])]))

    results = np.concatenate((data_results, fit_results, traceB_results, traceA_results))

    print(results)

    if result_array is None:
        result_array = results
    else:
        result_array = np.vstack((result_array, results))

print(np.mean(result_array, axis=0))

columns = ['data1sq',
           'dataNsq',
           'datasq',
           'datacorr',
           'decay',
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

df.to_csv('results/BMBA_'+str(delta_t)+'_'+str(N)+'.csv',index=False)
