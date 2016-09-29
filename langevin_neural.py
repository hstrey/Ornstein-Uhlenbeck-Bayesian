import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import langevin_cached_model as lcm
import pymc3 as pm
import scipy.io
import scipy as sp

oxy_data37=scipy.io.loadmat('OXY37_MRI_1_ts.mat')

mpfc_r_ts=oxy_data37['mpfc_r_ts']

mpfc=np.array(mpfc_r_ts[0])
plt.plot(mpfc)
print("mean: ",mpfc.mean())
print("var: ",mpfc.std()**2)
N=len(mpfc)
print("N: ",N)
mpfc.shape

# initial prior
# both D and A have mean 1 and std 10
alpha_A=400.0/16.0
beta_A=1.0/16.0
alpha_N=400.0/16.0
beta_N=1.0/16.0
alpha_D=2.0+1.0/1.6
beta_D=100*(alpha_D-1)
delta_t=0.802

with pm.Model() as model:
    D = pm.InverseGamma('D', alpha=alpha_D, beta=beta_D)
    A = pm.Gamma('A', alpha=alpha_A, beta=beta_A)
    sN = pm.InverseGamma('sN', alpha=alpha_N, beta=beta_N)

    B = pm.Deterministic('B', pm.exp(-delta_t * D / A))

    path = lcm.Ornstein_Uhlenbeck('path', D=D, A=A, B=B, shape=mpfc.shape)

    X_obs = pm.Normal('X_obs', mu=path, sd=sN, observed=mpfc)

    start = pm.find_MAP(fmin=sp.optimize.fmin_powell)

    trace = pm.sample(10000, start=start)

pm.summary(trace)

pm.traceplot(trace)

plt.show()