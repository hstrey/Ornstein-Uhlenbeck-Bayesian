import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import langevin_cached_model as lcm
import pymc3 as pm
import scipy.io
import scipy as sp

datadir = "results/fMRI/"
region = "mpfc"
voxel = 0

oxy_data37 = scipy.io.loadmat(datadir+'OXY37_MRI_1_ts.mat')

data_region=oxy_data37[region+'_r_ts']

time_series=np.array(data_region[voxel])
plt.plot(time_series)
print("mean: ",time_series.mean())
print("var: ",time_series.std()**2)
N=len(time_series)
print("N: ",N)

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
    B = pm.Deterministic('B', pm.exp(-delta_t * D / A))

    path = lcm.Ornstein_Uhlenbeck('path', D=D, A=A, B=B, observed=time_series)

    start = pm.find_MAP(fmin=sp.optimize.fmin_powell)

    trace = pm.sample(100000, start=start)

pm.summary(trace)

data_dict={ 'D':trace['D'],
            'A':trace['A'],
            'B':trace['B'],
}

df=pd.DataFrame(data_dict)
df.to_csv(datadir+'LIG'+region+str(voxel)+'.csv', index=False)

pm.traceplot(trace)
plt.savefig(datadir+'LIG'+region+str(voxel)+'.pdf')
