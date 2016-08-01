import pandas as pd
import numpy as np
import os
import sys
import langevin
import pystan
import matplotlib.pyplot as plt

langevin_code = """
data {
int<lower=0> N;
real<lower=0> alpha_D;
real<lower=0> beta_D;
real<lower=0> alpha_A;
real<lower=0> beta_A;
vector[N] y;
}
parameters {
real<lower=0> D;
real<lower=0> A;
}
model {
D ~ gamma(alpha_D,beta_D);
A ~ gamma(alpha_A, beta_A);
y[1] ~ normal(0,sqrt(A));
for (n in 2:N)
    y[n] ~ normal(y[n-1]*exp(-0.01*D/A), sqrt(A*(1-exp(-0.02*D/A))));
}
"""

data_dir='results/data50/'

filelist=os.listdir(data_dir)

data_list=[]

for file in filelist:
    if file.startswith('data') and file.endswith('.csv'):
        data_list.append(file)

print(data_list)

# initial prior
alpha_A=0.0025
alpha_D=0.01
beta_A=0.0025
beta_D=0.01

#lists for data storage
mA,sA,mD,sD = [],[],[],[]

# compile Stan model for reuse
sm = pystan.StanModel(model_code=langevin_code)

for i,file in enumerate(data_list):
    y = pd.read_csv(data_dir+file)
    N = len(y['x'])

    langevin_dat = {'N': N,
                    'alpha_A' : alpha_A,
                    'beta_A' : beta_A,
                    'alpha_D' : alpha_D,
                    'beta_D' : beta_D,
                    'y' : np.array(y['x'])}

    fit = sm.sampling(data=langevin_dat, iter=10000, chains=4)

    la = fit.extract(permuted=True)
    A = la['A']
    D = la['D']

    # save the data
    tracedict = {}
    tracedict['D'] = D
    tracedict['A'] = A

    tdf = pd.DataFrame(tracedict)
    tdf.to_csv(data_dir + 'trace' + str(i) + '.csv', index=False)

    mean_D=D.mean()
    std_D=D.std()
    mD.append(mean_D)
    sD.append(std_D)
    print('mean_D: ',mean_D,'std_D: ',std_D)
    beta_D=mean_D/std_D**2
    alpha_D=mean_D*beta_D
    print('alpha_D: ',alpha_D,'beta_D: ',beta_D)

    mean_A=A.mean()
    std_A=A.std()
    mA.append(mean_A)
    sA.append(std_A)
    print('mean_A: ',mean_A,'std_A: ',std_A)
    beta_A=mean_A/std_A**2
    alpha_A=mean_A*beta_A
    print('alpha_A: ',alpha_A,'beta_A: ',beta_A)

resultdict={ 'mean_A' : np.array(mA),
             'std_A' : np.array(sA),
             'mean_D' : np.array(mD),
             'std_D' : np.array(sD),
             }

df=pd.DataFrame(resultdict)
df.to_csv(data_dir+'stan_results3.csv',index=False)

