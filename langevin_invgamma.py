import langevin_cached_model as lcm
import pandas as pd
import numpy as np
import os

data_dir='results/delta10-2/data1/'
data_file='data.csv'
N=50
delta_t=0.01

data=pd.read_csv(data_dir+data_file)
data_length=len(data)

# initial prior
# both D and A have mean 1 and std 10
alpha_A=2.01
beta_A=1.01
alpha_D=2.01
beta_D=1.01

#lists for data storage
mA,sA,mD,sD = [beta_A/(alpha_A-1)],[np.sqrt(beta_A**2/(alpha_A-1)**2/(alpha_A-2))],[beta_D/(alpha_D-1)],[np.sqrt(beta_D**2/(alpha_D-1)**2/(alpha_D-2))]
aA,bA,aD,bD = [alpha_A],[beta_A],[alpha_D],[beta_D]
# compile Stan model for reuse
sm = lcm.LangevinIG()

for i in range(int(data_length/N)):

    x=data[i*N : (i+1)*N]

    trace = sm.run(x=x,
                   aD=alpha_D,
                   bD=beta_D,
                   aA=alpha_A,
                   bA=beta_A,
                   delta_t=delta_t,
                   N=N)

    A = trace['A']
    D = trace['D']

    # save the data
    tracedict = {}
    tracedict['D'] = D
    tracedict['A'] = A

    tdf = pd.DataFrame(tracedict)
    tdf.to_csv(data_dir + 'trace_IG'+str(N)+'_'+ str(i) + '.csv', index=False)

    mean_D=D.mean()
    std_D=D.std()
    mD.append(mean_D)
    sD.append(std_D)
    print('mean_D: ',mean_D,'std_D: ',std_D)

    mean_A=A.mean()
    std_A=A.std()
    mA.append(mean_A)
    sA.append(std_A)
    print('mean_A: ',mean_A,'std_A: ',std_A)

    alpha_A = (mean_A ** 2 / std_A ** 2) +2
    beta_A = mean_A*(alpha_A - 1)
    aA.append(alpha_A)
    bA.append(beta_A)

    alpha_D = (mean_D ** 2 / std_D ** 2) + 2
    beta_D = mean_D * (alpha_D - 1)
    aD.append(alpha_D)
    bD.append(beta_D)

resultdict={ 'mean_A' : np.array(mA),
             'std_A' : np.array(sA),
             'mean_D' : np.array(mD),
             'std_D' : np.array(sD),
             'alpha_A' : np.array(aA),
             'beta_A' : np.array(bA),
             'alpha_D' : np.array(aD),
             'beta_D' : np.array(bD),
             }

df=pd.DataFrame(resultdict)
df.to_csv(data_dir+'resultsIG'+str(N)+'.csv',index=False)
