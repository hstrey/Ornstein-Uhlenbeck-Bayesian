import langevin_cached_model as lcm
import pandas as pd
import numpy as np
import os

data_dir='results/delta10-2/data1/'
data_file='data.csv'
N=100
delta_t=0.01

data=pd.read_csv(data_dir+data_file)
data_length=len(data)

# initial prior
mean_A=1.0
std_A=40.0
mean_D=1.0
std_D=10.0

#lists for data storage
mA,sA,mD,sD = [mean_A],[std_A],[mean_D],[std_D]
# compile Stan model for reuse
sm = lcm.Langevin()

for i in range(int(data_length/N)):

    x=data[i*N : (i+1)*N]

    trace = sm.run(x=x,
                   mu_D=mean_D,
                   sd_D=std_D,
                   mu_A=mean_A,
                   sd_A=std_A,
                   delta_t=delta_t,
                   N=N)

    A = trace['A']
    D = trace['D']

    # save the data
    tracedict = {}
    tracedict['D'] = D
    tracedict['A'] = A

    tdf = pd.DataFrame(tracedict)
    tdf.to_csv(data_dir + 'trace'+str(N)+'_'+ str(i) + '.csv', index=False)

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

resultdict={ 'mean_A' : np.array(mA),
             'std_A' : np.array(sA),
             'mean_D' : np.array(mD),
             'std_D' : np.array(sD),
             }

df=pd.DataFrame(resultdict)
df.to_csv(data_dir+'results'+str(N)+'.csv',index=False)
