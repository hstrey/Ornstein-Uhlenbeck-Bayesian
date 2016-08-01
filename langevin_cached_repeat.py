import langevin_cached_model as lcm
import pandas as pd
import numpy as np
import os

data_dir='results/data50/'

filelist=os.listdir(data_dir)

data_list=[]

for file in filelist:
    if file.startswith('data') and file.endswith('.csv'):
        data_list.append(file)

print(data_list)

# initial prior
mean_A=1.0
std_A=40.0
mean_D=1.0
std_D=10.0

#lists for data storage
mA,sA,mD,sD = [mean_A],[std_A],[mean_D],[std_D]
# compile Stan model for reuse
sm = lcm.Langevin()

for i,file in enumerate(data_list):
    x = pd.read_csv(data_dir+file)

    trace = sm.run(x=x,
                   mu_D=mean_D,
                   sd_D=std_D,
                   mu_A=mean_A,
                   sd_A=std_A,
                   delta_t=0.01)

    A = trace['A']
    D = trace['D']

    # save the data
    tracedict = {}
    tracedict['D'] = D
    tracedict['A'] = A

    tdf = pd.DataFrame(tracedict)
    tdf.to_csv(data_dir + 'trace_pymc3_' + str(i) + '.csv', index=False)

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
df.to_csv(data_dir+'pymc3_results.csv',index=False)
