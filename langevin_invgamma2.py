import langevin_cached_model as lcm
import pandas as pd
import numpy as np
import argparse
import lmfit as lm
from scipy.stats import gamma

def mygamma(x,alpha, beta):
    return gamma.pdf(x,alpha, scale=1/beta)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', action='store', default="./",
                        help='data directory')
    parser.add_argument('-f', '--datafile', action='store', default="data.csv",
                        help='data filename')
    parser.add_argument('-n', '--datasets', action='store', type=int, default=100,
                        help='length of datasets')
    parser.add_argument('-t', '--timestep', action='store', type=float, default=0.01,
                        help='timestep')
    parser.add_argument('-s', '--samples', action='store', type=int, default=10000,
                        help='MCMC samples per run')

    arg = parser.parse_args()

    data_dir=arg.dir
    data_file=arg.datafile
    N=arg.datasets
    delta_t=arg.timestep

    data=pd.read_csv(data_dir+data_file)
    data_length=len(data)

    # initial prior
    # both D and A have mean 1 and std 10
    alpha_A1=0.01
    beta_A1=0.01
    alpha_A2=0.01
    beta_A2=0.01
    alpha_D=2.01
    beta_D=1.01

    #lists for data storage
    mA1,sA1,mD,sD = [alpha_A1/beta_A1],[np.sqrt(alpha_A1/beta_A1**2)],[beta_D/(alpha_D-1.0)],[np.sqrt(beta_D**2/(alpha_D-1.0)**2/(alpha_D-2.0))]
    mA2,sA2 = [alpha_A2/beta_A2],[np.sqrt(alpha_A2/beta_A2**2)]
    aA1,bA1,aA2, bA2, aD,bD = [alpha_A1],[beta_A1],[alpha_A2],[beta_A2],[alpha_D],[beta_D]

    gModel = lm.Model(mygamma)

    # compile model for reuse
    sm = lcm.LangevinIG2()
    sm.samples=arg.samples

    for i in range(int(data_length/N)):

        x=data[i*N : (i+1)*N]
        x1=np.array(x['x1'])+np.array(x['x2'])
        x2=np.array(x['x1'])-np.array(x['x2'])

        trace = sm.run(x1=x1,
                       x2=x2,
                       aD=alpha_D,
                       bD=beta_D,
                       aA1=alpha_A1,
                       bA1=beta_A1,
                       aA2=alpha_A1,
                       bA2=beta_A1,
                       delta_t=delta_t,
                       N=N)

        A1 = trace['A1']
        A2 = trace['A2']
        D = trace['D']

        # save the data
        tracedict = {}
        tracedict['D'] = D
        tracedict['A1'] = A1
        tracedict['A2'] = A2

        tdf = pd.DataFrame(tracedict)
        tdf.to_csv(data_dir + 'trace_IG2_G'+str(N)+'_'+ str(i) + '.csv', index=False)

        mean_D=D.mean()
        std_D=D.std()
        mD.append(mean_D)
        sD.append(std_D)
        print('mean_D: ',mean_D,'std_D: ',std_D)

        alpha_D = (mean_D ** 2 / std_D ** 2) + 2
        beta_D = mean_D * (alpha_D - 1)
        aD.append(alpha_D)
        bD.append(beta_D)

        mean_A1=A1.mean()
        std_A1=A1.std()
        mA1.append(mean_A1)
        sA1.append(std_A1)
        print('mean_A1: ',mean_A1,'std_A: ',std_A1)

        alpha_A1 = (mean_A1 ** 2 / std_A1 ** 2)
        beta_A1 = alpha_A1/mean_A1

        mean_A2=A2.mean()
        std_A2=A2.std()
        mA2.append(mean_A2)
        sA2.append(std_A2)
        print('mean_A2: ',mean_A2,'std_A2: ',std_A2)

        alpha_A2 = (mean_A2 ** 2 / std_A2 ** 2)
        beta_A2 = alpha_A2/mean_A2
        # hist, bin_edges = np.histogram(A, bins='auto', density=True)
        # delta = bin_edges[1] - bin_edges[0]
        # x = bin_edges[:-1] + delta / 2
        #
        # result = gModel.fit(hist, x=x, alpha=alpha_A, beta=beta_A)
        # print(result.fit_report())
        #
        #alpha_A = result.best_values['alpha']
        #beta_A = result.best_values['beta']

        aA1.append(alpha_A1)
        bA1.append(beta_A1)
        aA2.append(alpha_A2)
        bA2.append(beta_A2)

    resultdict={ 'mean_A1' : np.array(mA1),
                 'std_A1' : np.array(sA1),
                 'mean_A2' : np.array(mA2),
                 'std_A2' : np.array(sA2),
                 'mean_D' : np.array(mD),
                 'std_D' : np.array(sD),
                 'alpha_A1' : np.array(aA1),
                 'beta_A1' : np.array(bA1),
                 'alpha_A2' : np.array(aA2),
                 'beta_A2' : np.array(bA2),
                 'alpha_D' : np.array(aD),
                 'beta_D' : np.array(bD),
                 }

    df=pd.DataFrame(resultdict)
    df.to_csv(data_dir+'resultsIG2_G'+str(N)+'.csv',index=False)

if __name__ == "__main__":
    main()