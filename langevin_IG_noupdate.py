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
                        help='number of datasets')
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
    # D has mean 1 and std 10
    # A has mean 1 and std 10
    alpha_A=0.01
    beta_A=0.01
    alpha_D=2.01
    beta_D=1.01

    #lists for data storage
    mA,sA,mD,sD = [alpha_A/beta_A],[np.sqrt(alpha_A/beta_A**2)],[beta_D/(alpha_D-1.0)],[np.sqrt(beta_D**2/(alpha_D-1.0)**2/(alpha_D-2.0))]
    aA,bA,aD,bD = [alpha_A],[beta_A],[alpha_D],[beta_D]

    gModel = lm.Model(mygamma)

    # compile model for reuse
    sm = lcm.LangevinIG()
    sm.samples=arg.samples

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
        tdf.to_csv(data_dir + 'trace_IGnu_G'+str(N)+'_'+ str(i) + '.csv', index=False)

        mean_D=D.mean()
        std_D=D.std()
        mD.append(mean_D)
        sD.append(std_D)
        print('mean_D: ',mean_D,'std_D: ',std_D)

#        alpha_D = (mean_D ** 2 / std_D ** 2) + 2
#        beta_D = mean_D * (alpha_D - 1)
        aD.append((mean_D ** 2 / std_D ** 2) + 2)
        bD.append(mean_D * (alpha_D - 1))

        mean_A=A.mean()
        std_A=A.std()
        mA.append(mean_A)
        sA.append(std_A)
        print('mean_A: ',mean_A,'std_A: ',std_A)

#        alpha_A = (mean_A ** 2 / std_A ** 2)
#        beta_A = alpha_A/mean_A

        # hist, bin_edges = np.histogram(A, bins='auto', density=True)
        # delta = bin_edges[1] - bin_edges[0]
        # x = bin_edges[:-1] + delta / 2
        #
        # result = gModel.fit(hist, x=x, alpha=alpha_A, beta=beta_A)
        # print(result.fit_report())
        #
        #alpha_A = result.best_values['alpha']
        #beta_A = result.best_values['beta']

        aA.append((mean_A ** 2 / std_A ** 2))
        bA.append(alpha_A/mean_A)


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
    df.to_csv(data_dir+'resultsIGnu_G'+str(N)+'.csv',index=False)

if __name__ == "__main__":
    main()