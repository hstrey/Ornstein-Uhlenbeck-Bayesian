import numpy as np
import matplotlib.pyplot as plt, seaborn as sns
import lmfit as lm
from scipy import signal
from lmfit.models import ExponentialModel
import pandas as pd
from itertools import accumulate
import langevin

A1 = 1.0
A2 = 0.5
D = 1.0
delta_t=0.01

datadir='results/delta10-2/corr2/'

N=10000 # length of data set

P=1000 # range to fit acf

x1 = langevin.time_series(A1,D,delta_t,N)
x2 = langevin.time_series(A2,D,delta_t,N)

xx1 = (x1+x2)/2.0
xx2 = (x1-x2)/2.0

df = pd.DataFrame({'x1':xx1, 'x2':xx2})
df.to_csv(datadir+'data.csv',index=False)
