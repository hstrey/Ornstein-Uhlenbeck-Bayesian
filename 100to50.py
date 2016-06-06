import numpy as np
import matplotlib.pyplot as plt, seaborn as sns
import lmfit as lm
import pandas as pd
import glob

N=50
source_dir="results/data100/"
dest_dir="results/data50/"

data=pd.read_csv(source_dir+"data.csv")

data_length=len(data)

for i in range(int(data_length/N)):
    data_slice=data[i*N : (i+1)*N]
    data_slice.to_csv(dest_dir+"data"+str(i)+".csv",index=False)
    print(i)



