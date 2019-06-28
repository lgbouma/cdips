import os
from glob import glob
import numpy as np, pandas as pd
from cdips.utils import today_YYYYMMDD

def main():

    paramglob = ("/home/lbouma/proj/cdips/results/fit_gold"
                 "/sector-*/fitresults/hlsp_*gaiatwo*_llc/*fitparameters.csv")
    parampaths = glob(paramglob)

    df = pd.concat((pd.read_csv(f, sep='|') for f in parampaths))

    outpath = ("/home/lbouma/proj/cdips/results/fit_gold/"
               "{}_sector6_and_sector7_gold_mergedfitparams.csv".
               format(today_YYYYMMDD()))
    df.to_csv(outpath, index=False, sep='|')
    print('made {}'.format(outpath))

if __name__=="__main__":
    main()
